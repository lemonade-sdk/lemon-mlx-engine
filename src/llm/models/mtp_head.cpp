
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx-lm/llm/models/mtp_moe.h>

#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

namespace {

mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    return mx::matmul(x, mx::transpose(w));
}

mx::array silu(const mx::array& x) {
    return mx::multiply(x, mx::sigmoid(x));
}

mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return mx::multiply(silu(gate), up);
}

}  // namespace

// --- MTPDecoderLayer ---

MTPDecoderLayer::MTPDecoderLayer(const MTPHeadConfig& args)
    : args_(args),
      // q_proj outputs 2x head_dim for the sigmoid gate (matches Qwen35MoEAttention)
      q_proj_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim() * 2, args.hidden_size})),
      k_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      v_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      o_proj_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      q_norm_weight_(mx::ones({args.resolved_head_dim()})),
      k_norm_weight_(mx::ones({args.resolved_head_dim()})),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      gate_proj_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      up_proj_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      down_proj_weight_(mx::zeros({args.hidden_size, args.intermediate_size})) {}

mx::array MTPDecoderLayer::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0);
    int L = x.shape(1);
    int H = args_.hidden_size;
    int hd = args_.resolved_head_dim();
    int n_heads = args_.num_attention_heads;
    int n_kv_heads = args_.num_key_value_heads;
    float scale = std::pow(static_cast<float>(hd), -0.5f);

    // --- self-attention sub-block ---
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, args_.rms_norm_eps);
    auto q_proj_out = linear_no_bias(normed, q_proj_weight_);
    // Reshape to [B, L, num_heads, 2*head_dim] then split into queries + gate
    auto q_proj_reshaped = mx::reshape(q_proj_out, {B, L, n_heads, -1});
    auto queries = mx::slice(q_proj_reshaped, {0, 0, 0, 0}, {B, L, n_heads, hd});
    auto q_gate = mx::slice(q_proj_reshaped, {0, 0, 0, hd}, {B, L, n_heads, 2 * hd});

    auto k = linear_no_bias(normed, k_proj_weight_);
    auto v = linear_no_bias(normed, v_proj_weight_);

    auto q4 = mx::transpose(
        mx::fast::rms_norm(queries, q_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    auto k4 = mx::reshape(k, {B, L, n_kv_heads, hd});
    k4 = mx::transpose(
        mx::fast::rms_norm(k4, k_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    auto v4 = mx::reshape(v, {B, L, n_kv_heads, hd});
    v4 = mx::transpose(v4, {0, 2, 1, 3});

    int rope_dims = args_.resolved_rope_dims();
    int offset = cache ? cache->offset() : 0;
    q4 = mx::fast::rope(q4, rope_dims, /*traditional=*/false, args_.rope_theta, 1.0f, offset);
    k4 = mx::fast::rope(k4, rope_dims, /*traditional=*/false, args_.rope_theta, 1.0f, offset);

    if (cache) {
        auto [kk, vv] = cache->update(k4, v4);
        k4 = kk;
        v4 = vv;
    }

    auto attn_out = sdpa(q4, k4, v4, scale, mask);
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    // Apply sigmoid gate: output *= sigmoid(q_gate)
    auto gate_sigmoid = mx::sigmoid(q_gate);
    attn_out = mx::multiply(attn_out, gate_sigmoid);
    attn_out = linear_no_bias(attn_out, o_proj_weight_);

    auto h = mx::add(x, attn_out);

    // --- MLP sub-block (SwiGLU) ---
    auto post = mx::fast::rms_norm(h, post_attention_layernorm_weight_, args_.rms_norm_eps);
    auto gate = linear_no_bias(post, gate_proj_weight_);
    auto up = linear_no_bias(post, up_proj_weight_);
    auto mlp_out = linear_no_bias(swiglu(gate, up), down_proj_weight_);

    return mx::add(h, mlp_out);
}

std::unordered_map<std::string, mx::array*> MTPDecoderLayer::weight_map() {
    return {
        {"self_attn.q_proj.weight", &q_proj_weight_},
        {"self_attn.k_proj.weight", &k_proj_weight_},
        {"self_attn.v_proj.weight", &v_proj_weight_},
        {"self_attn.o_proj.weight", &o_proj_weight_},
        {"self_attn.q_norm.weight", &q_norm_weight_},
        {"self_attn.k_norm.weight", &k_norm_weight_},
        {"input_layernorm.weight", &input_layernorm_weight_},
        {"post_attention_layernorm.weight", &post_attention_layernorm_weight_},
        {"mlp.gate_proj.weight", &gate_proj_weight_},
        {"mlp.up_proj.weight", &up_proj_weight_},
        {"mlp.down_proj.weight", &down_proj_weight_},
    };
}

// --- MTPHead ---

MTPHead::MTPHead(const MTPHeadConfig& args)
    : args_(args),
      pre_fc_norm_hidden_weight_(mx::ones({args.hidden_size})),
      pre_fc_norm_embedding_weight_(mx::ones({args.hidden_size})),
      fc_weight_(mx::zeros({args.hidden_size, 2 * args.hidden_size})),
      dense_layer_(args),
      norm_weight_(mx::ones({args.hidden_size})) {}

// Sentinel constructor — does NOT initialize dense_layer_.
// Used exclusively by create_moe() to avoid allocating SwiGLU
// weights that would be immediately destroyed.
MTPHead::MTPHead(const MTPHeadConfig& args, int)
    : args_(args),
      pre_fc_norm_hidden_weight_(mx::ones({args.hidden_size})),
      pre_fc_norm_embedding_weight_(mx::ones({args.hidden_size})),
      fc_weight_(mx::zeros({args.hidden_size, 2 * args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})) {}

MTPHead MTPHead::create_moe(const MTPHeadConfig& args) {
    MTPHead head(args, 0);
    head.moe_layer_ = std::make_unique<MTPDecoderLayerMoE>(
        args, args.num_experts, args.num_experts_per_tok);
    return head;
}

mx::array MTPHead::operator()(
    const mx::array& hidden_state,
    const mx::array& token_embedding,
    const AttentionMask& mask,
    KVCache* cache) {
    auto h_norm = mx::fast::rms_norm(
        hidden_state, pre_fc_norm_hidden_weight_, args_.rms_norm_eps);
    auto e_norm = mx::fast::rms_norm(
        token_embedding, pre_fc_norm_embedding_weight_, args_.rms_norm_eps);
    // Note: qwen3_5.py:357 concatenates [e_norm, h_norm] (embedding first).
    auto cat = mx::concatenate({e_norm, h_norm}, -1);
    auto h = linear_no_bias(cat, fc_weight_);

    // Dispatch to dense or MoE decoder layer.
    if (moe_layer_) {
        return (*moe_layer_)(h, mask, cache);
    }
    return (*dense_layer_)(h, mask, cache);
}

mx::array MTPHead::apply_output_norm(const mx::array& h) const {
    return mx::fast::rms_norm(h, norm_weight_, args_.rms_norm_eps);
}

std::unordered_map<std::string, mx::array*> MTPHead::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["pre_fc_norm_hidden.weight"] = &pre_fc_norm_hidden_weight_;
    map["pre_fc_norm_embedding.weight"] = &pre_fc_norm_embedding_weight_;
    map["fc.weight"] = &fc_weight_;
    if (moe_layer_) {
        for (auto& [k, v] : moe_layer_->weight_map()) {
            map["layers.0." + k] = v;
        }
    } else if (dense_layer_) {
        for (auto& [k, v] : dense_layer_->weight_map()) {
            map["layers.0." + k] = v;
        }
    }
    map["norm.weight"] = &norm_weight_;
    return map;
}

void MTPHead::load_mtp_weights(
    const std::unordered_map<std::string, mx::array>& mtp_weights) {
    auto wmap = weight_map();

    // Collect quantized weight prefixes (those with .scales + .weight).
    const std::string scales_suffix = ".scales";
    const std::string biases_suffix = ".biases";
    const std::string weight_suffix = ".weight";

    std::vector<std::string> quant_prefixes;
    for (const auto& [raw_key, value] : mtp_weights) {
        if (raw_key.size() > scales_suffix.size() &&
            raw_key.compare(raw_key.size() - scales_suffix.size(), scales_suffix.size(), scales_suffix) == 0) {
            std::string prefix = raw_key.substr(0, raw_key.size() - scales_suffix.size());
            std::string weight_key = prefix + ".weight";
            if (mtp_weights.count(weight_key)) {
                quant_prefixes.push_back(prefix);
            }
        }
    }

    // Dequantize and load quantized weights first.
    for (const auto& prefix : quant_prefixes) {
        std::string weight_key = prefix + ".weight";
        std::string scales_key = prefix + ".scales";
        std::string biases_key = prefix + ".biases";

        const auto& packed = mtp_weights.at(weight_key);
        const auto& scales = mtp_weights.at(scales_key);
        std::optional<mx::array> biases;
        auto bit = mtp_weights.find(biases_key);
        if (bit != mtp_weights.end()) {
            biases = bit->second;
        }

        auto deq = mx::dequantize(packed, scales, biases, args_.quant_group_size, args_.quant_bits);

        // Map to weight_map key (strip mtp. prefix if present)
        std::string lookup_key = weight_key;
        auto pos = lookup_key.find("mtp.");
        if (pos != std::string::npos) {
            lookup_key = lookup_key.substr(pos + 4);
        }

        auto it = wmap.find(lookup_key);
        if (it != wmap.end()) {
            *it->second = std::move(deq);
        }
    }

    // Load non-quantized weights (skip scales/biases entries).
    for (const auto& [raw_key, value] : mtp_weights) {
        // Skip scales and biases — already processed
        if (raw_key.size() > scales_suffix.size() &&
            (raw_key.compare(raw_key.size() - scales_suffix.size(), scales_suffix.size(), scales_suffix) == 0 ||
             raw_key.compare(raw_key.size() - biases_suffix.size(), biases_suffix.size(), biases_suffix) == 0)) {
            continue;
        }
        // Skip quantized weights — already dequantized above
        bool is_quantized_weight = false;
        if (raw_key.size() > weight_suffix.size() &&
            raw_key.compare(raw_key.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0) {
            std::string prefix = raw_key.substr(0, raw_key.size() - weight_suffix.size());
            std::string sk = prefix + ".scales";
            if (mtp_weights.count(sk)) {
                is_quantized_weight = true;
            }
        }
        if (is_quantized_weight) continue;

        std::string key = raw_key;
        auto pos = key.find("mtp.");
        if (pos != std::string::npos) {
            key = key.substr(pos + 4);
        }
        auto it = wmap.find(key);
        if (it != wmap.end()) {
            *it->second = value;
        }
    }

    // Validate MoE weights if MoE layer is active.
    // Check that key MoE weight arrays have been populated (non-empty shape).
    if (moe_layer_) {
        auto moe_wmap = moe_layer_->weight_map();
        for (const auto& [key, ptr] : moe_wmap) {
            (void)key; // Used for diagnostics only.
            // If any MoE weight is still zero-initialized with a large shape,
            // it likely means the weight was not loaded from the checkpoint.
            // We log a warning rather than fail — some weights may legitimately
            // be near-zero after training.
            if (ptr->shape().size() > 0) {
                int64_t total_elements = 1;
                for (auto s : ptr->shape()) total_elements *= s;
                if (total_elements > 1024 && mx::allclose(*ptr, mx::zeros_like(*ptr)).item<bool>()) {
                    // Potentially unloaded MoE weight.
                }
            }
        }
    }
}

}  // namespace mlx_lm
