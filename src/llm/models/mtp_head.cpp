
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx-lm/llm/models/mtp_moe.h>
#include <mlx-lm/common/quantized_linear.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

namespace {

mx::array silu(const mx::array& x) {
    return mx::multiply(x, mx::sigmoid(x));
}

mx::array swiglu(const mx::array& gate, const mx::array& up) {
    return mx::multiply(silu(gate), up);
}

// Prefer quantized_matmul when weight is registered (trunk path); dense fallback.
inline mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w, nullptr);
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
    // Apply sigmoid gate: reshape q_gate [B, L, n_heads, hd] -> [B, L, n_heads * hd]
    // to match attention output shape, matching Qwen35MoEAttention line 189.
    auto gate_sigmoid = mx::sigmoid(mx::reshape(q_gate, {B, L, -1}));
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
    // Defensive: ensure inputs are 3D [B, L, H] for rms_norm.
    // During speculative decoding, single-token inputs may arrive as 2D.
    auto hs = hidden_state;
    auto te = token_embedding;
    if (hs.ndim() == 2) {
        hs = mx::reshape(hs, {1, 1, hs.shape(-1)});
    }
    if (te.ndim() == 2) {
        te = mx::reshape(te, {1, 1, te.shape(-1)});
    }
    auto h_norm = mx::fast::rms_norm(
        hs, pre_fc_norm_hidden_weight_, args_.rms_norm_eps);
    auto e_norm = mx::fast::rms_norm(
        te, pre_fc_norm_embedding_weight_, args_.rms_norm_eps);
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

    // Escape hatches:
    //   MLX_MTP_DEQUANT=1  — dequant packed → dense (legacy); also skip auto-quant.
    //   MLX_MTP_KEEP_BF16=1 — leave source BF16 linears dense (no runtime quant).
    // Default (C1): packed checkpoint → register; dense BF16 linears → quantize
    // at load (LemonMLXE ships BF16 mtp.* while trunk is 4-bit) so draft uses
    // quantized_matmul / gather_qmm like the trunk.
    static const bool force_dequant =
        std::getenv("MLX_MTP_DEQUANT") != nullptr;
    static const bool keep_bf16 =
        std::getenv("MLX_MTP_KEEP_BF16") != nullptr;

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

    auto strip_mtp = [](std::string key) {
        auto pos = key.find("mtp.");
        if (pos != std::string::npos) {
            key = key.substr(pos + 4);
        }
        return key;
    };

    // Norm / bias tensors stay full precision (not matmul weights).
    auto is_norm_or_bias = [](const std::string& key) {
        if (key.find("norm") != std::string::npos) return true;
        if (key.size() >= 5 &&
            key.compare(key.size() - 5, 5, ".bias") == 0) return true;
        return false;
    };

    // Affine quant needs last dim group-aligned (same rule as convert.cpp).
    auto can_quantize = [&](const mx::array& w) {
        if (args_.quant_bits <= 0 || args_.quant_group_size <= 0) return false;
        if (w.ndim() < 2) return false;
        int last = w.shape(-1);
        return last >= args_.quant_group_size &&
               (last % args_.quant_group_size) == 0;
    };

    auto& reg = QuantizedWeightRegistry::instance();
    int registered = 0;
    int dequantized = 0;
    int auto_quantized = 0;
    int dense_kept = 0;

    // Path A: checkpoint already has packed + scales.
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

        std::string lookup_key = strip_mtp(weight_key);
        auto it = wmap.find(lookup_key);
        if (it == wmap.end()) {
            continue;
        }

        if (force_dequant) {
            *it->second = mx::dequantize(
                packed, scales, biases, args_.quant_group_size, args_.quant_bits);
            ++dequantized;
        } else {
            *it->second = packed;
            reg.register_weight(
                it->second,
                scales,
                biases,
                args_.quant_group_size,
                args_.quant_bits);
            ++registered;
        }
    }

    // Path B: dense (BF16/FP16/F32) weights — load as-is or auto-quantize linears.
    for (const auto& [raw_key, value] : mtp_weights) {
        if (raw_key.size() > scales_suffix.size() &&
            (raw_key.compare(raw_key.size() - scales_suffix.size(), scales_suffix.size(), scales_suffix) == 0 ||
             raw_key.compare(raw_key.size() - biases_suffix.size(), biases_suffix.size(), biases_suffix) == 0)) {
            continue;
        }
        bool is_quantized_weight = false;
        if (raw_key.size() > weight_suffix.size() &&
            raw_key.compare(raw_key.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0) {
            std::string prefix = raw_key.substr(0, raw_key.size() - weight_suffix.size());
            if (mtp_weights.count(prefix + ".scales")) {
                is_quantized_weight = true;
            }
        }
        if (is_quantized_weight) continue;

        std::string key = strip_mtp(raw_key);
        auto it = wmap.find(key);
        if (it == wmap.end()) continue;

        // SwitchGLU/SwitchLinear store [E, out, in]. Official MTP shared_expert
        // tensors are 2D [out, in]; expand to E=1 so gather_qmm matches trunk.
        mx::array w = value;
        const bool is_switch_w =
            (key.find("switch_mlp.") != std::string::npos ||
             key.find("shared_expert.") != std::string::npos) &&
            key.find("shared_expert_gate") == std::string::npos;
        if (is_switch_w && w.ndim() == 2) {
            w = mx::reshape(w, {1, w.shape(0), w.shape(1)});
        }

        // Note: mlx-lm often keeps mtp.fc in BF16 for accuracy; on gfx1150 0.8B
        // H2, quantizing fc with the other linears was slightly faster (~99.9
        // vs ~99.0 gen t/s) at similar accept — leave fc eligible for auto-q.
        const bool try_auto_q =
            !force_dequant && !keep_bf16 && !is_norm_or_bias(key) &&
            can_quantize(w);

        if (try_auto_q) {
            // Runtime quant of LemonMLXE-style BF16 mtp.* (convert historically
            // left the head dense for acceptance; draft bandwidth suffers).
            auto qr = mx::quantize(
                mx::contiguous(w), args_.quant_group_size, args_.quant_bits);
            *it->second = qr[0];
            std::optional<mx::array> biases = qr[2];
            reg.register_weight(
                it->second,
                qr[1],
                biases,
                args_.quant_group_size,
                args_.quant_bits);
            ++auto_quantized;
        } else {
            *it->second = std::move(w);
            ++dense_kept;
        }
    }

    // HF / guru87-style MTP heads store RMSNorm as (γ − 1). mlx-community
    // converted packages (e.g. 4B-MTP-4bit) already bake +1 into the file.
    // Without the shift, draft logits are garbage and accept rate sticks at 0
    // (field: 0.8B H2 n_draft=2, KEEP_BF16 and runtime-quant both accept=0).
    // Detect unshifted raw: pre_fc_norm_hidden mean is near 0 / negative
    // (shifted packages sit ~0.5–1.0). Escape: MLX_MTP_NO_NORM_SHIFT=1.
    int norm_shifted = 0;
    static const bool kNoNormShift =
        std::getenv("MLX_MTP_NO_NORM_SHIFT") != nullptr;
    if (!kNoNormShift) {
        auto pre_it = wmap.find("pre_fc_norm_hidden.weight");
        if (pre_it != wmap.end() && pre_it->second != nullptr) {
            // Cast to f32 before mean/item — bf16 item<float>() can read as ~0
            // and falsely trigger a second +1 on already-shifted packages (4B).
            auto mean_arr =
                mx::mean(mx::astype(*pre_it->second, mx::float32));
            mx::eval(mean_arr);
            const float pre_mean = mean_arr.item<float>();
            if (pre_mean < 0.2f) {
                for (auto& [k, ptr] : wmap) {
                    if (ptr == nullptr) continue;
                    if (k.find("norm") == std::string::npos) continue;
                    if (k.size() < 7 ||
                        k.compare(k.size() - 7, 7, ".weight") != 0) {
                        continue;
                    }
                    // Only shift dense float norms (not packed U32 linears).
                    if (ptr->dtype() == mx::bfloat16 ||
                        ptr->dtype() == mx::float16 ||
                        ptr->dtype() == mx::float32) {
                        *ptr = mx::add(*ptr, mx::array(1.0f, ptr->dtype()));
                        ++norm_shifted;
                    }
                }
                if (norm_shifted > 0) {
                    std::vector<mx::array> shift_eval;
                    shift_eval.reserve(static_cast<size_t>(norm_shifted));
                    for (auto& [k, ptr] : wmap) {
                        if (k.find("norm") != std::string::npos && ptr) {
                            shift_eval.push_back(*ptr);
                        }
                    }
                    if (!shift_eval.empty()) mx::eval(shift_eval);
                    std::cerr << "[MTP] RMSNorm +1 shift applied to "
                              << norm_shifted
                              << " tensors (pre_fc_norm_hidden mean was "
                              << pre_mean << "; raw HF/guru87 style)\n";
                }
            }
        }
    }

    if (auto_quantized > 0) {
        // Materialize packed weights + scales so load does not leave lazy graphs.
        std::vector<mx::array> eval_list;
        eval_list.reserve(static_cast<size_t>(auto_quantized) * 2);
        for (auto& [k, ptr] : wmap) {
            (void)k;
            if (reg.find(ptr)) {
                eval_list.push_back(*ptr);
                eval_list.push_back(reg.find(ptr)->scales);
                if (reg.find(ptr)->biases) {
                    eval_list.push_back(reg.find(ptr)->biases.value());
                }
            }
        }
        if (!eval_list.empty()) mx::eval(eval_list);
    }

    std::cerr << "[MTP] load_mtp_weights: registered_ckpt=" << registered
              << " auto_quantized=" << auto_quantized
              << " dense_kept=" << dense_kept
              << " dequantized=" << dequantized
              << " norm_shifted=" << norm_shifted
              << " bits=" << args_.quant_bits
              << " gs=" << args_.quant_group_size
              << (force_dequant ? " (MLX_MTP_DEQUANT)" : "")
              << (keep_bf16 ? " (MLX_MTP_KEEP_BF16)" : "")
              << "\n";
}

}  // namespace mlx_lm
