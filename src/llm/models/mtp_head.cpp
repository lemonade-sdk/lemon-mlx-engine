
#include <mlx-lm/llm/models/mtp_head.h>

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
      q_proj_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim(), args.hidden_size})),
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
    auto q = linear_no_bias(normed, q_proj_weight_);
    auto k = linear_no_bias(normed, k_proj_weight_);
    auto v = linear_no_bias(normed, v_proj_weight_);

    auto q4 = mx::reshape(q, {B, L, n_heads, hd});
    auto k4 = mx::reshape(k, {B, L, n_kv_heads, hd});
    auto v4 = mx::reshape(v, {B, L, n_kv_heads, hd});

    q4 = mx::transpose(
        mx::fast::rms_norm(q4, q_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    k4 = mx::transpose(
        mx::fast::rms_norm(k4, k_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    v4 = mx::transpose(v4, {0, 2, 1, 3});

    int rope_dims = args_.rope_dims != 0 ? args_.rope_dims : hd;
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
      layer_(args),
      norm_weight_(mx::ones({args.hidden_size})) {}

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
    return layer_(h, mask, cache);
}

mx::array MTPHead::apply_output_norm(const mx::array& h) const {
    return mx::fast::rms_norm(h, norm_weight_, args_.rms_norm_eps);
}

std::unordered_map<std::string, mx::array*> MTPHead::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["pre_fc_norm_hidden.weight"] = &pre_fc_norm_hidden_weight_;
    map["pre_fc_norm_embedding.weight"] = &pre_fc_norm_embedding_weight_;
    map["fc.weight"] = &fc_weight_;
    for (auto& [k, v] : layer_.weight_map()) {
        map["layers.0." + k] = v;
    }
    map["norm.weight"] = &norm_weight_;
    return map;
}

void MTPHead::load_mtp_weights(
    const std::unordered_map<std::string, mx::array>& mtp_weights) {
    auto wmap = weight_map();
    for (const auto& [raw_key, value] : mtp_weights) {
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
}

}  // namespace mlx_lm
