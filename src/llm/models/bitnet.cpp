// BitNet 1.58-bit model implementation for lemon-mlx-engine
// Llama architecture with relu_squared activation and ternary weights
// "Little bones" — Gord Downie

#include <mlx-lm/llm/models/bitnet.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization (reuse Llama's with BitNet extras) ---

void from_json(const nlohmann::json& j, BitNetConfiguration& c) {
    // Reuse Llama config parsing
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    if (j.contains("max_position_embeddings"))
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();
    if (j.contains("rope_theta"))
        c.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
}

// --- RMS Norm helper ---

static mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    auto ms = mx::mean(mx::square(x), {-1}, true);
    auto norm = x * mx::rsqrt(ms + eps);
    return norm * weight;
}

// --- BitNet Attention ---

BitNetAttention::BitNetAttention(const BitNetConfiguration& args)
    : args_(args),
      scale_(1.0f / std::sqrt(static_cast<float>(args.resolved_head_dim()))),
      rope_(args.resolved_head_dim(),
            args.max_position_embeddings,
            args.rope_traditional,
            args.rope_theta),
      q_proj_(args.hidden_size, args.num_attention_heads * args.resolved_head_dim(), false),
      k_proj_(args.hidden_size, args.num_key_value_heads * args.resolved_head_dim(), false),
      v_proj_(args.hidden_size, args.num_key_value_heads * args.resolved_head_dim(), false),
      o_proj_(args.num_attention_heads * args.resolved_head_dim(), args.hidden_size, false) {}

void BitNetAttention::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    q_proj_.load_weights(weights, prefix + "q_proj.");
    k_proj_.load_weights(weights, prefix + "k_proj.");
    v_proj_.load_weights(weights, prefix + "v_proj.");
    o_proj_.load_weights(weights, prefix + "o_proj.");
    // BitNet sub-layer norm
    auto it = weights.find(prefix + "attn_sub_norm.weight");
    if (it != weights.end()) {
        attn_sub_norm_weight_ = it->second;
    }
}

mx::array BitNetAttention::operator()(
    const mx::array& x, const mx::array& mask, KVCache* cache) {
    int B = x.shape(0);
    int L = x.shape(1);
    int head_dim = args_.resolved_head_dim();

    auto queries = q_proj_(x);
    auto keys = k_proj_(x);
    auto values = v_proj_(x);

    queries = mx::reshape(queries, {B, L, args_.num_attention_heads, head_dim});
    keys = mx::reshape(keys, {B, L, args_.num_key_value_heads, head_dim});
    values = mx::reshape(values, {B, L, args_.num_key_value_heads, head_dim});

    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys = rope_(keys, offset);

    if (cache) {
        auto [k, v] = cache->update_and_fetch(keys, values);
        keys = k;
        values = v;
    }

    auto output = scaled_dot_product_attention(
        queries, keys, values, scale_, mask);

    output = mx::reshape(output, {B, L, -1});

    // BitNet: apply sub-layer norm before output projection
    if (attn_sub_norm_weight_.size() > 0) {
        output = rms_norm(output, attn_sub_norm_weight_, args_.rms_norm_eps);
    }

    return o_proj_(output);
}

// --- BitNet MLP (relu_squared instead of SiLU) ---

BitNetMLP::BitNetMLP(int hidden_size, int intermediate_size)
    : gate_proj_(hidden_size, intermediate_size, false),
      down_proj_(intermediate_size, hidden_size, false),
      up_proj_(hidden_size, intermediate_size, false) {}

void BitNetMLP::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    gate_proj_.load_weights(weights, prefix + "gate_proj.");
    down_proj_.load_weights(weights, prefix + "down_proj.");
    up_proj_.load_weights(weights, prefix + "up_proj.");
    auto it = weights.find(prefix + "ffn_sub_norm.weight");
    if (it != weights.end()) {
        ffn_sub_norm_weight_ = it->second;
    }
}

mx::array BitNetMLP::operator()(const mx::array& x) {
    // BitNet uses relu_squared: relu(x)^2 instead of SiLU
    auto gate = relu_squared(gate_proj_(x));
    auto up = up_proj_(x);
    auto hidden = gate * up;

    // BitNet: apply sub-layer norm before down projection
    if (ffn_sub_norm_weight_.size() > 0) {
        hidden = rms_norm(hidden, ffn_sub_norm_weight_, 1e-5f);
    }

    return down_proj_(hidden);
}

// --- BitNet Decoder Layer ---

BitNetDecoderLayer::BitNetDecoderLayer(const BitNetConfiguration& args)
    : self_attn_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      rms_norm_eps_(args.rms_norm_eps) {}

void BitNetDecoderLayer::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    self_attn_.load_weights(weights, prefix + "self_attn.");
    mlp_.load_weights(weights, prefix + "mlp.");
    input_layernorm_weight_ = weights.at(prefix + "input_layernorm.weight");
    post_attention_layernorm_weight_ = weights.at(prefix + "post_attention_layernorm.weight");
}

mx::array BitNetDecoderLayer::operator()(
    const mx::array& x, const mx::array& mask, KVCache* cache) {
    auto residual = x;
    auto hidden = rms_norm(x, input_layernorm_weight_, rms_norm_eps_);
    hidden = self_attn_(hidden, mask, cache);
    hidden = residual + hidden;

    residual = hidden;
    hidden = rms_norm(hidden, post_attention_layernorm_weight_, rms_norm_eps_);
    hidden = mlp_(hidden);
    hidden = residual + hidden;

    return hidden;
}

// --- BitNet Model ---

BitNetModel::BitNetModel(const BitNetConfiguration& config)
    : config_(config) {
    for (int i = 0; i < config.num_hidden_layers; i++) {
        layers_.emplace_back(config);
    }
}

void BitNetModel::load_weights(
    const std::unordered_map<std::string, mx::array>& weights) {
    embed_tokens_ = weights.at("model.embed_tokens.weight");
    norm_weight_ = weights.at("model.norm.weight");

    auto lm_it = weights.find("lm_head.weight");
    if (lm_it != weights.end()) {
        lm_head_weight_ = lm_it->second;
    }

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        layers_[i].load_weights(weights, prefix);
    }
}

mx::array BitNetModel::operator()(const mx::array& inputs, KVCacheVector& cache) {
    auto x = mx::take(embed_tokens_, inputs, 0);

    auto mask = create_causal_mask(x.shape(1),
        cache.empty() ? 0 : cache[0]->offset());

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        x = layers_[i](x, mask, cache.empty() ? nullptr : cache[i].get());
    }

    x = rms_norm(x, norm_weight_, config_.rms_norm_eps);

    if (lm_head_weight_.has_value()) {
        x = mx::matmul(x, mx::transpose(*lm_head_weight_));
    } else if (config_.tie_word_embeddings) {
        x = mx::matmul(x, mx::transpose(embed_tokens_));
    }

    return x;
}

} // namespace mlx_lm
