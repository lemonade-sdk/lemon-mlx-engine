// BitNet 1.58-bit model implementation for lemon-mlx-engine
// Llama architecture with relu_squared activation and ternary weights
// Dequantizes uint8 packed ternary weights in sanitize_impl
// "Little bones" — Gord Downie

#include <mlx-lm/llm/models/bitnet.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

// BitNet reuses Llama's from_json since BitNetConfiguration = LlamaConfiguration

// --- RMS Norm helper ---

static mx::array rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    auto ms = mx::mean(mx::square(x), {-1}, true);
    auto norm = x * mx::rsqrt(ms + eps);
    return norm * weight;
}

// --- BitNet ternary dequantization ---

mx::array dequantize_bitnet_weight(
    const mx::array& packed_weight,
    const mx::array& weight_scale,
    int out_features) {

    auto packed = mx::astype(packed_weight, mx::int32);
    int packed_rows = packed_weight.shape(0);
    int in_features = packed_weight.shape(1);

    // Extract 4 ternary values from each byte along row dimension
    auto v0 = mx::bitwise_and(packed, mx::array(0x03));
    auto v1 = mx::bitwise_and(mx::right_shift(packed, mx::array(2)), mx::array(0x03));
    auto v2 = mx::bitwise_and(mx::right_shift(packed, mx::array(4)), mx::array(0x03));
    auto v3 = mx::bitwise_and(mx::right_shift(packed, mx::array(6)), mx::array(0x03));

    // Stack along row dim: [packed_rows, 4, in_features] → [out_features, in_features]
    auto stacked = mx::stack({v0, v1, v2, v3}, 1);
    auto flat = mx::reshape(stacked, {out_features, in_features});

    // Map 0→-1, 1→0, 2→+1
    auto ternary = mx::astype(flat - 1, mx::float16);
    auto scale = mx::astype(weight_scale, mx::float16);
    return ternary * scale;
}

// --- Quantized-aware linear (same as Llama) ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight,
    const std::optional<mx::array>& bias = std::nullopt) {
    return linear_forward(x, weight, bias.has_value() ? &bias.value() : nullptr);
}

// --- BitNet Attention ---

BitNetAttention::BitNetAttention(const BitNetConfiguration& args)
    : args_(args),
      scale_(1.0f / std::sqrt(static_cast<float>(args.resolved_head_dim()))),
      rope_(args.resolved_head_dim(),
            args.max_position_embeddings,
            args.rope_traditional,
            args.rope_theta),
      q_proj_w_(mx::zeros({args.num_attention_heads * args.resolved_head_dim(), args.hidden_size})),
      k_proj_w_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      v_proj_w_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      o_proj_w_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      attn_sub_norm_weight_(mx::zeros({args.hidden_size})) {}

void BitNetAttention::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    q_proj_w_ = weights.at(prefix + "q_proj.weight");
    k_proj_w_ = weights.at(prefix + "k_proj.weight");
    v_proj_w_ = weights.at(prefix + "v_proj.weight");
    o_proj_w_ = weights.at(prefix + "o_proj.weight");

    auto it = weights.find(prefix + "attn_sub_norm.weight");
    if (it != weights.end()) {
        attn_sub_norm_weight_ = it->second;
    }
}

std::unordered_map<std::string, mx::array*>
BitNetAttention::weight_map(const std::string& prefix) {
    std::unordered_map<std::string, mx::array*> map;
    map[prefix + "q_proj.weight"] = &q_proj_w_;
    map[prefix + "k_proj.weight"] = &k_proj_w_;
    map[prefix + "v_proj.weight"] = &v_proj_w_;
    map[prefix + "o_proj.weight"] = &o_proj_w_;
    map[prefix + "attn_sub_norm.weight"] = &attn_sub_norm_weight_;
    return map;
}

mx::array BitNetAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0);
    int L = x.shape(1);
    int head_dim = args_.resolved_head_dim();

    auto queries = linear_fwd(x, q_proj_w_);
    auto keys = linear_fwd(x, k_proj_w_);
    auto values = linear_fwd(x, v_proj_w_);

    // [B, L, n_heads, head_dim] → [B, n_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, args_.num_attention_heads, head_dim}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys = rope_(keys, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);

    // [B, n_heads, L, head_dim] → [B, L, n_heads * head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    // BitNet: sub-layer norm before output projection
    if (attn_sub_norm_weight_.size() > 0) {
        output = rms_norm(output, attn_sub_norm_weight_, args_.rms_norm_eps);
    }

    return linear_fwd(output, o_proj_w_);
}

// --- BitNet MLP (relu_squared) ---

void BitNetMLP::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    gate_proj_w_ = weights.at(prefix + "gate_proj.weight");
    down_proj_w_ = weights.at(prefix + "down_proj.weight");
    up_proj_w_ = weights.at(prefix + "up_proj.weight");

    auto it = weights.find(prefix + "ffn_sub_norm.weight");
    if (it != weights.end()) {
        ffn_sub_norm_weight_ = it->second;
    }
}

std::unordered_map<std::string, mx::array*>
BitNetMLP::weight_map(const std::string& prefix) {
    std::unordered_map<std::string, mx::array*> map;
    map[prefix + "gate_proj.weight"] = &gate_proj_w_;
    map[prefix + "down_proj.weight"] = &down_proj_w_;
    map[prefix + "up_proj.weight"] = &up_proj_w_;
    map[prefix + "ffn_sub_norm.weight"] = &ffn_sub_norm_weight_;
    return map;
}

mx::array BitNetMLP::operator()(const mx::array& x) {
    auto gate = relu_squared(linear_fwd(x, gate_proj_w_));
    auto up = linear_fwd(x, up_proj_w_);
    auto hidden = gate * up;

    if (ffn_sub_norm_weight_.size() > 0) {
        hidden = rms_norm(hidden, ffn_sub_norm_weight_, rms_norm_eps_);
    }

    return linear_fwd(hidden, down_proj_w_);
}

// --- BitNet Transformer Block ---

BitNetTransformerBlock::BitNetTransformerBlock(const BitNetConfiguration& args)
    : self_attn_(args), mlp_(args.rms_norm_eps),
      input_layernorm_weight_(mx::zeros({args.hidden_size})),
      post_attention_layernorm_weight_(mx::zeros({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps) {}

void BitNetTransformerBlock::load_weights(
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix) {
    self_attn_.load_weights(weights, prefix + "self_attn.");
    mlp_.load_weights(weights, prefix + "mlp.");
    input_layernorm_weight_ = weights.at(prefix + "input_layernorm.weight");
    post_attention_layernorm_weight_ = weights.at(prefix + "post_attention_layernorm.weight");
}

std::unordered_map<std::string, mx::array*>
BitNetTransformerBlock::weight_map(const std::string& prefix) {
    auto map = self_attn_.weight_map(prefix + "self_attn.");
    auto mlp_map = mlp_.weight_map(prefix + "mlp.");
    map.insert(mlp_map.begin(), mlp_map.end());
    map[prefix + "input_layernorm.weight"] = &input_layernorm_weight_;
    map[prefix + "post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

mx::array BitNetTransformerBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = self_attn_(rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = x + r;
    r = mlp_(rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return h + r;
}

// --- BitNet Model Inner ---

BitNetModelInner::BitNetModelInner(const BitNetConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::zeros({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps) {
    for (int i = 0; i < args.num_hidden_layers; i++) {
        layers_.emplace_back(args);
    }
}

mx::array BitNetModelInner::operator()(
    const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        KVCache* layer_cache = (cache && !cache->empty()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array BitNetModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> BitNetModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;

    for (int i = 0; i < static_cast<int>(layers_.size()); i++) {
        std::string prefix = "layers." + std::to_string(i) + ".";
        auto layer_map = layers_[i].weight_map(prefix);
        map.insert(layer_map.begin(), layer_map.end());
    }

    return map;
}

// --- BitNet Top-Level Model ---

BitNetModel::BitNetModel(const BitNetConfiguration& args)
    : config_(args), model_(args) {
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

PrepareResult BitNetModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size) {
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput BitNetModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/) {
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array BitNetModel::forward_impl(
    const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    } else {
        return model_.embed_as_linear(out);
    }
}

std::unordered_map<std::string, mx::array>
BitNetModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Dequantize uint8 ternary weights at sanitize time
    std::vector<std::string> to_remove;
    std::vector<std::pair<std::string, mx::array>> to_add;

    for (auto& [key, val] : weights) {
        const std::string suffix = ".weight_scale";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            auto prefix = key.substr(0, key.size() - suffix.size());
            auto weight_key = prefix + ".weight";

            auto w_it = weights.find(weight_key);
            if (w_it != weights.end() && w_it->second.dtype() == mx::uint8) {
                int packed_rows = w_it->second.shape(0);
                int out_features = packed_rows * 4;

                std::cout << "[BitNet] Dequantizing " << weight_key
                          << " (" << packed_rows << "x" << w_it->second.shape(1)
                          << " uint8 -> " << out_features << "x" << w_it->second.shape(1)
                          << " float16)" << std::endl;

                to_add.emplace_back(weight_key, dequantize_bitnet_weight(
                    w_it->second, val, out_features));
                to_remove.push_back(key);
            }
        }
    }

    for (auto& [k, v] : to_add) {
        weights.insert_or_assign(k, std::move(v));
    }
    for (const auto& k : to_remove) {
        weights.erase(k);
    }

    // Remove unused rotary embeddings
    std::vector<std::string> rotary_remove;
    for (auto& [k, v] : weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            rotary_remove.push_back(k);
        }
    }
    for (const auto& k : rotary_remove) {
        weights.erase(k);
    }

    return weights;
}

void BitNetModel::load_weights(
    const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
        }
    }

    std::cout << "[BitNet] Loaded " << config_.num_hidden_layers << " layers, "
              << "hidden_size=" << config_.hidden_size
              << ", intermediate_size=" << config_.intermediate_size
              << ", vocab_size=" << config_.vocab_size << std::endl;
}

std::unordered_map<std::string, mx::array*> BitNetModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : model_.weight_map()) {
        map["model." + k] = v;
    }

    if (lm_head_weight_.has_value()) {
        map["lm_head.weight"] = &lm_head_weight_.value();
    }

    return map;
}

} // namespace mlx_lm
