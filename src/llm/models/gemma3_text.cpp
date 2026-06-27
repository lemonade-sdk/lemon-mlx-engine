// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3Text.swift
// Gemma3 uses Q/K RMSNorm with +1 offset (Gemma style), GELU activation,
// sliding window pattern, 4 norms per block, query_pre_attn_scalar for scale,
// embed scaling by sqrt(hidden_size)

#include <mlx-lm/llm/models/gemma3_text.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <iostream>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, Gemma3TextConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.rope_local_base_freq = j.value("rope_local_base_freq", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.query_pre_attn_scalar = j.value("query_pre_attn_scalar", 256.0f);
    c.sliding_window = j.value("sliding_window", 512);
    c.sliding_window_pattern = j.value("sliding_window_pattern", 6);
    c.max_position_embeddings = j.value("max_position_embeddings", 32768);

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// Gemma-style RMSNorm: rms_norm(x, weight + 1.0, eps)
static mx::array gemma_rms_norm(const mx::array& x, const mx::array& weight, float eps) {
    auto adjusted = mx::add(mx::array(1.0f), weight);
    return mx::fast::rms_norm(x, adjusted, eps);
}

// GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
static mx::array gelu(const mx::array& x) {
    return mx::multiply(
        mx::multiply(x, mx::array(0.5f)),
        mx::add(mx::array(1.0f), mx::erf(mx::divide(x, mx::array(std::sqrt(2.0f))))));
}

// --- Gemma3TextAttention ---

Gemma3TextAttention::Gemma3TextAttention(const Gemma3TextConfiguration& config, int layer_idx)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(std::pow(config.query_pre_attn_scalar, -0.5f)),
      is_sliding_((layer_idx + 1) % config.sliding_window_pattern != 0),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      q_norm_weight_(mx::ones({config.head_dim})),
      k_norm_weight_(mx::ones({config.head_dim})),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(is_sliding_ ? config.rope_local_base_freq : config.rope_theta)
{}

mx::array Gemma3TextAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // Q/K RMSNorm with Gemma +1 style
    queries = gemma_rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    keys = gemma_rms_norm(keys, k_norm_weight_, rms_norm_eps_);

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // Scaled dot-product attention
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3TextAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- Gemma3TextMLP ---

Gemma3TextMLP::Gemma3TextMLP(int dim, int hidden_dim)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array Gemma3TextMLP::operator()(const mx::array& x) {
    // gate projection with GELU activation, then element-wise multiply with up projection
    auto g = gelu(linear_fwd(x, gate_weight_));
    return linear_fwd(mx::multiply(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3TextMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Gemma3TextTransformerBlock ---
// 4 norms per block, all Gemma +1 style:
//   input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm
// Residual pattern:
//   normed = input_layernorm(x); r = attn(normed); h = x + post_attention_layernorm(r)
//   normed = pre_feedforward_layernorm(h); r = mlp(normed); out = h + post_feedforward_layernorm(r)

Gemma3TextTransformerBlock::Gemma3TextTransformerBlock(const Gemma3TextConfiguration& config, int layer_idx)
    : self_attn_(config, layer_idx),
      mlp_(config.hidden_size, config.intermediate_size),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      pre_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Gemma3TextTransformerBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = self_attn_(gemma_rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = clip_residual(x, gemma_rms_norm(r, post_attention_layernorm_weight_, rms_norm_eps_));
    r = mlp_(gemma_rms_norm(h, pre_feedforward_layernorm_weight_, rms_norm_eps_));
    return clip_residual(h, gemma_rms_norm(r, post_feedforward_layernorm_weight_, rms_norm_eps_));
}

std::unordered_map<std::string, mx::array*> Gemma3TextTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["pre_feedforward_layernorm.weight"] = &pre_feedforward_layernorm_weight_;
    map["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;
    return map;
}

// --- Gemma3TextModelInner ---

Gemma3TextModelInner::Gemma3TextModelInner(const Gemma3TextConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      hidden_size_(config.hidden_size),
      sliding_window_(config.sliding_window),
      sliding_window_pattern_(config.sliding_window_pattern)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config, i);
}

mx::array Gemma3TextModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    // Embedding lookup with scaling by sqrt(hidden_size)
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    h = mx::multiply(h, mx::array(std::sqrt(static_cast<float>(hidden_size_))));

    // Create two masks: global (no window) and sliding (with window)
    // Global layer index: the last layer in each sliding_window_pattern group
    // Use the first global layer's cache for global mask, first sliding layer's cache for sliding mask
    int global_idx = sliding_window_pattern_ - 1;
    AttentionMask global_mask = create_attention_mask(
        h, cache && global_idx < static_cast<int>(cache->size()) ? &(*cache)[global_idx] : nullptr);

    AttentionMask sliding_mask;
    if (sliding_window_pattern_ > 1) {
        sliding_mask = create_attention_mask(
            h, cache && !cache->empty() ? &(*cache)[0] : nullptr, sliding_window_);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        // Global layers are those at positions: sliding_window_pattern - 1, 2*sliding_window_pattern - 1, ...
        bool is_global = (static_cast<int>(i) % sliding_window_pattern_ == sliding_window_pattern_ - 1);
        const auto& mask = is_global ? global_mask : sliding_mask;
        h = layers_[i](h, mask, lc);
    }

    // Final norm with Gemma +1 style
    return gemma_rms_norm(h, norm_weight_, rms_norm_eps_);
}

std::unordered_map<std::string, mx::array*> Gemma3TextModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Gemma3TextModel ---

Gemma3TextModel::Gemma3TextModel(const Gemma3TextConfiguration& config)
    : config_(config),
      model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult Gemma3TextModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Gemma3TextModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Gemma3TextModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    // Always use lm_head (not tied embeddings)
    return linear_forward(out, lm_head_weight_);
}

std::vector<KVCache> Gemma3TextModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        bool is_global = (i % config_.sliding_window_pattern == config_.sliding_window_pattern - 1);
        if (is_global) {
            // Global layers: standard cache with large step size
            caches.emplace_back(KVCacheSimple{});
        } else {
            // Sliding window layers: rotating cache
            caches.emplace_back(RotatingKVCache(config_.sliding_window, 0));
        }
    }

    return caches;
}

std::unordered_map<std::string, mx::array>
Gemma3TextModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Handle language_model.* prefix (VLM compatibility — strip it).
    // NOTE: The factory-level code in llm_factory.cpp has ALREADY stripped
    // "language_model.model." -> "model." for all weight keys. This sanitize
    // handles any remaining "language_model." prefix (e.g., lm_head.*).
    // Must keep ALL keys, not just those with the LM prefix, because the
    // factory may have already converted model keys.
    std::unordered_map<std::string, mx::array> processed;
    processed.reserve(weights.size());
    for (auto& [key, val] : weights) {
        if (key.find("language_model.") == 0) {
            processed.insert_or_assign(key.substr(15), std::move(val));
        } else {
            processed.insert_or_assign(key, std::move(val));
        }
    }
    weights = std::move(processed);

    // If "model.embed_tokens.weight" exists but "lm_head.weight" doesn't, copy embed to lm_head
    if (weights.find("lm_head.weight") == weights.end()) {
        for (const auto& suffix : {"weight", "scales", "biases"}) {
            auto embed_key = std::string("model.embed_tokens.") + suffix;
            auto it = weights.find(embed_key);
            if (it != weights.end()) {
                weights.insert_or_assign(std::string("lm_head.") + suffix, it->second);
            }
        }
    }

    return weights;
}

void Gemma3TextModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Gemma3TextModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

} // namespace mlx_lm
