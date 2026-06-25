// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Exaone4.swift — Exaone4 with per-layer local/global attention,
// Q/K RMSNorm, post-norm, and sliding window

#include <mlx-lm/llm/models/exaone4.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx/mlx.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, Exaone4Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.head_dim = j.at("head_dim").get<int>();
    c.tie_word_embeddings = j.at("tie_word_embeddings").get<bool>();

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }

    if (j.contains("sliding_window") && !j["sliding_window"].is_null()) {
        c.sliding_window = j["sliding_window"].get<int>();
    }

    if (j.contains("sliding_window_pattern") && !j["sliding_window_pattern"].is_null()) {
        c.sliding_window_pattern = j["sliding_window_pattern"].get<std::string>();
    }
}

// --- Linear helper ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Exaone4Attention ---

Exaone4Attention::Exaone4Attention(const Exaone4Configuration& config, bool is_local, bool use_rope)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      is_local_(is_local),
      use_rope_(use_rope),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      q_norm_weight_(mx::ones({config.head_dim})),
      k_norm_weight_(mx::ones({config.head_dim})),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta),
      rope_scale_(1.0f)
{
    // Determine rope_scale from rope_scaling
    if (use_rope && config.rope_scaling.has_value()) {
        const auto& rs = config.rope_scaling.value();
        auto type_it = rs.find("type");
        if (type_it != rs.end() && type_it->second.is_string() &&
            type_it->second.as_string() == "linear") {
            auto factor_it = rs.find("factor");
            if (factor_it != rs.end() && factor_it->second.is_float()) {
                rope_scale_ = 1.0f / factor_it->second.as_float();
            }
        }
    }
}

mx::array Exaone4Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Reshape and apply Q/K RMSNorm
    queries = mx::fast::rms_norm(
        mx::reshape(queries, {B, L, num_heads_, -1}), q_norm_weight_, rms_norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3});

    keys = mx::fast::rms_norm(
        mx::reshape(keys, {B, L, num_kv_heads_, -1}), k_norm_weight_, rms_norm_eps_);
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // Apply RoPE if enabled
    if (use_rope_) {
        int offset = cache ? cache->offset() : 0;
        queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, rope_scale_, offset);
        keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, rope_scale_, offset);
    }

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

std::unordered_map<std::string, mx::array*> Exaone4Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- Exaone4MLP ---

Exaone4MLP::Exaone4MLP(int dim, int hidden_dim)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array Exaone4MLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Exaone4MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Exaone4TransformerBlock ---

Exaone4TransformerBlock::Exaone4TransformerBlock(
    const Exaone4Configuration& config, bool is_local, bool use_rope)
    : self_attn_(config, is_local, use_rope),
      mlp_(config.hidden_size, config.intermediate_size),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Exaone4TransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // POST-norm: r = attn(x); h = x + rms_norm(r)
    auto r = self_attn_(x, mask, cache);
    auto h = mx::add(x, mx::fast::rms_norm(r, post_attention_layernorm_weight_, rms_norm_eps_));

    // r = mlp(h); out = h + rms_norm(r)
    r = mlp_(h);
    return mx::add(h, mx::fast::rms_norm(r, post_feedforward_layernorm_weight_, rms_norm_eps_));
}

std::unordered_map<std::string, mx::array*> Exaone4TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : self_attn_.weight_map()) {
        map["self_attn." + k] = v;
    }

    for (auto& [k, v] : mlp_.weight_map()) {
        map["mlp." + k] = v;
    }

    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;

    return map;
}

// --- Exaone4ModelInner ---

Exaone4ModelInner::Exaone4ModelInner(const Exaone4Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        // Determine per-layer local/global based on sliding_window_pattern
        // Pattern is a string where each character maps to layers cyclically.
        // "L" = local, anything else = global.
        // If no pattern, isLocal is effectively nil -> isLocal=false, useRope=true
        bool is_local = false;
        bool use_rope = true;

        if (config.sliding_window_pattern.has_value()) {
            const auto& pattern = config.sliding_window_pattern.value();
            int pattern_idx = i % static_cast<int>(pattern.size());
            char ch = pattern[pattern_idx];
            is_local = (ch == 'L');
            // In Swift: useRope = isLocal == nil || (isLocal ?? false)
            // When pattern exists, isLocal is not nil, so useRope = isLocal
            use_rope = is_local;
        }

        layers_.emplace_back(config, is_local, use_rope);
    }
}

mx::array Exaone4ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    // Embedding lookup
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Exaone4ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Exaone4ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }

    return map;
}

// --- Exaone4Model ---

Exaone4Model::Exaone4Model(const Exaone4Configuration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult Exaone4Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput Exaone4Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array Exaone4Model::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return linear_forward(out, lm_head_weight_.value());
    } else {
        return model_.embed_as_linear(out);
    }
}

std::vector<KVCache> Exaone4Model::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (const auto& layer : model_.get_layers()) {
        if (layer.is_local() && config_.sliding_window.has_value()) {
            caches.emplace_back(RotatingKVCache(config_.sliding_window.value()));
        } else {
            caches.emplace_back(KVCacheSimple{});
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
Exaone4Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Remove precomputed rotary embedding inverse frequencies
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("rotary_emb.inv_freq") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    // If tie_word_embeddings, remove lm_head.weight
    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    return weights;
}

void Exaone4Model::load_weights(
    const std::unordered_map<std::string, mx::array>& weights)
{
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
        }
    }
}

std::unordered_map<std::string, mx::array*> Exaone4Model::weight_map() {
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
