// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmo3.py

#include <mlx-lm/llm/models/olmo3.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- linear_fwd helper ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, Olmo3Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.sliding_window = j.at("sliding_window").get<int>();

    if (j.contains("rope_theta"))
        c.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("attention_bias"))
        c.attention_bias = j["attention_bias"].get<bool>();
    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();

    // Decode layer_types or generate default pattern
    if (j.contains("layer_types") && !j["layer_types"].is_null()) {
        c.layer_types = j["layer_types"].get<std::vector<std::string>>();
    } else {
        // Default: full_attention every 4th layer, else sliding_attention
        c.layer_types.resize(c.num_hidden_layers);
        for (int i = 0; i < c.num_hidden_layers; ++i) {
            c.layer_types[i] = ((i + 1) % 4 == 0) ? "full_attention" : "sliding_attention";
        }
    }

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

// --- Olmo3Attention ---

Olmo3Attention::Olmo3Attention(const Olmo3Configuration& config, int layer_idx)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      q_norm_weight_(mx::ones({config.num_attention_heads * config.resolved_head_dim()})),
      k_norm_weight_(mx::ones({config.num_key_value_heads * config.resolved_head_dim()})),
      rms_norm_eps_(config.rms_norm_eps),
      // For sliding_attention layers, use plain RoPE (no scaling).
      // For full_attention layers, use the rope_scaling config.
      rope_(config.resolved_head_dim(),
            config.max_position_embeddings,
            false, // traditional
            config.rope_theta,
            [&]() -> float {
                if (config.layer_types[layer_idx] == "full_attention" && config.rope_scaling.has_value()) {
                    auto it = config.rope_scaling->find("type");
                    if (it == config.rope_scaling->end())
                        it = config.rope_scaling->find("rope_type");
                    if (it != config.rope_scaling->end() && it->second.is_string() &&
                        it->second.as_string() == "linear") {
                        auto fit = config.rope_scaling->find("factor");
                        if (fit != config.rope_scaling->end() && fit->second.is_float())
                            return 1.0f / fit->second.as_float();
                    }
                }
                return 1.0f;
            }(),
            [&]() -> std::string {
                if (config.layer_types[layer_idx] != "full_attention")
                    return "default";
                if (config.rope_scaling.has_value()) {
                    auto it = config.rope_scaling->find("type");
                    if (it == config.rope_scaling->end())
                        it = config.rope_scaling->find("rope_type");
                    if (it != config.rope_scaling->end() && it->second.is_string())
                        return it->second.as_string();
                }
                return "default";
            }(),
            (config.layer_types[layer_idx] == "full_attention") ? config.rope_scaling : std::nullopt)
{}

mx::array Olmo3Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    // Project Q, K, V (no bias in Olmo3)
    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    // Apply Q/K RMSNorm BEFORE reshape
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    keys    = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys    = rope_(keys, offset);

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

std::unordered_map<std::string, mx::array*> Olmo3Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- Olmo3MLP ---

Olmo3MLP::Olmo3MLP(const Olmo3Configuration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size}))
{}

mx::array Olmo3MLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Olmo3MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Olmo3TransformerBlock ---
// Olmo3 uses POST-norm architecture (same as Olmo2):
//   r = attn(x);  h = x + rms_norm(r, post_attention_layernorm_weight)
//   r = mlp(h);   out = h + rms_norm(r, post_feedforward_layernorm_weight)

Olmo3TransformerBlock::Olmo3TransformerBlock(const Olmo3Configuration& config, int layer_idx)
    : self_attn_(config, layer_idx),
      mlp_(config),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Olmo3TransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // POST-norm: norm is applied AFTER the sublayer, not before
    auto r = mx::fast::rms_norm(self_attn_(x, mask, cache),
                                 post_attention_layernorm_weight_, rms_norm_eps_);
    auto h = mx::add(x, r);
    r = mx::fast::rms_norm(mlp_(h), post_feedforward_layernorm_weight_, rms_norm_eps_);
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Olmo3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : self_attn_.weight_map())
        map["self_attn." + k] = v;

    for (auto& [k, v] : mlp_.weight_map())
        map["mlp." + k] = v;

    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;

    return map;
}

// --- Olmo3ModelInner ---

Olmo3ModelInner::Olmo3ModelInner(const Olmo3Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      layer_types_(config.layer_types),
      sliding_window_(config.sliding_window)
{
    // Find first occurrence of each layer type for mask creation
    swa_idx_ = 0;
    ga_idx_ = 0;
    for (size_t i = 0; i < layer_types_.size(); ++i) {
        if (layer_types_[i] == "sliding_attention") {
            swa_idx_ = static_cast<int>(i);
            break;
        }
    }
    for (size_t i = 0; i < layer_types_.size(); ++i) {
        if (layer_types_[i] == "full_attention") {
            ga_idx_ = static_cast<int>(i);
            break;
        }
    }

    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }
}

mx::array Olmo3ModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    // Embedding lookup
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Create TWO masks: one for full attention, one for sliding window
    auto full_mask = create_attention_mask(
        h, cache && ga_idx_ < static_cast<int>(cache->size()) ? &(*cache)[ga_idx_] : nullptr);
    auto sliding_mask = create_attention_mask(
        h, cache && swa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[swa_idx_] : nullptr,
        sliding_window_);

    // Forward through layers, selecting the appropriate mask per layer
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        const auto& mask = (layer_types_[i] == "full_attention") ? full_mask : sliding_mask;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Olmo3ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Olmo3ModelInner::weight_map() {
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

// --- Olmo3Model ---

Olmo3Model::Olmo3Model(const Olmo3Configuration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult Olmo3Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput Olmo3Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Olmo3Model::forward_impl(
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

std::vector<KVCache> Olmo3Model::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);

    for (const auto& layer_type : config_.layer_types) {
        if (layer_type == "sliding_attention") {
            caches.emplace_back(RotatingKVCache(config_.sliding_window));
        } else {
            // full_attention
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            } else {
                caches.emplace_back(KVCacheSimple{});
            }
        }
    }

    return caches;
}

std::unordered_map<std::string, mx::array>
Olmo3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Remove unused precomputed rotary frequencies
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    // Erase lm_head.weight if embeddings are tied
    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    return weights;
}

void Olmo3Model::load_weights(
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

std::unordered_map<std::string, mx::array*> Olmo3Model::weight_map() {
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
