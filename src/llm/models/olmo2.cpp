// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/olmo2.py

#include <mlx-lm/llm/models/olmo2.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx/mlx.h>
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

void from_json(const nlohmann::json& j, Olmo2Configuration& c) {
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
    if (j.contains("rope_traditional"))
        c.rope_traditional = j["rope_traditional"].get<bool>();
    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    if (j.contains("attention_bias"))
        c.attention_bias = j["attention_bias"].get<bool>();
    if (j.contains("mlp_bias"))
        c.mlp_bias = j["mlp_bias"].get<bool>();

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

// --- Olmo2Attention ---

Olmo2Attention::Olmo2Attention(const Olmo2Configuration& config)
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
      rope_(config.resolved_head_dim(),
            config.max_position_embeddings,
            config.rope_traditional,
            config.rope_theta,
            [&]() -> float {
                // Compute rope scale: for "linear" type, scale = 1/factor
                if (config.rope_scaling.has_value()) {
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
                if (config.rope_scaling.has_value()) {
                    auto it = config.rope_scaling->find("type");
                    if (it == config.rope_scaling->end())
                        it = config.rope_scaling->find("rope_type");
                    if (it != config.rope_scaling->end() && it->second.is_string())
                        return it->second.as_string();
                }
                return "default";
            }(),
            config.rope_scaling)
{
    if (config.attention_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * config.resolved_head_dim()});
        wk_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wv_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array Olmo2Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    // Project Q, K, V
    auto queries = linear_fwd(x, wq_weight_, wq_bias_ ? &wq_bias_.value() : nullptr);
    auto keys    = linear_fwd(x, wk_weight_, wk_bias_ ? &wk_bias_.value() : nullptr);
    auto values  = linear_fwd(x, wv_weight_, wv_bias_ ? &wv_bias_.value() : nullptr);

    // Apply Q/K RMSNorm BEFORE reshape (on the full projected dimension)
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

    return linear_fwd(output, wo_weight_, wo_bias_ ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> Olmo2Attention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
    if (wq_bias_) map["q_proj.bias"] = &wq_bias_.value();
    if (wk_bias_) map["k_proj.bias"] = &wk_bias_.value();
    if (wv_bias_) map["v_proj.bias"] = &wv_bias_.value();
    if (wo_bias_) map["o_proj.bias"] = &wo_bias_.value();
    return map;
}

// --- Olmo2MLP ---

Olmo2MLP::Olmo2MLP(const Olmo2Configuration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size}))
{
    if (config.mlp_bias) {
        gate_bias_ = mx::zeros({config.intermediate_size});
        down_bias_ = mx::zeros({config.hidden_size});
        up_bias_ = mx::zeros({config.intermediate_size});
    }
}

mx::array Olmo2MLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    auto gate_out = linear_fwd(x, gate_weight_, gate_bias_ ? &gate_bias_.value() : nullptr);
    auto up_out = linear_fwd(x, up_weight_, up_bias_ ? &up_bias_.value() : nullptr);
    return linear_fwd(swiglu(gate_out, up_out), down_weight_,
                      down_bias_ ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> Olmo2MLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
    if (gate_bias_) map["gate_proj.bias"] = &gate_bias_.value();
    if (down_bias_) map["down_proj.bias"] = &down_bias_.value();
    if (up_bias_) map["up_proj.bias"] = &up_bias_.value();
    return map;
}

// --- Olmo2TransformerBlock ---
// Olmo2 uses POST-norm architecture (NOT pre-norm like Llama):
//   r = attn(x);  h = x + rms_norm(r, post_attention_layernorm_weight)
//   r = mlp(h);   out = h + rms_norm(r, post_feedforward_layernorm_weight)

Olmo2TransformerBlock::Olmo2TransformerBlock(const Olmo2Configuration& config)
    : self_attn_(config),
      mlp_(config),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Olmo2TransformerBlock::operator()(
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

std::unordered_map<std::string, mx::array*> Olmo2TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : self_attn_.weight_map())
        map["self_attn." + k] = v;

    for (auto& [k, v] : mlp_.weight_map())
        map["mlp." + k] = v;

    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;

    return map;
}

// --- Olmo2ModelInner ---

Olmo2ModelInner::Olmo2ModelInner(const Olmo2Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array Olmo2ModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
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

mx::array Olmo2ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Olmo2ModelInner::weight_map() {
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

// --- Olmo2Model ---

Olmo2Model::Olmo2Model(const Olmo2Configuration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult Olmo2Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput Olmo2Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Olmo2Model::forward_impl(
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

std::unordered_map<std::string, mx::array>
Olmo2Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
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

void Olmo2Model::load_weights(
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

std::unordered_map<std::string, mx::array*> Olmo2Model::weight_map() {
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
