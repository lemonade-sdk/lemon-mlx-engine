// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Granite.swift

#include <mlx-lm/llm/models/granite.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx/mlx.h>
#include <cmath>
#include <stdexcept>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, GraniteConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.logits_scaling = j.at("logits_scaling").get<float>();
    c.attention_multiplier = j.at("attention_multiplier").get<float>();
    c.embedding_multiplier = j.at("embedding_multiplier").get<float>();
    c.residual_multiplier = j.at("residual_multiplier").get<float>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);

    if (j.contains("max_position_embeddings"))
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();
    if (j.contains("rope_theta"))
        c.rope_theta = j["rope_theta"].get<float>();
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

// --- Linear helper ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight,
    const std::optional<mx::array>& bias)
{
    return linear_forward(x, weight, bias.has_value() ? &bias.value() : nullptr);
}

// --- GraniteAttention ---

GraniteAttention::GraniteAttention(const GraniteConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(config.attention_multiplier),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      rope_theta_(config.rope_theta),
      rope_scale_(1.0f)
{
    // Determine RoPE scale from rope_scaling config
    if (config.rope_scaling.has_value()) {
        const auto& rs = config.rope_scaling.value();
        auto type_it = rs.find("type");
        if (type_it == rs.end())
            type_it = rs.find("rope_type");
        if (type_it != rs.end() && type_it->second.is_string() &&
            type_it->second.as_string() == "linear") {
            auto factor_it = rs.find("factor");
            if (factor_it != rs.end() && factor_it->second.is_float()) {
                rope_scale_ = 1.0f / factor_it->second.as_float();
            }
        }
    }

    // Initialize optional biases
    if (config.attention_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * config.resolved_head_dim()});
        wk_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wv_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array GraniteAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_);
    auto values = linear_fwd(x, wv_weight_, wv_bias_);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, rope_scale_, offset);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // Scaled dot-product attention (scale = attention_multiplier)
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, wo_weight_, wo_bias_);
}

std::unordered_map<std::string, mx::array*> GraniteAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_) map["q_proj.bias"] = &wq_bias_.value();
    if (wk_bias_) map["k_proj.bias"] = &wk_bias_.value();
    if (wv_bias_) map["v_proj.bias"] = &wv_bias_.value();
    if (wo_bias_) map["o_proj.bias"] = &wo_bias_.value();
    return map;
}

// --- GraniteMLP ---

GraniteMLP::GraniteMLP(const GraniteConfiguration& config)
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

mx::array GraniteMLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    auto g = linear_fwd(x, gate_weight_, gate_bias_);
    auto up_out = linear_fwd(x, up_weight_, up_bias_);
    return linear_fwd(swiglu(g, up_out), down_weight_, down_bias_);
}

std::unordered_map<std::string, mx::array*> GraniteMLP::weight_map() {
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

// --- GraniteTransformerBlock ---

GraniteTransformerBlock::GraniteTransformerBlock(const GraniteConfiguration& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      residual_multiplier_(config.residual_multiplier)
{}

mx::array GraniteTransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Pre-norm + attention + scaled residual
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, mx::multiply(r, mx::array(residual_multiplier_)));

    // Pre-norm + MLP + scaled residual
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, mx::multiply(r, mx::array(residual_multiplier_)));
}

std::unordered_map<std::string, mx::array*> GraniteTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // Attention weights
    for (auto& [k, v] : self_attn_.weight_map()) {
        map["self_attn." + k] = v;
    }

    // MLP weights
    for (auto& [k, v] : mlp_.weight_map()) {
        map["mlp." + k] = v;
    }

    // LayerNorm weights
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;

    return map;
}

// --- GraniteModelInner ---

GraniteModelInner::GraniteModelInner(const GraniteConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      embedding_multiplier_(config.embedding_multiplier)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array GraniteModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    // Embedding lookup, scaled by embedding_multiplier
    auto h = mx::multiply(
        mx::take(embed_tokens_weight_, inputs, 0),
        mx::array(embedding_multiplier_));

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array GraniteModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> GraniteModelInner::weight_map() {
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

// --- GraniteModel ---

GraniteModel::GraniteModel(const GraniteConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult GraniteModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput GraniteModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array GraniteModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);

    auto logits = lm_head_weight_.has_value()
        ? mx::matmul(out, mx::transpose(lm_head_weight_.value()))
        : model_.embed_as_linear(out);

    // Scale logits by 1/logits_scaling
    return mx::divide(logits, mx::array(config_.logits_scaling));
}

std::unordered_map<std::string, mx::array>
GraniteModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    return weights;
}

void GraniteModel::load_weights(
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

std::unordered_map<std::string, mx::array*> GraniteModel::weight_map() {
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
