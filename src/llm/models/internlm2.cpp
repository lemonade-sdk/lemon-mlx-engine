// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Internlm2.swift — InternLM2 with combined QKV, dynamic NTK RoPE

#include <mlx-lm/llm/models/internlm2.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx/mlx.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, InternLM2Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
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
    if (j.contains("bias"))
        c.bias = j["bias"].get<bool>();

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const std::optional<mx::array>& bias = std::nullopt) {
    return linear_forward(x, w, bias.has_value() ? &bias.value() : nullptr);
}

// --- InternLM2Attention ---

InternLM2Attention::InternLM2Attention(const InternLM2Configuration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      kv_groups_(config.kv_groups()),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      // Combined wqkv: (num_heads + 2*num_kv_heads) * head_dim rows
      wqkv_weight_(mx::zeros({(config.num_attention_heads + 2 * config.num_key_value_heads) * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      max_position_embeddings_(config.max_position_embeddings),
      rope_theta_(config.rope_theta),
      rope_traditional_(config.rope_traditional),
      rope_scale_(1.0f)
{
    // Optional biases
    if (config.bias) {
        wqkv_bias_ = mx::zeros({(config.num_attention_heads + 2 * config.num_key_value_heads) * config.resolved_head_dim()});
        wo_bias_ = mx::zeros({config.hidden_size});
    }

    // Determine rope_scale from rope_scaling
    if (config.rope_scaling.has_value()) {
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

mx::array InternLM2Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    // Combined QKV projection
    auto qkv = linear_fwd(x, wqkv_weight_, wqkv_bias_);

    // Split into Q, K, V along last axis
    // Q: [0, num_heads*head_dim)
    // K: [num_heads*head_dim, (num_heads+num_kv_heads)*head_dim)
    // V: [(num_heads+num_kv_heads)*head_dim, end)
    int q_dim = num_heads_ * head_dim_;
    int k_dim = num_kv_heads_ * head_dim_;

    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, q_dim});
    auto keys = mx::slice(qkv, {0, 0, q_dim}, {B, L, q_dim + k_dim});
    auto values = mx::slice(qkv, {0, 0, q_dim + k_dim}, {B, L, q_dim + 2 * k_dim});

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);

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

    return linear_fwd(output, wo_weight_, wo_bias_);
}

std::unordered_map<std::string, mx::array*> InternLM2Attention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"wqkv.weight", &wqkv_weight_},
        {"wo.weight", &wo_weight_},
    };
    if (wqkv_bias_) map["wqkv.bias"] = &wqkv_bias_.value();
    if (wo_bias_) map["wo.bias"] = &wo_bias_.value();
    return map;
}

// --- InternLM2MLP ---

InternLM2MLP::InternLM2MLP(int dim, int hidden_dim)
    : w1_weight_(mx::zeros({hidden_dim, dim})),
      w2_weight_(mx::zeros({dim, hidden_dim})),
      w3_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array InternLM2MLP::operator()(const mx::array& x) {
    // swiglu(w1(x), w3(x)) -> w2
    return linear_fwd(swiglu(linear_fwd(x, w1_weight_), linear_fwd(x, w3_weight_)), w2_weight_);
}

std::unordered_map<std::string, mx::array*> InternLM2MLP::weight_map() {
    return {
        {"w1.weight", &w1_weight_},
        {"w2.weight", &w2_weight_},
        {"w3.weight", &w3_weight_},
    };
}

// --- InternLM2TransformerBlock ---

InternLM2TransformerBlock::InternLM2TransformerBlock(const InternLM2Configuration& config)
    : attention_(config),
      feed_forward_(config.hidden_size, config.intermediate_size),
      attention_norm_weight_(mx::ones({config.hidden_size})),
      ffn_norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array InternLM2TransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Pre-norm: h = x + attn(rms_norm(x))
    auto r = attention_(mx::fast::rms_norm(x, attention_norm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);

    // out = h + mlp(rms_norm(h))
    r = feed_forward_(mx::fast::rms_norm(h, ffn_norm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> InternLM2TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // Attention weights — InternLM2 uses "attention.*" prefix
    for (auto& [k, v] : attention_.weight_map()) {
        map["attention." + k] = v;
    }

    // MLP weights — InternLM2 uses "feed_forward.*" prefix
    for (auto& [k, v] : feed_forward_.weight_map()) {
        map["feed_forward." + k] = v;
    }

    // Norm weights
    map["attention_norm.weight"] = &attention_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;

    return map;
}

// --- InternLM2ModelInner ---

InternLM2ModelInner::InternLM2ModelInner(const InternLM2Configuration& config)
    : tok_embeddings_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array InternLM2ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    // Embedding lookup
    auto h = mx::take(tok_embeddings_weight_, inputs, 0);

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array InternLM2ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(tok_embeddings_weight_));
}

std::unordered_map<std::string, mx::array*> InternLM2ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // InternLM2 uses "tok_embeddings.weight" not "embed_tokens.weight"
    map["tok_embeddings.weight"] = &tok_embeddings_weight_;
    map["norm.weight"] = &norm_weight_;

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }

    return map;
}

// --- InternLM2Model ---

InternLM2Model::InternLM2Model(const InternLM2Configuration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    // InternLM2 uses "output" not "lm_head" for the output projection
    if (!config.tie_word_embeddings) {
        output_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult InternLM2Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput InternLM2Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array InternLM2Model::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    if (output_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(output_weight_.value()));
    } else {
        return model_.embed_as_linear(out);
    }
}

std::unordered_map<std::string, mx::array>
InternLM2Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Remove precomputed rotary embedding inverse frequencies
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("attention.rope.inv_freq") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }
    return weights;
}

void InternLM2Model::load_weights(
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

std::unordered_map<std::string, mx::array*> InternLM2Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // InternLM2 weights do NOT have a "model." prefix
    for (auto& [k, v] : model_.weight_map()) {
        map[k] = v;
    }

    // InternLM2 uses "output.weight" not "lm_head.weight"
    if (output_weight_.has_value()) {
        map["output.weight"] = &output_weight_.value();
    }

    return map;
}

} // namespace mlx_lm
