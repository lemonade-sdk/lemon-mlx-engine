// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4.swift

#include <mlx-lm/llm/models/glm4.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx/mlx.h>
#include <cmath>
#include <stdexcept>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Linear helper ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight,
    const mx::array* bias)
{
    return linear_forward(x, weight, bias);
}

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, GLM4Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.attention_bias = j.at("attention_bias").get<bool>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.partial_rotary_factor = j.at("partial_rotary_factor").get<float>();

    if (j.contains("rope_theta"))
        c.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("rope_traditional"))
        c.rope_traditional = j["rope_traditional"].get<bool>();
    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    if (j.contains("max_position_embeddings"))
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();
}

// --- GLM4Attention ---

GLM4Attention::GLM4Attention(const GLM4Configuration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      attention_bias_(config.attention_bias),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      rope_theta_(config.rope_theta),
      rope_traditional_(config.rope_traditional),
      rope_dims_(static_cast<int>(config.head_dim * config.partial_rotary_factor))
{
    if (config.attention_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * config.head_dim});
        wk_bias_ = mx::zeros({config.num_key_value_heads * config.head_dim});
        wv_bias_ = mx::zeros({config.num_key_value_heads * config.head_dim});
    }
}

mx::array GLM4Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_ ? &wq_bias_.value() : nullptr);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_ ? &wk_bias_.value() : nullptr);
    auto values = linear_fwd(x, wv_weight_, wv_bias_ ? &wv_bias_.value() : nullptr);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE with partial rotary dimensions
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, rope_dims_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, rope_dims_, rope_traditional_, rope_theta_, 1.0f, offset);

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

    // O projection always has bias=false
    return linear_fwd(output, wo_weight_, nullptr);
}

std::unordered_map<std::string, mx::array*> GLM4Attention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_) map["q_proj.bias"] = &wq_bias_.value();
    if (wk_bias_) map["k_proj.bias"] = &wk_bias_.value();
    if (wv_bias_) map["v_proj.bias"] = &wv_bias_.value();
    return map;
}

// --- GLM4MLP ---

GLM4MLP::GLM4MLP(const GLM4Configuration& config)
    : gate_up_weight_(mx::zeros({2 * config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      intermediate_size_(config.intermediate_size)
{}

mx::array GLM4MLP::operator()(const mx::array& x) {
    auto gu = linear_fwd(x, gate_up_weight_, nullptr);
    auto parts = mx::split(gu, 2, -1);
    return linear_fwd(swiglu(parts[0], parts[1]), down_weight_, nullptr);
}

std::unordered_map<std::string, mx::array*> GLM4MLP::weight_map() {
    return {
        {"gate_up_proj.weight", &gate_up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- GLM4DecoderLayer ---

GLM4DecoderLayer::GLM4DecoderLayer(const GLM4Configuration& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      post_self_attn_layernorm_weight_(mx::ones({config.hidden_size})),
      post_mlp_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array GLM4DecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // x = x + post_self_attn_layernorm(attention(input_layernorm(x)))
    auto h = mx::add(x, mx::fast::rms_norm(
        self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache),
        post_self_attn_layernorm_weight_, rms_norm_eps_));

    // x = post_mlp_layernorm(mlp(post_attention_layernorm(x))) + x
    auto out = mx::add(mx::fast::rms_norm(
        mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_)),
        post_mlp_layernorm_weight_, rms_norm_eps_), h);

    return out;
}

std::unordered_map<std::string, mx::array*> GLM4DecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // Attention weights
    for (auto& [k, v] : self_attn_.weight_map()) {
        map["self_attn." + k] = v;
    }

    // MLP weights
    for (auto& [k, v] : mlp_.weight_map()) {
        map["mlp." + k] = v;
    }

    // Four layernorm weights
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_self_attn_layernorm.weight"] = &post_self_attn_layernorm_weight_;
    map["post_mlp_layernorm.weight"] = &post_mlp_layernorm_weight_;

    return map;
}

// --- GLM4ModelInner ---

GLM4ModelInner::GLM4ModelInner(const GLM4Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array GLM4ModelInner::operator()(
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

std::unordered_map<std::string, mx::array*> GLM4ModelInner::weight_map() {
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

// --- GLM4Model ---

GLM4Model::GLM4Model(const GLM4Configuration& config)
    : config_(config),
      model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult GLM4Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput GLM4Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array GLM4Model::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    return mx::matmul(out, mx::transpose(lm_head_weight_));
}

std::unordered_map<std::string, mx::array>
GLM4Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Remove unused precomputed rotary frequencies
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            to_remove.push_back(k);
        }
    }
    for (const auto& k : to_remove) {
        weights.erase(k);
    }

    // If tie_word_embeddings, remove lm_head.weight
    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    return weights;
}

void GLM4Model::load_weights(
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

std::unordered_map<std::string, mx::array*> GLM4Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : model_.weight_map()) {
        map["model." + k] = v;
    }

    map["lm_head.weight"] = &lm_head_weight_;

    return map;
}

} // namespace mlx_lm
