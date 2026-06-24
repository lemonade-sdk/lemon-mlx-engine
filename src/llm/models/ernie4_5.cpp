// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Ernie4_5.swift

#include <mlx-lm/llm/models/ernie4_5.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx/mlx.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, Ernie45Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.use_bias = j.at("use_bias").get<bool>();

    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();

    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    else
        c.tie_word_embeddings = false;
}

// --- Linear helper ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- Ernie45Attention ---

Ernie45Attention::Ernie45Attention(const Ernie45Configuration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      rope_theta_(config.rope_theta)
{
    if (config.use_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * config.resolved_head_dim()});
        wk_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wv_bias_ = mx::zeros({config.num_key_value_heads * config.resolved_head_dim()});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array Ernie45Attention::operator()(
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

    // Apply RoPE with traditional=true
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, /*traditional=*/true,
                              rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, /*traditional=*/true,
                           rope_theta_, 1.0f, offset);

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

std::unordered_map<std::string, mx::array*> Ernie45Attention::weight_map() {
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

// --- Ernie45MLP ---

Ernie45MLP::Ernie45MLP(int dim, int hidden_dim, bool use_bias)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{
    if (use_bias) {
        gate_bias_ = mx::zeros({hidden_dim});
        down_bias_ = mx::zeros({dim});
        up_bias_ = mx::zeros({hidden_dim});
    }
}

mx::array Ernie45MLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    auto g = linear_fwd(x, gate_weight_, gate_bias_ ? &gate_bias_.value() : nullptr);
    auto up = linear_fwd(x, up_weight_, up_bias_ ? &up_bias_.value() : nullptr);
    return linear_fwd(swiglu(g, up), down_weight_,
                      down_bias_ ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> Ernie45MLP::weight_map() {
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

// --- Ernie45DecoderLayer ---

Ernie45DecoderLayer::Ernie45DecoderLayer(const Ernie45Configuration& config)
    : self_attn_(config),
      mlp_(config.hidden_size, config.intermediate_size, config.use_bias),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Ernie45DecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Pre-norm: input_layernorm before attention
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);

    // Pre-norm: post_attention_layernorm before MLP
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Ernie45DecoderLayer::weight_map() {
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

// --- Ernie45ModelInner ---

Ernie45ModelInner::Ernie45ModelInner(const Ernie45Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array Ernie45ModelInner::operator()(
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

std::unordered_map<std::string, mx::array*> Ernie45ModelInner::weight_map() {
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

// --- Ernie45Model ---

Ernie45Model::Ernie45Model(const Ernie45Configuration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult Ernie45Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput Ernie45Model::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Ernie45Model::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    }
    // Tied embeddings: use embed_tokens weight as linear head
    auto wmap = model_.weight_map();
    return mx::matmul(out, mx::transpose(*wmap["embed_tokens.weight"]));
}

std::unordered_map<std::string, mx::array>
Ernie45Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    return weights;
}

void Ernie45Model::load_weights(
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

std::unordered_map<std::string, mx::array*> Ernie45Model::weight_map() {
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
