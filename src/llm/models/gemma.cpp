// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/gemma.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, GemmaConfiguration& c) {
    c.model_type = j.value("model_type", std::string("gemma"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- GemmaRMSNorm ---

GemmaRMSNorm::GemmaRMSNorm(int dimensions, float eps)
    : weight_(mx::ones({dimensions})), eps_(eps) {}

mx::array GemmaRMSNorm::operator()(const mx::array& x) const {
    // Gemma: rms_norm(x, 1.0 + weight)
    auto adjusted = mx::add(mx::array(1.0f), weight_);
    return mx::fast::rms_norm(x, adjusted, eps_);
}

// --- GemmaAttention ---

GemmaAttention::GemmaAttention(const GemmaConfiguration& args)
    : args_(args),
      scale_(std::pow(static_cast<float>(args.head_dim), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim, args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim}))
{}

mx::array GemmaAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, args_.num_attention_heads, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, args_.num_key_value_heads, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, args_.num_key_value_heads, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, args_.head_dim, args_.rope_traditional,
                              args_.rope_theta, 1.0f, offset);
    keys = mx::fast::rope(keys, args_.head_dim, args_.rope_traditional,
                           args_.rope_theta, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> GemmaAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- GemmaMLP (uses GELU instead of SiLU) ---

GemmaMLP::GemmaMLP(int dim, int hidden)
    : gate_weight_(mx::zeros({hidden, dim})),
      down_weight_(mx::zeros({dim, hidden})),
      up_weight_(mx::zeros({hidden, dim})) {}

mx::array GemmaMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    // Gemma uses GELU activation
    auto activation = mx::multiply(g,
        mx::multiply(mx::array(0.5f),
            mx::add(mx::array(1.0f), mx::erf(mx::divide(g, mx::array(std::sqrt(2.0f)))))));
    auto up = linear_fwd(x, up_weight_);
    return linear_fwd(mx::multiply(activation, up), down_weight_);
}

std::unordered_map<std::string, mx::array*> GemmaMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- GemmaTransformerBlock ---

GemmaTransformerBlock::GemmaTransformerBlock(const GemmaConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_(args.hidden_size, args.rms_norm_eps),
      post_attention_layernorm_(args.hidden_size, args.rms_norm_eps) {}

mx::array GemmaTransformerBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = attention_(input_layernorm_(x), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(post_attention_layernorm_(h));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> GemmaTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = input_layernorm_.weight_ptr();
    map["post_attention_layernorm.weight"] = post_attention_layernorm_.weight_ptr();
    return map;
}

// --- GemmaModelInner ---

GemmaModelInner::GemmaModelInner(const GemmaConfiguration& args)
    : args_(args),
      embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_(args.hidden_size, args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array GemmaModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    // Gemma scales embeddings by sqrt(hidden_size)
    h = mx::multiply(h, mx::array(std::pow(static_cast<float>(args_.hidden_size), 0.5f)));

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return norm_(h);
}

mx::array GemmaModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> GemmaModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = norm_.weight_ptr();
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- GemmaModel ---

GemmaModel::GemmaModel(const GemmaConfiguration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

PrepareResult GemmaModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput GemmaModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array GemmaModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    // Gemma always uses tied embeddings
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
GemmaModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void GemmaModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> GemmaModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

} // namespace mlx_lm
