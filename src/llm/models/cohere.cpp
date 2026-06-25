// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/cohere.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, CohereConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.layer_norm_eps = j.at("layer_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 8000000.0f);
    c.rope_traditional = j.value("rope_traditional", true);
    c.logit_scale = j.at("logit_scale").get<float>();
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- CohereAttention ---

CohereAttention::CohereAttention(const CohereConfiguration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.hidden_size / args.num_attention_heads),
      scale_(std::pow(static_cast<float>(args.hidden_size / args.num_attention_heads), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional)
{}

mx::array CohereAttention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> CohereAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- CohereMLP ---

CohereMLP::CohereMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array CohereMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> CohereMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- CohereTransformerBlock ---
// Parallel attention + MLP: out = attn(h) + mlp(h) + x

CohereTransformerBlock::CohereTransformerBlock(const CohereConfiguration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      input_layernorm_bias_(mx::zeros({args.hidden_size})),
      norm_eps_(args.layer_norm_eps)
{}

mx::array CohereTransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto h = mx::fast::layer_norm(x, input_layernorm_weight_, input_layernorm_bias_, norm_eps_);
    auto attn_out = attention_(h, mask, cache);
    auto ff_out = mlp_(h);
    return mx::add(mx::add(attn_out, ff_out), x);
}

std::unordered_map<std::string, mx::array*> CohereTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["input_layernorm.bias"] = &input_layernorm_bias_;
    return map;
}

// --- CohereModelInner ---

CohereModelInner::CohereModelInner(const CohereConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      norm_bias_(mx::zeros({args.hidden_size})),
      norm_eps_(args.layer_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array CohereModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::layer_norm(h, norm_weight_, norm_bias_, norm_eps_);
}

mx::array CohereModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> CohereModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    map["norm.bias"] = &norm_bias_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- CohereModel ---

CohereModel::CohereModel(const CohereConfiguration& args)
    : config_(args), model_(config_), logit_scale_(args.logit_scale)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

PrepareResult CohereModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput CohereModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array CohereModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    out = model_.embed_as_linear(out);
    return mx::multiply(out, mx::array(logit_scale_));
}

std::unordered_map<std::string, mx::array>
CohereModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void CohereModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> CohereModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

} // namespace mlx_lm
