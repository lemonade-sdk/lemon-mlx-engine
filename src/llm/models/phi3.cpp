// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/phi3.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, Phi3RopeScaling& c) {
    if (j.contains("long_factor")) c.long_factor = j["long_factor"].get<std::vector<float>>();
    if (j.contains("short_factor")) c.short_factor = j["short_factor"].get<std::vector<float>>();
    if (j.contains("factor")) c.factor = j["factor"].get<float>();
    if (j.contains("type")) c.type = j["type"].get<std::string>();
}

void from_json(const nlohmann::json& j, Phi3Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.partial_rotary_factor = j.value("partial_rotary_factor", 1.0f);
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.original_max_position_embeddings = j.at("original_max_position_embeddings").get<int>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        c.rope_scaling = j["rope_scaling"].get<Phi3RopeScaling>();
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Phi3Attention ---

Phi3Attention::Phi3Attention(const Phi3Configuration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.hidden_size / args.num_attention_heads),
      rope_dim_(static_cast<int>(static_cast<float>(args.hidden_size / args.num_attention_heads) * args.partial_rotary_factor)),
      scale_(std::pow(static_cast<float>(args.hidden_size / args.num_attention_heads), -0.5f)),
      wqkv_weight_(mx::zeros({(args.num_attention_heads + 2 * args.num_key_value_heads) * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional),
      rope_scale_(1.0f)
{
    if (args.rope_scaling.has_value()) {
        auto& rs = args.rope_scaling.value();
        if (rs.type.has_value() && rs.type.value() == "linear" && rs.factor.has_value()) {
            rope_scale_ = 1.0f / rs.factor.value();
        }
    }
}

mx::array Phi3Attention::operator()(const mx::array& x,
                                      const AttentionMask& mask,
                                      KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);
    int query_pos = num_heads_ * head_dim_;
    int kv_pos = num_kv_heads_ * head_dim_;

    auto qkv = linear_fwd(x, wqkv_weight_);
    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, query_pos});
    auto keys = mx::slice(qkv, {0, 0, query_pos}, {B, L, query_pos + kv_pos});
    auto values = mx::slice(qkv, {0, 0, query_pos + kv_pos}, {B, L, query_pos + 2 * kv_pos});

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, rope_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, rope_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Phi3Attention::weight_map() {
    return {
        {"qkv_proj.weight", &wqkv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- Phi3MLP ---

Phi3MLP::Phi3MLP(int dimensions, int hidden_dimensions)
    : gate_up_weight_(mx::zeros({2 * hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions}))
{}

mx::array Phi3MLP::operator()(const mx::array& x) {
    auto gu = linear_fwd(x, gate_up_weight_);
    int half = gu.shape(-1) / 2;
    auto gate = mx::slice(gu, {0, 0, 0}, {gu.shape(0), gu.shape(1), half});
    auto up = mx::slice(gu, {0, 0, half}, {gu.shape(0), gu.shape(1), gu.shape(2)});
    return linear_fwd(swiglu(gate, up), down_weight_);
}

std::unordered_map<std::string, mx::array*> Phi3MLP::weight_map() {
    return {
        {"gate_up_proj.weight", &gate_up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- Phi3TransformerBlock ---

Phi3TransformerBlock::Phi3TransformerBlock(const Phi3Configuration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Phi3TransformerBlock::operator()(const mx::array& x,
                                             const AttentionMask& mask,
                                             KVCache* cache) {
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Phi3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Phi3ModelInner ---

Phi3ModelInner::Phi3ModelInner(const Phi3Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Phi3ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Phi3ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Phi3ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Phi3Model ---

Phi3Model::Phi3Model(const Phi3Configuration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Phi3Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Phi3Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Phi3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    }
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Phi3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void Phi3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Phi3Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
