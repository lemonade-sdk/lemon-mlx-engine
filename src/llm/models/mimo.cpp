// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of MiMo.swift

#include <mlx-lm/llm/models/mimo.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx/mlx.h>
#include <cmath>
#include <string>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, MiMoConfiguration& c) {
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

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- MiMoAttention ---

MiMoAttention::MiMoAttention(const MiMoConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      wq_bias_(mx::zeros({config.num_attention_heads * config.resolved_head_dim()})),
      wk_bias_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim()})),
      wv_bias_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim()})),
      rope_theta_(config.rope_theta),
      rope_traditional_(config.rope_traditional),
      rope_scale_(1.0f)
{
    // If rope_scaling has type="linear" and a factor, use scale = 1/factor
    if (config.rope_scaling.has_value()) {
        const auto& rs = config.rope_scaling.value();
        auto type_it = rs.find("type");
        if (type_it == rs.end()) type_it = rs.find("rope_type");
        if (type_it != rs.end() && type_it->second.is_string()
            && type_it->second.as_string() == "linear") {
            auto factor_it = rs.find("factor");
            if (factor_it != rs.end() && factor_it->second.is_float()) {
                float factor = factor_it->second.as_float();
                if (factor > 0.0f) {
                    rope_scale_ = 1.0f / factor;
                }
            }
        }
    }
}

mx::array MiMoAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys    = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values  = linear_fwd(x, wv_weight_, &wv_bias_);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_,
                              rope_theta_, rope_scale_, offset);
    keys    = mx::fast::rope(keys, head_dim_, rope_traditional_,
                              rope_theta_, rope_scale_, offset);

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

    // O projection (no bias)
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> MiMoAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- MiMoMLP ---

MiMoMLP::MiMoMLP(int dim, int hidden_dim)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array MiMoMLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> MiMoMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- MiMoTransformerBlock ---

MiMoTransformerBlock::MiMoTransformerBlock(const MiMoConfiguration& config)
    : self_attn_(config),
      mlp_(config.hidden_size, config.intermediate_size),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array MiMoTransformerBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> MiMoTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- MiMoModelInner ---

MiMoModelInner::MiMoModelInner(const MiMoConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array MiMoModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array MiMoModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> MiMoModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- MiMoModel ---

MiMoModel::MiMoModel(const MiMoConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult MiMoModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size) {
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput MiMoModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array MiMoModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value())
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
MiMoModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Remove lm_head.weight when using tied embeddings
    if (config_.tie_word_embeddings)
        weights.erase("lm_head.weight");

    // Filter rotary_emb.inv_freq, stash model.mtp_layers.* for MTPHead.
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos)
            to_remove.push_back(k);
        else if (k.rfind("model.mtp_layers.", 0) == 0) {
            mtp_weights_.emplace(k, v);
            to_remove.push_back(k);
        }
    }
    for (auto& k : to_remove)
        weights.erase(k);

    return weights;
}

void MiMoModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> MiMoModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
