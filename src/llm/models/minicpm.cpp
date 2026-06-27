// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of MiniCPM.swift

#include <mlx-lm/llm/models/minicpm.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx/mlx.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, MiniCPMConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();

    if (j.contains("dim_model_base") && !j["dim_model_base"].is_null())
        c.dim_model_base = j["dim_model_base"].get<int>();
    else
        c.dim_model_base = c.hidden_size;

    if (j.contains("scale_depth") && !j["scale_depth"].is_null())
        c.scale_depth = j["scale_depth"].get<float>();
    if (j.contains("scale_emb") && !j["scale_emb"].is_null())
        c.scale_emb = j["scale_emb"].get<float>();
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

// --- Linear helper (no bias variant) ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- MiniCPMAttention ---

MiniCPMAttention::MiniCPMAttention(const MiniCPMConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      rope_theta_(config.rope_theta),
      rope_scale_(1.0f)
{
    // Handle rope_scaling: if type="linear" and factor present, rope_scale = 1/factor
    if (config.rope_scaling.has_value()) {
        const auto& rs = config.rope_scaling.value();
        auto type_it = rs.find("type");
        if (type_it == rs.end()) type_it = rs.find("rope_type");
        if (type_it != rs.end() && type_it->second.is_string() &&
            type_it->second.as_string() == "linear") {
            auto factor_it = rs.find("factor");
            if (factor_it != rs.end() && factor_it->second.is_float()) {
                rope_scale_ = 1.0f / factor_it->second.as_float();
            }
        }
    }
}

mx::array MiniCPMAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

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

    // Scaled dot-product attention
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> MiniCPMAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- MiniCPMMLP ---

MiniCPMMLP::MiniCPMMLP(int dim, int hidden_dim)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array MiniCPMMLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> MiniCPMMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- MiniCPMDecoderLayer ---

MiniCPMDecoderLayer::MiniCPMDecoderLayer(const MiniCPMConfiguration& config)
    : self_attn_(config),
      mlp_(config.hidden_size, config.intermediate_size),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      residual_scale_(config.scale_depth / std::sqrt(static_cast<float>(config.num_hidden_layers)))
{}

mx::array MiniCPMDecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // h = x + self_attn(norm(x)) * residual_scale
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_);
    auto attn_out = self_attn_(normed, mask, cache);
    auto h = mx::add(x, mx::multiply(attn_out, mx::array(residual_scale_)));

    // out = h + mlp(norm(h)) * residual_scale
    auto normed_h = mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_);
    auto mlp_out = mlp_(normed_h);
    return mx::add(h, mx::multiply(mlp_out, mx::array(residual_scale_)));
}

std::unordered_map<std::string, mx::array*> MiniCPMDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- MiniCPMModelInner ---

MiniCPMModelInner::MiniCPMModelInner(const MiniCPMConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      scale_emb_(config.scale_emb)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array MiniCPMModelInner::operator()(
    const mx::array& inputs, std::vector<KVCache>* cache)
{
    // Embedding lookup
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Scale embeddings
    if (scale_emb_ != 1.0f) {
        h = mx::multiply(h, mx::array(scale_emb_));
    }

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array MiniCPMModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> MiniCPMModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- MiniCPMModel ---

MiniCPMModel::MiniCPMModel(const MiniCPMConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);

    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult MiniCPMModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput MiniCPMModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array MiniCPMModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);

    // Output scaling: logits / (hidden_size / dim_model_base)
    float denom = static_cast<float>(config_.hidden_size) / static_cast<float>(config_.dim_model_base);
    if (denom != 0.0f) {
        out = mx::divide(out, mx::array(denom));
    }

    if (lm_head_weight_.has_value()) {
        return linear_forward(out, lm_head_weight_.value());
    } else {
        return model_.embed_as_linear(out);
    }
}

std::unordered_map<std::string, mx::array>
MiniCPMModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
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

void MiniCPMModel::load_weights(
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

std::unordered_map<std::string, mx::array*> MiniCPMModel::weight_map() {
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
