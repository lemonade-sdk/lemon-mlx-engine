// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Apertus.swift
// Apertus uses XIELU (learned activation), Q/K RMSNorm, and LlamaDynamicNTKScalingRoPE.

#include <mlx-lm/llm/models/apertus.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, ApertusConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);

    if (j.contains("max_position_embeddings") && !j["max_position_embeddings"].is_null())
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();

    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.rope_traditional = j.value("rope_traditional", false);

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- XIELU (Expanded Integral of the Exponential Linear Unit) ---

XIELU::XIELU()
    : alpha_p_(mx::array(0.55f)),
      alpha_n_(mx::array(0.55f)),
      beta_(mx::array(0.5f)),
      eps_(mx::array(-1e-6f))
{}

mx::array XIELU::operator()(const mx::array& x) {
    // softplus(alpha_p) and beta + softplus(alpha_n)
    auto sp_alpha_p = mx::log(mx::add(mx::array(1.0f), mx::exp(alpha_p_)));
    auto sp_alpha_n = mx::log(mx::add(mx::array(1.0f), mx::exp(alpha_n_)));
    auto alpha_n_eff = mx::add(beta_, sp_alpha_n);

    // pos = softplus(alpha_p) * x^2 + beta * x
    auto pos = mx::add(
        mx::multiply(sp_alpha_p, mx::square(x)),
        mx::multiply(beta_, x));

    // neg = alpha_n_eff * (exp(min(x, eps)) - 1) - alpha_n_eff * x + beta * x
    auto neg = mx::add(
        mx::subtract(
            mx::multiply(alpha_n_eff, mx::subtract(mx::exp(mx::minimum(x, eps_)), mx::array(1.0f))),
            mx::multiply(alpha_n_eff, x)),
        mx::multiply(beta_, x));

    // Combine: where(x >= 0, pos, neg)
    return mx::where(mx::greater(x, mx::array(0.0f)), pos, neg);
}

std::unordered_map<std::string, mx::array*> XIELU::weight_map() {
    return {
        {"alpha_p", &alpha_p_},
        {"alpha_n", &alpha_n_},
        {"beta", &beta_},
        {"eps", &eps_},
    };
}

// --- ApertusAttention ---

ApertusAttention::ApertusAttention(const ApertusConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      q_norm_weight_(mx::ones({config.resolved_head_dim()})),
      k_norm_weight_(mx::ones({config.resolved_head_dim()})),
      rms_norm_eps_(config.rms_norm_eps),
      rope_(config.resolved_head_dim(),
            config.max_position_embeddings,
            config.rope_traditional,
            config.rope_theta,
            1.0f,
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
{}

mx::array ApertusAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Reshape: [B, L, heads*head_dim] -> [B, L, heads, head_dim]
    queries = mx::reshape(queries, {B, L, num_heads_, head_dim_});
    keys = mx::reshape(keys, {B, L, num_kv_heads_, head_dim_});
    values = mx::reshape(values, {B, L, num_kv_heads_, head_dim_});

    // Apply Q/K RMSNorm (standard, not Gemma +1 style)
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);

    // Transpose to [B, heads, L, head_dim] for RoPE and SDPA
    queries = mx::transpose(queries, {0, 2, 1, 3});
    keys = mx::transpose(keys, {0, 2, 1, 3});
    values = mx::transpose(values, {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys = rope_(keys, offset);

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

std::unordered_map<std::string, mx::array*> ApertusAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- ApertusMLP ---
// up projection -> XIELU activation -> down projection (NO gate)

ApertusMLP::ApertusMLP(int dim, int hidden_dim)
    : up_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim}))
{}

mx::array ApertusMLP::operator()(const mx::array& x) {
    return linear_fwd(act_(linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> ApertusMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
    for (auto& [k, v] : act_.weight_map()) {
        map["act_fn." + k] = v;
    }
    return map;
}

// --- ApertusBlock ---
// Pre-norm: h = x + attn(rms_norm(x, attention_layernorm))
//           out = h + mlp(rms_norm(h, feedforward_layernorm))

ApertusBlock::ApertusBlock(const ApertusConfiguration& config)
    : self_attn_(config),
      mlp_(config.hidden_size, config.intermediate_size),
      attention_layernorm_weight_(mx::ones({config.hidden_size})),
      feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array ApertusBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    auto r = self_attn_(mx::fast::rms_norm(x, attention_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, feedforward_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> ApertusBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["attention_layernorm.weight"] = &attention_layernorm_weight_;
    map["feedforward_layernorm.weight"] = &feedforward_layernorm_weight_;
    return map;
}

// --- ApertusModelInner ---

ApertusModelInner::ApertusModelInner(const ApertusConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array ApertusModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array ApertusModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> ApertusModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- ApertusModel ---

ApertusModel::ApertusModel(const ApertusConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult ApertusModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput ApertusModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array ApertusModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    }
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
ApertusModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Remove unused precomputed rotary frequencies
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    // If tie_word_embeddings, erase lm_head.weight
    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    return weights;
}

void ApertusModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> ApertusModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) {
        map["lm_head.weight"] = &lm_head_weight_.value();
    }
    return map;
}

} // namespace mlx_lm
