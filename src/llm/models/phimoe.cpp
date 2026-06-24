// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of PhiMoE.swift

#include <mlx-lm/llm/models/phimoe.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, PhiMoEConfiguration& c) {
    c.vocab_size = j.value("vocab_size", 32064);
    c.hidden_size = j.value("hidden_size", 4096);
    c.intermediate_size = j.value("intermediate_size", 6400);
    c.num_hidden_layers = j.value("num_hidden_layers", 32);
    c.num_attention_heads = j.value("num_attention_heads", 32);
    c.num_key_value_heads = j.value("num_key_value_heads", 8);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.num_local_experts = j.value("num_local_experts", 16);
    c.num_experts_per_tok = j.value("num_experts_per_tok", 2);
    c.rope_theta = j.value("rope_theta", 10000.0f);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- PhiMoEAttention ---

PhiMoEAttention::PhiMoEAttention(const PhiMoEConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim()),
      scale_(std::pow(static_cast<float>(config.head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim(), config.hidden_size})),
      wq_bias_(mx::zeros({config.num_attention_heads * config.head_dim()})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wk_bias_(mx::zeros({config.num_key_value_heads * config.head_dim()})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wv_bias_(mx::zeros({config.num_key_value_heads * config.head_dim()})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim()})),
      wo_bias_(mx::zeros({config.hidden_size})),
      rope_theta_(config.rope_theta)
{}

mx::array PhiMoEAttention::operator()(const mx::array& x,
                                        const AttentionMask& mask,
                                        KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values = linear_fwd(x, wv_weight_, &wv_bias_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> PhiMoEAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_}, {"o_proj.bias", &wo_bias_},
    };
}

// --- PhiMoESparseMoeBlock ---

PhiMoESparseMoeBlock::PhiMoESparseMoeBlock(const PhiMoEConfiguration& config)
    : num_experts_(config.num_local_experts),
      top_k_(config.num_experts_per_tok),
      gate_weight_(mx::zeros({config.num_local_experts, config.hidden_size})),
      switch_mlp_(config.hidden_size, config.intermediate_size, config.num_local_experts)
{}

mx::array PhiMoESparseMoeBlock::operator()(const mx::array& x) {
    auto gates = linear_fwd(x, gate_weight_);

    int k = top_k_;
    auto inds = mx::argpartition(mx::negative(gates), k - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    inds = mx::stop_gradient(inds);

    auto scores = mx::softmax(mx::take_along_axis(gates, inds, -1), -1);

    auto y = switch_mlp_(x, inds);
    return mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
}

std::unordered_map<std::string, mx::array*> PhiMoESparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    return map;
}

// --- PhiMoEBlock ---

PhiMoEBlock::PhiMoEBlock(const PhiMoEConfiguration& config)
    : self_attn_(config),
      block_sparse_moe_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      input_layernorm_bias_(mx::zeros({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_bias_(mx::zeros({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array PhiMoEBlock::operator()(const mx::array& x,
                                    const AttentionMask& mask,
                                    KVCache* cache) {
    auto h = mx::fast::layer_norm(x, input_layernorm_weight_, input_layernorm_bias_, norm_eps_);
    h = self_attn_(h, mask, cache);
    auto residual = mx::add(x, h);

    h = mx::fast::layer_norm(residual, post_attention_layernorm_weight_, post_attention_layernorm_bias_, norm_eps_);
    h = block_sparse_moe_(h);
    return mx::add(residual, h);
}

std::unordered_map<std::string, mx::array*> PhiMoEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : block_sparse_moe_.weight_map()) map["block_sparse_moe." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["input_layernorm.bias"] = &input_layernorm_bias_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_attention_layernorm.bias"] = &post_attention_layernorm_bias_;
    return map;
}

// --- PhiMoEModelInner ---

PhiMoEModelInner::PhiMoEModelInner(const PhiMoEConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_bias_(mx::zeros({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array PhiMoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::layer_norm(h, norm_weight_, norm_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> PhiMoEModelInner::weight_map() {
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

// --- PhiMoEModel ---

PhiMoEModel::PhiMoEModel(const PhiMoEConfiguration& config)
    : config_(config),
      model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      lm_head_bias_(mx::zeros({config.vocab_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult PhiMoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput PhiMoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array PhiMoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return linear_fwd(out, lm_head_weight_, &lm_head_bias_);
}

std::unordered_map<std::string, mx::array>
PhiMoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Stack per-expert weights into SwitchGLU format
    if (weights.find("model.layers.0.block_sparse_moe.experts.0.w1.weight") == weights.end())
        return weights;

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".block_sparse_moe.";
        for (const auto& [n, m] : std::vector<std::pair<std::string, std::string>>{
                {"w1", "gate_proj"}, {"w2", "down_proj"}, {"w3", "up_proj"}}) {
            for (const auto& k : {"weight", "scales", "biases"}) {
                std::string key0 = prefix + "experts.0." + n + "." + k;
                if (weights.find(key0) != weights.end()) {
                    std::vector<mx::array> to_join;
                    to_join.reserve(config_.num_local_experts);
                    for (int e = 0; e < config_.num_local_experts; ++e) {
                        std::string ek = prefix + "experts." + std::to_string(e) + "." + n + "." + k;
                        auto it = weights.find(ek);
                        to_join.push_back(std::move(it->second));
                        weights.erase(it);
                    }
                    weights.insert_or_assign(prefix + "switch_mlp." + m + "." + k, mx::stack(to_join));
                }
            }
        }
    }
    return weights;
}

void PhiMoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> PhiMoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    map["lm_head.bias"] = &lm_head_bias_;
    return map;
}

} // namespace mlx_lm
