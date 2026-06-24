// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of OlmoE.swift

#include <mlx-lm/llm/models/olmoe.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, OlmoEConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);
    c.attention_bias = j.value("attention_bias", false);
    c.mlp_bias = j.value("mlp_bias", false);
    c.num_experts = j.at("num_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.norm_topk_prob = j.value("norm_topk_prob", false);
    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- OlmoEAttention ---

OlmoEAttention::OlmoEAttention(const OlmoEConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.resolved_head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.resolved_head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      q_norm_weight_(mx::ones({config.num_attention_heads * config.resolved_head_dim()})),
      k_norm_weight_(mx::ones({config.num_key_value_heads * config.resolved_head_dim()})),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta),
      rope_traditional_(config.rope_traditional)
{}

mx::array OlmoEAttention::operator()(const mx::array& x,
                                       const AttentionMask& mask,
                                       KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    // Q/K norms applied before reshape (on full projection dim)
    auto queries = mx::fast::rms_norm(linear_fwd(x, wq_weight_), q_norm_weight_, rms_norm_eps_);
    auto keys = mx::fast::rms_norm(linear_fwd(x, wk_weight_), k_norm_weight_, rms_norm_eps_);
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

std::unordered_map<std::string, mx::array*> OlmoEAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_}, {"k_norm.weight", &k_norm_weight_},
    };
}

// --- OlmoESparseMoeBlock ---

OlmoESparseMoeBlock::OlmoESparseMoeBlock(const OlmoEConfiguration& config)
    : num_experts_(config.num_experts),
      top_k_(config.num_experts_per_tok),
      norm_topk_prob_(config.norm_topk_prob),
      gate_weight_(mx::zeros({config.num_experts, config.hidden_size})),
      switch_mlp_(config.hidden_size, config.intermediate_size, config.num_experts, config.mlp_bias)
{}

mx::array OlmoESparseMoeBlock::operator()(const mx::array& x) {
    auto router_logits = linear_fwd(x, gate_weight_);
    auto routing_weights = mx::softmax(router_logits, -1);

    int k = top_k_;
    auto inds = mx::argpartition(mx::negative(routing_weights), k - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto scores = mx::take_along_axis(routing_weights, inds, -1);

    if (norm_topk_prob_) {
        scores = mx::divide(scores, mx::sum(scores, -1, true));
    }

    auto y = switch_mlp_(x, inds);
    return mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
}

std::unordered_map<std::string, mx::array*> OlmoESparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    return map;
}

// --- OlmoEBlock ---

OlmoEBlock::OlmoEBlock(const OlmoEConfiguration& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array OlmoEBlock::operator()(const mx::array& x,
                                   const AttentionMask& mask,
                                   KVCache* cache) {
    auto h = mx::add(x, self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_), mask, cache));
    return mx::add(h, mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, norm_eps_)));
}

std::unordered_map<std::string, mx::array*> OlmoEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- OlmoEModelInner ---

OlmoEModelInner::OlmoEModelInner(const OlmoEConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array OlmoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array OlmoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> OlmoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- OlmoEModel ---

OlmoEModel::OlmoEModel(const OlmoEConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult OlmoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput OlmoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array OlmoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
OlmoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    if (weights.find("model.layers.0.mlp.experts.0.up_proj.weight") == weights.end())
        return weights;

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        for (const auto& n : {"up_proj", "down_proj", "gate_proj"}) {
            for (const auto& k : {"weight", "scales", "biases"}) {
                std::string key0 = prefix + "experts.0." + n + "." + k;
                if (weights.find(key0) != weights.end()) {
                    std::vector<mx::array> to_join;
                    to_join.reserve(config_.num_experts);
                    for (int e = 0; e < config_.num_experts; ++e) {
                        std::string ek = prefix + "experts." + std::to_string(e) + "." + n + "." + k;
                        auto it = weights.find(ek);
                        to_join.push_back(std::move(it->second));
                        weights.erase(it);
                    }
                    weights.insert_or_assign(prefix + "switch_mlp." + n + "." + k, mx::stack(to_join));
                }
            }
        }
    }
    return weights;
}

void OlmoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> OlmoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
