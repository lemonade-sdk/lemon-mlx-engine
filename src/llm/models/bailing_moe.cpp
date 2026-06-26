// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of BailingMoe.swift

#include <mlx-lm/llm/models/bailing_moe.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, BailingMoeConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.num_experts = j.at("num_experts").get<int>();
    c.num_shared_experts = j.at("num_shared_experts").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.first_k_dense_replace = j.at("first_k_dense_replace").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.norm_topk_prob = j.at("norm_topk_prob").get<bool>();
    c.use_bias = j.value("use_bias", false);
    c.use_qkv_bias = j.value("use_qkv_bias", false);
    c.use_qk_norm = j.value("use_qk_norm", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.partial_rotary_factor = j.value("partial_rotary_factor", 1.0f);
    c.routed_scaling_factor = j.value("routed_scaling_factor", 1.0f);
    c.score_function = j.value("score_function", std::string("softmax"));
    c.n_group = j.value("n_group", 1);
    c.topk_group = j.value("topk_group", 4);
    if (j.contains("moe_shared_expert_intermediate_size") && !j["moe_shared_expert_intermediate_size"].is_null())
        c.moe_shared_expert_intermediate_size = j["moe_shared_expert_intermediate_size"].get<int>();
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- BailingMoeAttention ---

BailingMoeAttention::BailingMoeAttention(const BailingMoeConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim()),
      rope_dim_(static_cast<int>(static_cast<float>(config.head_dim()) * config.partial_rotary_factor)),
      scale_(std::pow(static_cast<float>(config.head_dim()), -0.5f)),
      qkv_weight_(mx::zeros({(config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim()})),
      q_norm_weight_(config.use_qk_norm ? mx::ones({config.head_dim()}) : mx::array(0.0f)),
      k_norm_weight_(config.use_qk_norm ? mx::ones({config.head_dim()}) : mx::array(0.0f)),
      has_qk_norm_(config.use_qk_norm),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta)
{
    if (config.use_qkv_bias) {
        qkv_bias_ = mx::zeros({(config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim()});
    }
    if (config.use_bias) {
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array BailingMoeAttention::operator()(const mx::array& x,
                                            const AttentionMask& mask,
                                            KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);
    int q_size = num_heads_ * head_dim_;
    int kv_size = num_kv_heads_ * head_dim_;

    auto qkv = linear_fwd(x, qkv_weight_, qkv_bias_.has_value() ? &qkv_bias_.value() : nullptr);

    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, q_size});
    auto keys = mx::slice(qkv, {0, 0, q_size}, {B, L, q_size + kv_size});
    auto values = mx::slice(qkv, {0, 0, q_size + kv_size}, {B, L, q_size + 2 * kv_size});

    queries = mx::reshape(queries, {B, L, num_heads_, -1});
    keys = mx::reshape(keys, {B, L, num_kv_heads_, -1});

    if (has_qk_norm_) {
        queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
        keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    }

    queries = mx::transpose(queries, {0, 2, 1, 3});
    keys = mx::transpose(keys, {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, rope_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, rope_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, wo_bias_.has_value() ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> BailingMoeAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"query_key_value.weight", &qkv_weight_},
        {"dense.weight", &wo_weight_},
    };
    if (qkv_bias_.has_value()) map["query_key_value.bias"] = &qkv_bias_.value();
    if (wo_bias_.has_value()) map["dense.bias"] = &wo_bias_.value();
    if (has_qk_norm_) {
        map["query_layernorm.weight"] = &q_norm_weight_;
        map["key_layernorm.weight"] = &k_norm_weight_;
    }
    return map;
}

// --- BailingMoeMLP ---

BailingMoeMLP::BailingMoeMLP(int hidden_size, int inter_size, bool use_bias)
    : gate_weight_(mx::zeros({inter_size, hidden_size})),
      up_weight_(mx::zeros({inter_size, hidden_size})),
      down_weight_(mx::zeros({hidden_size, inter_size}))
{
    if (use_bias) {
        gate_bias_ = mx::zeros({inter_size});
        up_bias_ = mx::zeros({inter_size});
        down_bias_ = mx::zeros({hidden_size});
    }
}

mx::array BailingMoeMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_, gate_bias_.has_value() ? &gate_bias_.value() : nullptr);
    auto up = linear_fwd(x, up_weight_, up_bias_.has_value() ? &up_bias_.value() : nullptr);
    return linear_fwd(swiglu(g, up), down_weight_, down_bias_.has_value() ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> BailingMoeMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
    if (gate_bias_.has_value()) {
        map["gate_proj.bias"] = &gate_bias_.value();
        map["up_proj.bias"] = &up_bias_.value();
        map["down_proj.bias"] = &down_bias_.value();
    }
    return map;
}

// --- BailingMoeGate ---

BailingMoeGate::BailingMoeGate(const BailingMoeConfiguration& config)
    : top_k_(config.num_experts_per_tok),
      n_group_(config.n_group),
      topk_group_(config.topk_group),
      num_experts_(config.num_experts),
      routed_scaling_factor_(config.routed_scaling_factor),
      norm_topk_prob_(config.norm_topk_prob),
      gate_proj_weight_(mx::zeros({config.num_experts, config.hidden_size})),
      expert_bias_(mx::zeros({config.num_experts}))
{}

std::pair<mx::array, mx::array> BailingMoeGate::group_select(const mx::array& x) {
    auto logits = linear_fwd(x, gate_proj_weight_);
    auto scores = mx::sigmoid(mx::astype(logits, mx::float32));
    auto scores_for_choice = mx::add(scores, expert_bias_);

    // Group-based selection
    auto grouped = mx::reshape(scores_for_choice, {
        x.shape(0), x.shape(1), n_group_, num_experts_ / n_group_});

    auto top2 = mx::topk(grouped, 2, -1);
    auto topk_group_scores = mx::sum(top2, -1, true);

    int k = n_group_ - topk_group_;
    auto group_idx = mx::argpartition(topk_group_scores, k - 1, -2);
    group_idx = mx::slice(group_idx, {0, 0, 0, 0},
        {group_idx.shape(0), group_idx.shape(1), k, group_idx.shape(3)});

    // Flatten back and select top-k
    scores = mx::reshape(grouped, {x.shape(0), x.shape(1), -1});

    int topk = top_k_;
    auto inds = mx::argpartition(mx::negative(scores), topk - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), topk});
    auto selected = mx::take_along_axis(scores, inds, -1);

    if (top_k_ > 1 && norm_topk_prob_) {
        selected = mx::divide(selected, mx::add(mx::sum(selected, -1, true), mx::array(1e-20f)));
    }
    selected = mx::multiply(selected, mx::array(routed_scaling_factor_));
    return {inds, mx::astype(selected, logits.dtype())};
}

std::unordered_map<std::string, mx::array*> BailingMoeGate::weight_map() {
    return {
        {"gate_proj.weight", &gate_proj_weight_},
        {"expert_bias", &expert_bias_},
    };
}

// --- BailingMoeSparseMoeBlock ---

BailingMoeSparseMoeBlock::BailingMoeSparseMoeBlock(const BailingMoeConfiguration& config)
    : gate_(config),
      switch_mlp_(config.hidden_size, config.moe_intermediate_size, config.num_experts, config.use_bias)
{
    if (config.num_shared_experts > 0) {
        int shared_dim = (config.moe_shared_expert_intermediate_size.value_or(config.moe_intermediate_size))
            * config.num_shared_experts;
        shared_experts_.emplace(config.hidden_size, shared_dim, config.use_bias);
    }
}

mx::array BailingMoeSparseMoeBlock::operator()(const mx::array& x) {
    auto [inds, weights] = gate_.group_select(x);
    auto out = switch_mlp_(x, inds);
    out = mx::sum(mx::multiply(out, mx::expand_dims(weights, -1)), -2);
    if (shared_experts_.has_value()) {
        out = mx::add(out, (*shared_experts_)(x));
    }
    return out;
}

std::unordered_map<std::string, mx::array*> BailingMoeSparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_.weight_map()) map["gate." + k] = v;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    if (shared_experts_.has_value()) {
        for (auto& [k, v] : shared_experts_->weight_map()) map["shared_experts." + k] = v;
    }
    return map;
}

// --- BailingMoeBlock ---

BailingMoeBlock::BailingMoeBlock(const BailingMoeConfiguration& config, int layer_idx)
    : attention_(config),
      use_moe_(config.num_experts > 0 && layer_idx >= config.first_k_dense_replace),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    if (use_moe_) {
        moe_mlp_.emplace(config);
    } else {
        dense_mlp_.emplace(config.hidden_size, config.intermediate_size, config.use_bias);
    }
}

mx::array BailingMoeBlock::operator()(const mx::array& x,
                                        const AttentionMask& mask,
                                        KVCache* cache) {
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    auto normed = mx::fast::rms_norm(h, post_attention_layernorm_weight_, norm_eps_);
    if (use_moe_) {
        r = (*moe_mlp_)(normed);
    } else {
        r = (*dense_mlp_)(normed);
    }
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> BailingMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attention." + k] = v;
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) map["mlp." + k] = v;
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- BailingMoeModelInner ---

BailingMoeModelInner::BailingMoeModelInner(const BailingMoeConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config, i);
}

mx::array BailingMoeModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array BailingMoeModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> BailingMoeModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["word_embeddings.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- BailingMoeModel ---

BailingMoeModel::BailingMoeModel(const BailingMoeConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult BailingMoeModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput BailingMoeModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array BailingMoeModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return linear_forward(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
BailingMoeModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Stack per-expert weights into SwitchGLU format
    if (weights.find("model.layers.0.mlp.experts.0.gate_proj.weight") == weights.end())
        return weights;

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        for (const auto& n : {"gate_proj", "down_proj", "up_proj"}) {
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

void BailingMoeModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> BailingMoeModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
