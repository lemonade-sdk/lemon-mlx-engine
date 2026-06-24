// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4MOE.swift

#include <mlx-lm/llm/models/glm4_moe.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, GLM4MoEConfiguration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.partial_rotary_factor = j.at("partial_rotary_factor").get<float>();
    c.use_qk_norm = j.at("use_qk_norm").get<bool>();
    c.tie_word_embeddings = j.at("tie_word_embeddings").get<bool>();
    c.attention_bias = j.at("attention_bias").get<bool>();
    c.norm_topk_prob = j.at("norm_topk_prob").get<bool>();
    c.n_group = j.at("n_group").get<int>();
    c.topk_group = j.at("topk_group").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.routed_scaling_factor = j.at("routed_scaling_factor").get<float>();
    c.first_k_dense_replace = j.at("first_k_dense_replace").get<int>();
    if (j.contains("n_routed_experts") && !j["n_routed_experts"].is_null())
        c.n_routed_experts = j["n_routed_experts"].get<int>();
    if (j.contains("n_shared_experts") && !j["n_shared_experts"].is_null())
        c.n_shared_experts = j["n_shared_experts"].get<int>();
    c.scoring_func = j.value("scoring_func", std::string("sigmoid"));
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- GLM4MoEAttention ---

GLM4MoEAttention::GLM4MoEAttention(const GLM4MoEConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads),
      rope_dim_(static_cast<int>(static_cast<float>(config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads) * config.partial_rotary_factor)),
      scale_(std::pow(static_cast<float>(config.head_dim > 0 ? config.head_dim : config.hidden_size / config.num_attention_heads), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * head_dim_, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * head_dim_, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * head_dim_, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * head_dim_})),
      q_norm_weight_(config.use_qk_norm ? mx::ones({head_dim_}) : mx::array(0.0f)),
      k_norm_weight_(config.use_qk_norm ? mx::ones({head_dim_}) : mx::array(0.0f)),
      has_qk_norm_(config.use_qk_norm),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta)
{
    if (config.attention_bias) {
        wq_bias_ = mx::zeros({config.num_attention_heads * head_dim_});
        wk_bias_ = mx::zeros({config.num_key_value_heads * head_dim_});
        wv_bias_ = mx::zeros({config.num_key_value_heads * head_dim_});
    }
}

mx::array GLM4MoEAttention::operator()(const mx::array& x,
                                          const AttentionMask& mask,
                                          KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_.has_value() ? &wq_bias_.value() : nullptr);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_.has_value() ? &wk_bias_.value() : nullptr);
    auto values = linear_fwd(x, wv_weight_, wv_bias_.has_value() ? &wv_bias_.value() : nullptr);

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
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> GLM4MoEAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_.has_value()) {
        map["q_proj.bias"] = &wq_bias_.value();
        map["k_proj.bias"] = &wk_bias_.value();
        map["v_proj.bias"] = &wv_bias_.value();
    }
    if (has_qk_norm_) {
        map["q_norm.weight"] = &q_norm_weight_;
        map["k_norm.weight"] = &k_norm_weight_;
    }
    return map;
}

// --- GLM4MoEMLP ---

GLM4MoEMLP::GLM4MoEMLP(int hidden_size, int intermediate_size)
    : gate_weight_(mx::zeros({intermediate_size, hidden_size})),
      up_weight_(mx::zeros({intermediate_size, hidden_size})),
      down_weight_(mx::zeros({hidden_size, intermediate_size}))
{}

mx::array GLM4MoEMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> GLM4MoEMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- GLM4MoEGate ---

GLM4MoEGate::GLM4MoEGate(const GLM4MoEConfiguration& config)
    : top_k_(config.num_experts_per_tok),
      n_routed_experts_(config.n_routed_experts.value()),
      n_group_(config.n_group),
      topk_group_(config.topk_group),
      norm_topk_prob_(config.norm_topk_prob),
      routed_scaling_factor_(config.routed_scaling_factor),
      scoring_func_(config.scoring_func),
      weight_(mx::zeros({config.n_routed_experts.value(), config.hidden_size})),
      e_score_correction_bias_(mx::zeros({config.n_routed_experts.value()}))
{}

std::pair<mx::array, mx::array> GLM4MoEGate::operator()(const mx::array& x) {
    auto hidden_states = mx::matmul(x, mx::transpose(weight_));

    auto scores = (scoring_func_ == "sigmoid")
        ? mx::sigmoid(mx::astype(hidden_states, mx::float32))
        : mx::softmax(mx::astype(hidden_states, mx::float32), -1);

    auto original_scores = scores;
    auto selection_scores = mx::add(scores, e_score_correction_bias_);

    if (n_group_ > 1) {
        // Reshape to groups
        auto grouped = mx::reshape(selection_scores, {
            selection_scores.shape(0), selection_scores.shape(1),
            n_group_, n_routed_experts_ / n_group_});

        // Get top-2 per group, sum for group scores
        auto top2 = mx::topk(grouped, 2, -1);
        auto group_scores = mx::sum(top2, -1, true);

        // Zero out bottom groups
        int k = n_group_ - topk_group_;
        auto group_idx = mx::argpartition(group_scores, k - 1, -2);
        group_idx = mx::slice(group_idx, {0, 0, 0, 0},
            {group_idx.shape(0), group_idx.shape(1), k, group_idx.shape(3)});

        // Flatten back
        selection_scores = mx::reshape(grouped, {
            selection_scores.shape(0), selection_scores.shape(1), -1});
    }

    int k = top_k_;
    auto inds = mx::argpartition(mx::negative(selection_scores), k - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto selected_scores = mx::take_along_axis(original_scores, inds, -1);

    if (top_k_ > 1 && norm_topk_prob_) {
        selected_scores = mx::divide(selected_scores, mx::sum(selected_scores, -1, true));
    }
    selected_scores = mx::multiply(selected_scores, mx::array(routed_scaling_factor_));

    return {inds, selected_scores};
}

std::unordered_map<std::string, mx::array*> GLM4MoEGate::weight_map() {
    return {
        {"weight", &weight_},
        {"e_score_correction_bias", &e_score_correction_bias_},
    };
}

// --- GLM4MoEMoEBlock ---

GLM4MoEMoEBlock::GLM4MoEMoEBlock(const GLM4MoEConfiguration& config)
    : gate_(config),
      switch_mlp_(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts.value())
{
    if (config.n_shared_experts.has_value() && config.n_shared_experts.value() > 0) {
        int shared_inter = config.moe_intermediate_size * config.n_shared_experts.value();
        shared_experts_.emplace(config.hidden_size, shared_inter);
    }
}

mx::array GLM4MoEMoEBlock::operator()(const mx::array& x) {
    auto [inds, scores] = gate_(x);
    auto y = switch_mlp_(x, inds);
    y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
    y = mx::astype(y, x.dtype());
    if (shared_experts_.has_value()) {
        y = mx::add(y, (*shared_experts_)(x));
    }
    return y;
}

std::unordered_map<std::string, mx::array*> GLM4MoEMoEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_.weight_map()) map["gate." + k] = v;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    if (shared_experts_.has_value()) {
        for (auto& [k, v] : shared_experts_->weight_map()) map["shared_experts." + k] = v;
    }
    return map;
}

// --- GLM4MoEBlock ---

GLM4MoEBlock::GLM4MoEBlock(const GLM4MoEConfiguration& config, int layer_idx)
    : self_attn_(config),
      use_moe_(config.n_routed_experts.has_value() && layer_idx >= config.first_k_dense_replace),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    if (use_moe_) {
        moe_mlp_.emplace(config);
    } else {
        dense_mlp_.emplace(config.hidden_size, config.intermediate_size);
    }
}

mx::array GLM4MoEBlock::operator()(const mx::array& x,
                                     const AttentionMask& mask,
                                     KVCache* cache) {
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    auto normed = mx::fast::rms_norm(h, post_attention_layernorm_weight_, norm_eps_);
    if (use_moe_) {
        r = (*moe_mlp_)(normed);
    } else {
        r = (*dense_mlp_)(normed);
    }
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> GLM4MoEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) map["mlp." + k] = v;
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- GLM4MoEModelInner ---

GLM4MoEModelInner::GLM4MoEModelInner(const GLM4MoEConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config, i);
}

mx::array GLM4MoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array GLM4MoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> GLM4MoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- GLM4MoEModel ---

GLM4MoEModel::GLM4MoEModel(const GLM4MoEConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult GLM4MoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput GLM4MoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array GLM4MoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
GLM4MoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Filter mpt layer (extra layer beyond num_hidden_layers)
    std::string mpt_prefix = "model.layers." + std::to_string(config_.num_hidden_layers);
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.substr(0, mpt_prefix.size()) == mpt_prefix) to_remove.push_back(k);
    }
    for (const auto& k : to_remove) weights.erase(k);

    // Stack per-expert weights into SwitchGLU format
    if (!config_.n_routed_experts.has_value()) return weights;
    if (weights.find("model.layers.0.mlp.experts.0.gate_proj.weight") == weights.end())
        return weights;

    int n_experts = config_.n_routed_experts.value();
    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        for (const auto& n : {"gate_proj", "down_proj", "up_proj"}) {
            for (const auto& k : {"weight", "scales", "biases"}) {
                std::string key0 = prefix + "experts.0." + n + "." + k;
                if (weights.find(key0) != weights.end()) {
                    std::vector<mx::array> to_join;
                    to_join.reserve(n_experts);
                    for (int e = 0; e < n_experts; ++e) {
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

void GLM4MoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> GLM4MoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
