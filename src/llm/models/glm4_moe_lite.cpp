// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GLM4MOELite.swift

#include <mlx-lm/llm/models/glm4_moe_lite.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, GLM4MoELiteConfiguration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    if (j.contains("n_shared_experts") && !j["n_shared_experts"].is_null())
        c.n_shared_experts = j["n_shared_experts"].get<int>();
    if (j.contains("n_routed_experts") && !j["n_routed_experts"].is_null())
        c.n_routed_experts = j["n_routed_experts"].get<int>();
    c.routed_scaling_factor = j.at("routed_scaling_factor").get<float>();
    c.kv_lora_rank = j.at("kv_lora_rank").get<int>();
    if (j.contains("q_lora_rank") && !j["q_lora_rank"].is_null())
        c.q_lora_rank = j["q_lora_rank"].get<int>();
    c.qk_rope_head_dim = j.at("qk_rope_head_dim").get<int>();
    c.qk_nope_head_dim = j.at("qk_nope_head_dim").get<int>();
    c.v_head_dim = j.at("v_head_dim").get<int>();
    c.topk_method = j.value("topk_method", std::string("noaux_tc"));
    c.scoring_func = j.value("scoring_func", std::string("sigmoid"));
    c.norm_topk_prob = j.value("norm_topk_prob", true);
    c.n_group = j.value("n_group", 1);
    c.topk_group = j.value("topk_group", 1);
    c.num_experts_per_tok = j.value("num_experts_per_tok", 4);
    c.moe_layer_freq = j.value("moe_layer_freq", 1);
    c.first_k_dense_replace = j.value("first_k_dense_replace", 1);
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.rope_traditional = j.value("rope_traditional", true);
    c.attention_bias = j.at("attention_bias").get<bool>();
    c.num_nextn_predict_layers = j.value("num_nextn_predict_layers", 1);

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- GLM4MoELiteAttention ---

GLM4MoELiteAttention::GLM4MoELiteAttention(const GLM4MoELiteConfiguration& config)
    : num_heads_(config.num_attention_heads),
      qk_rope_head_dim_(config.qk_rope_head_dim),
      kv_lora_rank_(config.kv_lora_rank),
      v_head_dim_(config.v_head_dim),
      qk_nope_head_dim_(config.qk_nope_head_dim),
      q_head_dim_(config.qk_nope_head_dim + config.qk_rope_head_dim),
      scale_(std::pow(static_cast<float>(config.qk_nope_head_dim + config.qk_rope_head_dim), -0.5f)),
      use_q_lora_(config.q_lora_rank.has_value()),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta),
      kv_a_proj_weight_(mx::zeros({config.kv_lora_rank + config.qk_rope_head_dim, config.hidden_size})),
      kv_a_layernorm_weight_(mx::ones({config.kv_lora_rank})),
      kv_b_proj_weight_(mx::zeros({config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim), config.kv_lora_rank})),
      o_proj_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.v_head_dim}))
{
    if (use_q_lora_) {
        int qlr = config.q_lora_rank.value();
        q_a_proj_weight_ = mx::zeros({qlr, config.hidden_size});
        if (config.attention_bias) q_a_proj_bias_ = mx::zeros({qlr});
        q_a_layernorm_weight_ = mx::ones({qlr});
        q_b_proj_weight_ = mx::zeros({num_heads_ * q_head_dim_, qlr});
    } else {
        q_proj_weight_ = mx::zeros({num_heads_ * q_head_dim_, config.hidden_size});
    }

    if (config.attention_bias) {
        kv_a_proj_bias_ = mx::zeros({config.kv_lora_rank + config.qk_rope_head_dim});
        o_proj_bias_ = mx::zeros({config.hidden_size});
    }

    // Yarn RoPE mscale correction
    if (config.rope_scaling.has_value()) {
        auto& rs = config.rope_scaling.value();
        auto get_float = [&](const std::string& key, float def) -> float {
            auto it = rs.find(key);
            return (it != rs.end() && it->second.is_float()) ? it->second.as_float() : def;
        };
        float mscale_all_dim = get_float("mscale_all_dim", 0.0f);
        float factor = get_float("factor", 1.0f);
        if (mscale_all_dim != 0.0f && factor > 1.0f) {
            float s = 0.1f * mscale_all_dim * std::log(factor) + 1.0f;
            scale_ = scale_ * s * s;
        }
    }
}

mx::array GLM4MoELiteAttention::operator()(const mx::array& x,
                                              const AttentionMask& mask,
                                              KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    // Q projection
    mx::array q = use_q_lora_
        ? linear_fwd(
            mx::fast::rms_norm(
                linear_fwd(x, q_a_proj_weight_.value(),
                           q_a_proj_bias_.has_value() ? &q_a_proj_bias_.value() : nullptr),
                q_a_layernorm_weight_.value(), rms_norm_eps_),
            q_b_proj_weight_.value())
        : linear_fwd(x, q_proj_weight_.value());

    q = mx::reshape(q, {B, L, num_heads_, q_head_dim_});
    q = mx::transpose(q, {0, 2, 1, 3});

    // Split q into nope and pe parts
    auto q_nope = mx::slice(q, {0, 0, 0, 0}, {B, num_heads_, L, qk_nope_head_dim_});
    auto q_pe = mx::slice(q, {0, 0, 0, qk_nope_head_dim_}, {B, num_heads_, L, q_head_dim_});

    // KV projection
    auto compressed_kv = linear_fwd(x, kv_a_proj_weight_,
        kv_a_proj_bias_.has_value() ? &kv_a_proj_bias_.value() : nullptr);

    // Split into kv_compressed and k_pe
    auto kv_compressed = mx::slice(compressed_kv, {0, 0, 0}, {B, L, kv_lora_rank_});
    auto k_pe = mx::slice(compressed_kv, {0, 0, kv_lora_rank_}, {B, L, kv_lora_rank_ + qk_rope_head_dim_});
    k_pe = mx::transpose(mx::reshape(k_pe, {B, L, 1, qk_rope_head_dim_}), {0, 2, 1, 3});

    // Project through kv_b after layernorm
    auto kv = linear_fwd(mx::fast::rms_norm(kv_compressed, kv_a_layernorm_weight_, rms_norm_eps_),
                          kv_b_proj_weight_);
    kv = mx::transpose(mx::reshape(kv, {B, L, num_heads_, -1}), {0, 2, 1, 3});

    // Split kv into k_nope and values
    auto k_nope = mx::slice(kv, {0, 0, 0, 0}, {B, num_heads_, L, qk_nope_head_dim_});
    auto values = mx::slice(kv, {0, 0, 0, qk_nope_head_dim_}, {B, num_heads_, L, qk_nope_head_dim_ + v_head_dim_});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    q_pe = mx::fast::rope(q_pe, qk_rope_head_dim_, false, rope_theta_, 1.0f, offset);
    k_pe = mx::fast::rope(k_pe, qk_rope_head_dim_, false, rope_theta_, 1.0f, offset);

    // Broadcast k_pe to all heads
    k_pe = mx::repeat(k_pe, num_heads_, 1);

    // Concatenate nope + pe parts
    auto keys = mx::concatenate({k_nope, k_pe}, -1);
    auto queries = mx::concatenate({q_nope, q_pe}, -1);

    // Cache update
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, o_proj_weight_,
        o_proj_bias_.has_value() ? &o_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> GLM4MoELiteAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (use_q_lora_) {
        map["q_a_proj.weight"] = &q_a_proj_weight_.value();
        if (q_a_proj_bias_.has_value()) map["q_a_proj.bias"] = &q_a_proj_bias_.value();
        map["q_a_layernorm.weight"] = &q_a_layernorm_weight_.value();
        map["q_b_proj.weight"] = &q_b_proj_weight_.value();
    } else {
        map["q_proj.weight"] = &q_proj_weight_.value();
    }
    map["kv_a_proj_with_mqa.weight"] = &kv_a_proj_weight_;
    if (kv_a_proj_bias_.has_value()) map["kv_a_proj_with_mqa.bias"] = &kv_a_proj_bias_.value();
    map["kv_a_layernorm.weight"] = &kv_a_layernorm_weight_;
    map["kv_b_proj.weight"] = &kv_b_proj_weight_;
    map["o_proj.weight"] = &o_proj_weight_;
    if (o_proj_bias_.has_value()) map["o_proj.bias"] = &o_proj_bias_.value();
    return map;
}

// --- GLM4MoELiteMLP ---

GLM4MoELiteMLP::GLM4MoELiteMLP(int hidden_size, int intermediate_size)
    : gate_weight_(mx::zeros({intermediate_size, hidden_size})),
      up_weight_(mx::zeros({intermediate_size, hidden_size})),
      down_weight_(mx::zeros({hidden_size, intermediate_size}))
{}

mx::array GLM4MoELiteMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> GLM4MoELiteMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- GLM4MoELiteGate ---

GLM4MoELiteGate::GLM4MoELiteGate(const GLM4MoELiteConfiguration& config)
    : top_k_(config.num_experts_per_tok),
      n_routed_experts_(config.n_routed_experts.value()),
      n_group_(config.n_group),
      topk_group_(config.topk_group),
      norm_topk_prob_(config.norm_topk_prob),
      routed_scaling_factor_(config.routed_scaling_factor),
      weight_(mx::zeros({config.n_routed_experts.value(), config.hidden_size})),
      e_score_correction_bias_(mx::zeros({config.n_routed_experts.value()}))
{}

std::pair<mx::array, mx::array> GLM4MoELiteGate::operator()(const mx::array& x) {
    auto hidden_states = mx::matmul(x, mx::transpose(weight_));
    auto original_scores = mx::sigmoid(mx::astype(hidden_states, mx::float32));
    auto selection_scores = mx::add(original_scores, e_score_correction_bias_);

    if (n_group_ > 1) {
        auto grouped = mx::reshape(selection_scores, {
            selection_scores.shape(0), selection_scores.shape(1),
            n_group_, n_routed_experts_ / n_group_});
        auto top2 = mx::topk(grouped, 2, -1);
        auto group_scores = mx::sum(top2, -1, true);
        int k = n_group_ - topk_group_;
        auto group_idx = mx::argpartition(group_scores, k - 1, -2);
        group_idx = mx::slice(group_idx, {0, 0, 0, 0},
            {group_idx.shape(0), group_idx.shape(1), k, group_idx.shape(3)});
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

std::unordered_map<std::string, mx::array*> GLM4MoELiteGate::weight_map() {
    return {
        {"weight", &weight_},
        {"e_score_correction_bias", &e_score_correction_bias_},
    };
}

// --- GLM4MoELiteMoE ---

GLM4MoELiteMoE::GLM4MoELiteMoE(const GLM4MoELiteConfiguration& config)
    : gate_(config),
      switch_mlp_(config.hidden_size, config.moe_intermediate_size, config.n_routed_experts.value())
{
    if (config.n_shared_experts.has_value() && config.n_shared_experts.value() > 0) {
        int shared_inter = config.moe_intermediate_size * config.n_shared_experts.value();
        shared_experts_.emplace(config.hidden_size, shared_inter);
    }
}

mx::array GLM4MoELiteMoE::operator()(const mx::array& x) {
    auto [inds, scores] = gate_(x);
    auto y = switch_mlp_(x, inds);
    y = mx::sum(mx::multiply(y, mx::expand_dims(scores, -1)), -2);
    y = mx::astype(y, x.dtype());
    if (shared_experts_.has_value()) {
        y = mx::add(y, (*shared_experts_)(x));
    }
    return y;
}

std::unordered_map<std::string, mx::array*> GLM4MoELiteMoE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_.weight_map()) map["gate." + k] = v;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    if (shared_experts_.has_value()) {
        for (auto& [k, v] : shared_experts_->weight_map()) map["shared_experts." + k] = v;
    }
    return map;
}

// --- GLM4MoELiteBlock ---

GLM4MoELiteBlock::GLM4MoELiteBlock(const GLM4MoELiteConfiguration& config, int layer_idx)
    : self_attn_(config),
      use_moe_(config.n_routed_experts.has_value() &&
               layer_idx >= config.first_k_dense_replace &&
               layer_idx % config.moe_layer_freq == 0),
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

mx::array GLM4MoELiteBlock::operator()(const mx::array& x,
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

std::unordered_map<std::string, mx::array*> GLM4MoELiteBlock::weight_map() {
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

// --- GLM4MoELiteModelInner ---

GLM4MoELiteModelInner::GLM4MoELiteModelInner(const GLM4MoELiteConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config, i);
}

mx::array GLM4MoELiteModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> GLM4MoELiteModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- GLM4MoELiteModel ---

GLM4MoELiteModel::GLM4MoELiteModel(const GLM4MoELiteConfiguration& config)
    : config_(config), model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult GLM4MoELiteModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput GLM4MoELiteModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array GLM4MoELiteModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return mx::matmul(out, mx::transpose(lm_head_weight_));
}

std::unordered_map<std::string, mx::array>
GLM4MoELiteModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Filter nextn predict layers
    int num_mpt = config_.num_nextn_predict_layers;
    if (num_mpt > 0) {
        std::vector<std::string> to_remove;
        for (auto& [k, v] : weights) {
            for (int idx = 0; idx < num_mpt; ++idx) {
                std::string prefix = "model.layers." + std::to_string(config_.num_hidden_layers + idx);
                if (k.substr(0, prefix.size()) == prefix) {
                    to_remove.push_back(k);
                    break;
                }
            }
        }
        for (const auto& k : to_remove) weights.erase(k);
    }

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

void GLM4MoELiteModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> GLM4MoELiteModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

} // namespace mlx_lm
