// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of AfMoE.swift

#include <mlx-lm/llm/models/afmoe.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, AfMoEConfiguration& c) {
    c.vocab_size = j.value("vocab_size", 200192);
    c.hidden_size = j.value("hidden_size", 2048);
    c.intermediate_size = j.value("intermediate_size", 6144);
    c.moe_intermediate_size = j.value("moe_intermediate_size", 1024);
    c.num_hidden_layers = j.value("num_hidden_layers", 32);
    c.num_attention_heads = j.value("num_attention_heads", 32);
    c.num_key_value_heads = j.value("num_key_value_heads", 4);
    c.head_dim = j.value("head_dim", 64);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-5f);
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.num_experts = j.value("num_experts", 128);
    c.num_experts_per_tok = j.value("num_experts_per_tok", 8);
    c.num_shared_experts = j.value("num_shared_experts", 1);
    c.num_dense_layers = j.value("num_dense_layers", 2);
    c.route_norm = j.value("route_norm", true);
    c.route_scale = j.value("route_scale", 2.826f);
    c.score_func = j.value("score_func", std::string("sigmoid"));
    c.n_group = j.value("n_group", 1);
    c.topk_group = j.value("topk_group", 1);
    c.layer_types = j.at("layer_types").get<std::vector<std::string>>();
    c.sliding_window = j.value("sliding_window", 2048);
    c.mup_enabled = j.value("mup_enabled", true);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- AfMoEAttention ---

AfMoEAttention::AfMoEAttention(const AfMoEConfiguration& config, bool is_local_attention)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      is_local_attention_(is_local_attention),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      q_norm_weight_(mx::ones({config.head_dim})),
      k_norm_weight_(mx::ones({config.head_dim})),
      gate_proj_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta),
      has_rope_(is_local_attention)
{}

mx::array AfMoEAttention::operator()(const mx::array& x,
                                       const AttentionMask& mask,
                                       KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Q/K norm
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);

    // RoPE only for local (sliding window) attention
    if (has_rope_) {
        int offset = cache ? cache->offset() : 0;
        queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
        keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);
    }

    // Cache update
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    // Attention gating
    auto gate = mx::sigmoid(linear_fwd(x, gate_proj_weight_));
    output = mx::multiply(output, gate);

    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> AfMoEAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_}, {"k_norm.weight", &k_norm_weight_},
        {"gate_proj.weight", &gate_proj_weight_},
    };
}

// --- AfMoEMLP ---

AfMoEMLP::AfMoEMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions}))
{}

mx::array AfMoEMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> AfMoEMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- AfMoEMoE ---

AfMoEMoE::AfMoEMoE(const AfMoEConfiguration& config)
    : num_experts_(config.num_experts),
      num_experts_per_tok_(config.num_experts_per_tok),
      n_group_(config.n_group),
      topk_group_(config.topk_group),
      route_norm_(config.route_norm),
      route_scale_(config.route_scale),
      score_func_(config.score_func),
      router_gate_weight_(mx::zeros({config.num_experts, config.hidden_size})),
      expert_bias_(mx::zeros({config.num_experts})),
      experts_(config.hidden_size, config.moe_intermediate_size, config.num_experts)
{
    if (config.num_shared_experts > 0) {
        int shared_inter = config.moe_intermediate_size * config.num_shared_experts;
        shared_experts_.emplace(config.hidden_size, shared_inter);
    }
}

mx::array AfMoEMoE::operator()(const mx::array& x) {
    auto gates = linear_fwd(x, router_gate_weight_);

    auto scores = (score_func_ == "sigmoid")
        ? mx::sigmoid(mx::astype(gates, mx::float32))
        : mx::softmax(mx::astype(gates, mx::float32), -1);

    auto selection_scores = mx::add(scores, expert_bias_);

    // Group-based expert selection if n_group > 1
    if (n_group_ > 1) {
        auto grouped = mx::reshape(selection_scores, {
            x.shape(0), x.shape(1), n_group_, num_experts_ / n_group_});
        auto top2 = mx::topk(grouped, 2, -1);
        auto group_scores = mx::sum(top2, -1, true);
        int k = n_group_ - topk_group_;
        auto group_idx = mx::argpartition(group_scores, k - 1, -2);
        group_idx = mx::slice(group_idx, {0, 0, 0, 0},
            {group_idx.shape(0), group_idx.shape(1), k, group_idx.shape(3)});
        selection_scores = mx::reshape(grouped, {x.shape(0), x.shape(1), -1});
    }

    int k = num_experts_per_tok_;
    auto inds = mx::argpartition(mx::negative(selection_scores), k - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});

    auto selected_scores = mx::take_along_axis(scores, inds, -1);

    if (route_norm_ && num_experts_per_tok_ > 1) {
        selected_scores = mx::divide(selected_scores, mx::sum(selected_scores, -1, true));
    }
    selected_scores = mx::multiply(selected_scores, mx::array(route_scale_));

    auto y = experts_(x, inds);
    y = mx::sum(mx::multiply(y, mx::expand_dims(selected_scores, -1)), -2);
    y = mx::astype(y, x.dtype());

    if (shared_experts_.has_value()) {
        y = mx::add(y, (*shared_experts_)(x));
    }
    return y;
}

std::unordered_map<std::string, mx::array*> AfMoEMoE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["router.gate.weight"] = &router_gate_weight_;
    map["expert_bias"] = &expert_bias_;
    for (auto& [k, v] : experts_.weight_map()) map["experts." + k] = v;
    if (shared_experts_.has_value()) {
        for (auto& [k, v] : shared_experts_->weight_map()) map["shared_experts." + k] = v;
    }
    return map;
}

// --- AfMoEBlock ---

AfMoEBlock::AfMoEBlock(const AfMoEConfiguration& config, int layer_idx, bool use_sliding)
    : self_attn_(config, use_sliding),
      use_moe_(layer_idx >= config.num_dense_layers),
      use_sliding_(use_sliding),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      pre_mlp_layernorm_weight_(mx::ones({config.hidden_size})),
      post_mlp_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    if (use_moe_) {
        moe_mlp_.emplace(config);
    } else {
        dense_mlp_.emplace(config.hidden_size, config.intermediate_size);
    }
}

mx::array AfMoEBlock::operator()(const mx::array& x,
                                   const AttentionMask& mask,
                                   KVCache* cache) {
    // Attention with pre and post norm
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_), mask, cache);
    r = mx::fast::rms_norm(r, post_attention_layernorm_weight_, norm_eps_);
    auto h = mx::add(x, r);

    // MLP with pre and post norm
    auto normed = mx::fast::rms_norm(h, pre_mlp_layernorm_weight_, norm_eps_);
    if (use_moe_) {
        r = (*moe_mlp_)(normed);
    } else {
        r = (*dense_mlp_)(normed);
    }
    r = mx::fast::rms_norm(r, post_mlp_layernorm_weight_, norm_eps_);
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> AfMoEBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) map["mlp." + k] = v;
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["pre_mlp_layernorm.weight"] = &pre_mlp_layernorm_weight_;
    map["post_mlp_layernorm.weight"] = &post_mlp_layernorm_weight_;
    return map;
}

// --- AfMoEModelInner ---

AfMoEModelInner::AfMoEModelInner(const AfMoEConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps),
      mup_enabled_(config.mup_enabled),
      hidden_size_(config.hidden_size),
      fa_idx_(0),
      swa_idx_(-1),
      sliding_window_(config.sliding_window)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < static_cast<int>(config.layer_types.size()); ++i) {
        bool use_sliding = (config.layer_types[i] == "sliding_attention");
        layers_.emplace_back(config, i, use_sliding);
        if (!use_sliding && fa_idx_ == 0) fa_idx_ = i;
        if (use_sliding && swa_idx_ < 0) swa_idx_ = i;
    }
    // Find first full_attention index
    for (int i = 0; i < static_cast<int>(config.layer_types.size()); ++i) {
        if (config.layer_types[i] == "full_attention") { fa_idx_ = i; break; }
    }
}

mx::array AfMoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // muP scaling
    if (mup_enabled_) {
        h = mx::multiply(h, mx::array(std::sqrt(static_cast<float>(hidden_size_))));
    }

    // Create attention masks
    auto fa_mask = create_attention_mask(h, cache && fa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[fa_idx_] : nullptr);

    AttentionMask swa_mask;
    if (swa_idx_ >= 0) {
        swa_mask = create_attention_mask(h,
            cache && swa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[swa_idx_] : nullptr,
            sliding_window_);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        auto& mask = layers_[i].uses_sliding() ? swa_mask : fa_mask;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array AfMoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> AfMoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- AfMoEModel ---

AfMoEModel::AfMoEModel(const AfMoEConfiguration& config)
    : config_(config), model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
    for (const auto& lt : config.layer_types) {
        layer_uses_sliding_.push_back(lt == "sliding_attention");
    }
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult AfMoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput AfMoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array AfMoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    return model_.embed_as_linear(out);
}

std::vector<KVCache> AfMoEModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(layer_uses_sliding_.size());
    for (bool uses_sliding : layer_uses_sliding_) {
        if (uses_sliding) {
            caches.emplace_back(RotatingKVCache(config_.sliding_window));
        } else {
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            } else {
                caches.emplace_back(KVCacheSimple{});
            }
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
AfMoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Remove rotary_emb keys
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.find("rotary_emb.inv_freq") != std::string::npos) to_remove.push_back(k);
    }
    for (const auto& k : to_remove) weights.erase(k);

    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Stack per-expert weights into SwitchGLU format
    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        if (l < config_.num_dense_layers) continue;
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
                    weights.insert_or_assign(prefix + "experts." + n + "." + k, mx::stack(to_join));
                }
            }
        }
    }
    return weights;
}

void AfMoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> AfMoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
