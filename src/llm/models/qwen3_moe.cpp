// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Qwen3MoE.swift

#include <mlx-lm/llm/models/qwen3_moe.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <algorithm>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, Qwen3MoEConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_experts = j.at("num_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.decoder_sparse_step = j.at("decoder_sparse_step").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.norm_topk_prob = j.value("norm_topk_prob", false);

    if (j.contains("mlp_only_layers") && !j["mlp_only_layers"].is_null()) {
        c.mlp_only_layers = j["mlp_only_layers"].get<std::vector<int>>();
    }

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

// --- Qwen3MoEAttention ---

Qwen3MoEAttention::Qwen3MoEAttention(const Qwen3MoEConfiguration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      scale_(std::pow(static_cast<float>(args.head_dim), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim, args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim})),
      q_norm_weight_(mx::ones({args.head_dim})),
      k_norm_weight_(mx::ones({args.head_dim})),
      rms_norm_eps_(args.rms_norm_eps),
      rope_theta_(args.rope_theta),
      rope_scale_(1.0f)
{
    if (args.rope_scaling.has_value()) {
        auto& scaling = args.rope_scaling.value();
        auto type_it = scaling.find("type");
        if (type_it != scaling.end() && type_it->second.is_string() && type_it->second.as_string() == "linear") {
            auto factor_it = scaling.find("factor");
            if (factor_it != scaling.end() && factor_it->second.is_float()) {
                rope_scale_ = 1.0f / factor_it->second.as_float();
            }
        }
    }
}

mx::array Qwen3MoEAttention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::reshape(queries, {B, L, num_heads_, -1});
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3});

    keys = mx::reshape(keys, {B, L, num_kv_heads_, -1});
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3MoEAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- Qwen3MoEMLP ---

Qwen3MoEMLP::Qwen3MoEMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen3MoEMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3MoEMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Qwen3MoESparseMoeBlock ---

Qwen3MoESparseMoeBlock::Qwen3MoESparseMoeBlock(const Qwen3MoEConfiguration& args)
    : num_experts_(args.num_experts),
      top_k_(args.num_experts_per_tok),
      norm_topk_prob_(args.norm_topk_prob),
      gate_weight_(mx::zeros({args.num_experts, args.hidden_size})),
      switch_mlp_(args.hidden_size, args.moe_intermediate_size, args.num_experts)
{}

mx::array Qwen3MoESparseMoeBlock::operator()(const mx::array& x) {
    auto gates = linear_fwd(x, gate_weight_);
    auto soft_gates = mx::softmax(gates, -1);

    int k = top_k_;
    auto neg_gates = mx::negative(gates);
    auto inds = mx::argpartition(neg_gates, k - 1, -1);
    // Take top-k indices
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto scores = mx::take_along_axis(soft_gates, inds, -1);

    if (norm_topk_prob_) {
        scores = mx::divide(scores, mx::sum(scores, -1, true));
    }

    auto y = switch_mlp_(x, inds);
    // y has shape [B, L, k, hidden], scores has shape [B, L, k]
    // Multiply and sum over expert dim
    auto scores_expanded = mx::expand_dims(scores, -1);
    return mx::sum(mx::multiply(y, scores_expanded), -2);
}

std::unordered_map<std::string, mx::array*> Qwen3MoESparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    return map;
}

// --- Qwen3MoETransformerBlock ---

Qwen3MoETransformerBlock::Qwen3MoETransformerBlock(const Qwen3MoEConfiguration& args, int layer_idx)
    : attention_(args),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      use_moe_(false)
{
    // Determine if this layer uses MoE or dense MLP
    bool is_mlp_only = std::find(args.mlp_only_layers.begin(), args.mlp_only_layers.end(), layer_idx) != args.mlp_only_layers.end();
    if (!is_mlp_only && args.num_experts > 0 && (layer_idx + 1) % args.decoder_sparse_step == 0) {
        use_moe_ = true;
        moe_mlp_.emplace(args);
    } else {
        dense_mlp_.emplace(args.hidden_size, args.intermediate_size);
    }
}

mx::array Qwen3MoETransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    auto normed = mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_);
    if (use_moe_) {
        r = (*moe_mlp_)(normed);
    } else {
        r = (*dense_mlp_)(normed);
    }
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen3MoETransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) map["mlp." + k] = v;
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Qwen3MoEModelInner ---

Qwen3MoEModelInner::Qwen3MoEModelInner(const Qwen3MoEConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args, i);
}

mx::array Qwen3MoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen3MoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen3MoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Qwen3MoEModel ---

Qwen3MoEModel::Qwen3MoEModel(const Qwen3MoEConfiguration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Qwen3MoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Qwen3MoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Qwen3MoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Qwen3MoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Check if per-expert weights need stacking into SwitchGLU format
    if (weights.find("model.layers.0.mlp.experts.0.up_proj.weight") == weights.end()) {
        return weights;
    }

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        for (const auto& n : {"up_proj", "down_proj", "gate_proj"}) {
            std::string key0 = prefix + "experts.0." + n + ".weight";
            if (weights.find(key0) != weights.end()) {
                std::vector<mx::array> to_join;
                to_join.reserve(config_.num_experts);
                for (int e = 0; e < config_.num_experts; ++e) {
                    std::string ek = prefix + "experts." + std::to_string(e) + "." + n + ".weight";
                    auto it = weights.find(ek);
                    to_join.push_back(std::move(it->second));
                    weights.erase(it);
                }
                weights.insert_or_assign(prefix + "switch_mlp." + n + ".weight", mx::stack(to_join));
            }
        }
    }

    return weights;
}

void Qwen3MoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen3MoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
