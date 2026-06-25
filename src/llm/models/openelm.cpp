// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of OpenELM.swift

#include <mlx-lm/llm/models/openelm.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Helper functions (from Swift) ---

static int compute_heads(int model_dim, int head_dim) {
    return model_dim / head_dim;
}

static int make_divisible(float v, int divisor = 8, float min_value = 0.0f) {
    float min_val = (min_value > 0.0f) ? min_value : static_cast<float>(divisor);
    int round_down = std::max(
        static_cast<int>(min_val),
        static_cast<int>((v + static_cast<float>(divisor) / 2.0f) / static_cast<float>(divisor)) * divisor
    );
    if (static_cast<float>(round_down) < 0.9f * v) {
        round_down += divisor;
    }
    return round_down;
}

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, OpenELMConfiguration& c) {
    c.head_dim = j.at("head_dim").get<int>();
    c.num_transformer_layers = j.at("num_transformer_layers").get<int>();
    c.model_dim = j.at("model_dim").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.ffn_dim_divisor = j.at("ffn_dim_divisor").get<int>();

    if (j.contains("ffn_with_glu")) c.ffn_with_glu = j["ffn_with_glu"].get<bool>();
    if (j.contains("normalize_qk_projections")) c.normalize_qk_projections = j["normalize_qk_projections"].get<bool>();
    if (j.contains("share_input_output_layers")) c.share_input_output_layers = j["share_input_output_layers"].get<bool>();

    // Defaults for multipliers and groups
    int num_gqa_groups = 4;
    std::vector<float> ffn_mult_range = {0.5f, 4.0f};
    std::vector<float> qkv_mult_range = {0.5f, 1.0f};

    if (j.contains("num_gqa_groups")) num_gqa_groups = j["num_gqa_groups"].get<int>();
    if (j.contains("ffn_multipliers")) ffn_mult_range = j["ffn_multipliers"].get<std::vector<float>>();
    if (j.contains("qkv_multiplier")) qkv_mult_range = j["qkv_multiplier"].get<std::vector<float>>();

    c.num_gqa_groups = num_gqa_groups;

    int n = c.num_transformer_layers;

    // Compute per-layer qkv multipliers via stride (if range) or use directly (if full list)
    std::vector<float> qkv_multipliers;
    if (qkv_mult_range.size() == 2 && n > 1) {
        float step = (qkv_mult_range[1] - qkv_mult_range[0]) / static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) {
            float val = qkv_mult_range[0] + step * static_cast<float>(i);
            qkv_multipliers.push_back(std::round(val * 100.0f) / 100.0f);
        }
    } else if (qkv_mult_range.size() == static_cast<size_t>(n)) {
        qkv_multipliers = qkv_mult_range;
    } else {
        qkv_multipliers.push_back(qkv_mult_range[0]);
    }

    // Use explicit num_query_heads from config if available — these match the
    // actual weight shapes in the MLX-converted model. Fall back to computing
    // from qkv_multipliers if not present.
    if (j.contains("num_query_heads") && j["num_query_heads"].is_array() &&
        j["num_query_heads"].size() == static_cast<size_t>(n)) {
        c.num_query_heads = j["num_query_heads"].get<std::vector<int>>();
        if (j.contains("num_kv_heads") && j["num_kv_heads"].is_array() &&
            j["num_kv_heads"].size() == static_cast<size_t>(n)) {
            c.kv_heads = j["num_kv_heads"].get<std::vector<int>>();
        } else {
            c.kv_heads.resize(n);
            for (int i = 0; i < n; ++i) {
                c.kv_heads[i] = c.num_query_heads[i] / num_gqa_groups;
            }
        }
    } else {
        int head_multiple_of = num_gqa_groups;
        c.num_query_heads.resize(n);
        c.kv_heads.resize(n);
        for (int i = 0; i < n; ++i) {
            int q_dim = make_divisible(
                static_cast<float>(c.model_dim) * qkv_multipliers[i],
                c.head_dim * head_multiple_of);
            c.num_query_heads[i] = compute_heads(q_dim, c.head_dim);
            c.kv_heads[i] = c.num_query_heads[i] / num_gqa_groups;
        }
    }

    // If the config provides explicit ffn_multipliers as a full per-layer list,
    // use them directly. Otherwise compute via stride from the [start, end] range.
    if (ffn_mult_range.size() == static_cast<size_t>(n)) {
        c.ffn_multipliers = ffn_mult_range;
    } else {
        c.ffn_multipliers.resize(n);
        if (n > 1) {
            float step = (ffn_mult_range[1] - ffn_mult_range[0]) / static_cast<float>(n - 1);
            for (int i = 0; i < n; ++i) {
                float val = ffn_mult_range[0] + step * static_cast<float>(i);
                c.ffn_multipliers[i] = std::round(val * 100.0f) / 100.0f;
            }
        } else {
            c.ffn_multipliers[0] = ffn_mult_range[0];
        }
    }

    if (j.contains("rms_norm_eps")) c.rms_norm_eps = j["rms_norm_eps"].get<float>();
    if (j.contains("rope_theta")) c.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("rope_freq_constant")) c.rope_theta = j["rope_freq_constant"].get<float>();
    if (j.contains("rope_traditional")) c.rope_traditional = j["rope_traditional"].get<bool>();
}

// --- Helpers ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- OpenELMAttention ---

OpenELMAttention::OpenELMAttention(const OpenELMConfiguration& config, int layer_id)
    : num_heads_(config.num_query_heads[layer_id]),
      num_kv_heads_(config.kv_heads[layer_id]),
      head_dim_(config.head_dim),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      qkv_proj_weight_(mx::zeros({(config.num_query_heads[layer_id] + 2 * config.kv_heads[layer_id]) * config.head_dim, config.model_dim})),
      out_proj_weight_(mx::zeros({config.model_dim, config.num_query_heads[layer_id] * config.head_dim})),
      q_norm_weight_(config.normalize_qk_projections ? mx::ones({config.head_dim}) : mx::array(0.0f)),
      k_norm_weight_(config.normalize_qk_projections ? mx::ones({config.head_dim}) : mx::array(0.0f)),
      has_qk_norm_(config.normalize_qk_projections),
      norm_eps_(config.rms_norm_eps),
      rope_theta_(config.rope_theta),
      rope_traditional_(config.rope_traditional)
{}

mx::array OpenELMAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    // Combined QKV projection
    auto qkv = linear_fwd(x, qkv_proj_weight_);

    // Reshape to [B, L, num_heads + 2*kv_heads, head_dim] then transpose to [B, heads, L, head_dim]
    qkv = mx::transpose(
        mx::reshape(qkv, {B, L, num_heads_ + 2 * num_kv_heads_, head_dim_}),
        {0, 2, 1, 3});

    // Split into Q, K, V along axis 1 (heads dimension)
    auto splits = mx::split(qkv, {num_heads_, num_heads_ + num_kv_heads_}, 1);
    auto queries = splits[0];
    auto keys = splits[1];
    auto values = splits[2];

    // Optional Q/K norm
    if (has_qk_norm_) {
        queries = mx::fast::rms_norm(queries, q_norm_weight_, norm_eps_);
        keys = mx::fast::rms_norm(keys, k_norm_weight_, norm_eps_);
    }

    // RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

    // KV cache update
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // SDPA
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, out_proj_weight_);
}

std::unordered_map<std::string, mx::array*> OpenELMAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"qkv_proj.weight", &qkv_proj_weight_},
        {"out_proj.weight", &out_proj_weight_},
    };
    if (has_qk_norm_) {
        map["q_norm.weight"] = &q_norm_weight_;
        map["k_norm.weight"] = &k_norm_weight_;
    }
    return map;
}

// --- OpenELMFeedForward ---

static int openelm_ffn_dim(const OpenELMConfiguration& config, int layer_id) {
    return make_divisible(
        config.ffn_multipliers[layer_id] * static_cast<float>(config.model_dim),
        config.ffn_dim_divisor);
}

OpenELMFeedForward::OpenELMFeedForward(const OpenELMConfiguration& config, int layer_id)
    : proj_1_weight_(mx::zeros({2 * openelm_ffn_dim(config, layer_id), config.model_dim})),
      proj_2_weight_(mx::zeros({config.model_dim, openelm_ffn_dim(config, layer_id)}))
{}

mx::array OpenELMFeedForward::operator()(const mx::array& x) {
    auto a = linear_fwd(x, proj_1_weight_);

    // Split into gate and value, apply SiLU gating
    auto parts = mx::split(a, 2, -1);
    auto gate = parts[0];
    auto val = parts[1];

    // swiglu(gate, val)
    return linear_fwd(swiglu(gate, val), proj_2_weight_);
}

std::unordered_map<std::string, mx::array*> OpenELMFeedForward::weight_map() {
    return {
        {"proj_1.weight", &proj_1_weight_},
        {"proj_2.weight", &proj_2_weight_},
    };
}

// --- OpenELMBlock ---

OpenELMBlock::OpenELMBlock(const OpenELMConfiguration& config, int layer_id)
    : attn_(config, layer_id),
      ffn_(config, layer_id),
      attn_norm_weight_(mx::ones({config.model_dim})),
      ffn_norm_weight_(mx::ones({config.model_dim})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array OpenELMBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Pre-norm attention
    auto r = attn_(mx::fast::rms_norm(x, attn_norm_weight_, norm_eps_), mask, cache);
    auto h = mx::add(x, r);

    // Pre-norm FFN
    r = ffn_(mx::fast::rms_norm(h, ffn_norm_weight_, norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> OpenELMBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["attn_norm.weight"] = &attn_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;
    for (auto& [k, v] : attn_.weight_map()) map["attn." + k] = v;
    for (auto& [k, v] : ffn_.weight_map()) map["ffn." + k] = v;
    return map;
}

// --- OpenELMModelInner ---

OpenELMModelInner::OpenELMModelInner(const OpenELMConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.model_dim})),
      norm_weight_(mx::ones({config.model_dim})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_transformer_layers);
    for (int i = 0; i < config.num_transformer_layers; ++i) {
        layers_.emplace_back(config, i);
    }
}

mx::array OpenELMModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array OpenELMModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> OpenELMModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["token_embeddings.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }
    return map;
}

// --- OpenELMModel ---

OpenELMModel::OpenELMModel(const OpenELMConfiguration& config)
    : config_(config),
      transformer_(config),
      lm_head_weight_(config.share_input_output_layers
          ? mx::array(0.0f)
          : mx::zeros({config.vocab_size, config.model_dim})),
      has_lm_head_(!config.share_input_output_layers),
      kv_heads_(config.kv_heads)
{}

PrepareResult OpenELMModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput OpenELMModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array OpenELMModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = transformer_(inputs, cache);
    if (has_lm_head_) {
        return linear_fwd(out, lm_head_weight_);
    }
    return transformer_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
OpenELMModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Remove rotary_emb keys
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.find("rotary_emb") != std::string::npos) {
            to_remove.push_back(k);
        }
    }
    for (const auto& k : to_remove) {
        weights.erase(k);
    }
    return weights;
}

void OpenELMModel::load_weights(
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

std::unordered_map<std::string, mx::array*> OpenELMModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    // OpenELM uses "transformer." prefix for inner model weights
    for (auto& [k, v] : transformer_.weight_map()) {
        map["transformer." + k] = v;
    }
    if (has_lm_head_) {
        map["lm_head.weight"] = &lm_head_weight_;
    }
    return map;
}

} // namespace mlx_lm
