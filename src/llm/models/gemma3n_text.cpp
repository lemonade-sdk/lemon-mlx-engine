// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3nText.swift
// Gemma3n: AltUp multi-stream, laurel blocks, KV sharing, per-layer embeddings,
// sliding/full attention with separate RoPE base frequencies, activation sparsity.

#include <mlx-lm/llm/models/gemma3n_text.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Helpers ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// GELU activation (exact)
static mx::array gelu(const mx::array& x) {
    return mx::multiply(
        mx::multiply(x, mx::array(0.5f)),
        mx::add(mx::array(1.0f), mx::erf(mx::divide(x, mx::array(std::sqrt(2.0f))))));
}

// RMSNorm without learned weight (RMSNoScale) — uses ones
static mx::array rms_norm_no_scale(const mx::array& x, float eps) {
    auto w = mx::ones({x.shape(-1)}, x.dtype());
    return mx::fast::rms_norm(x, w, eps);
}

// Standard deviation along axis with keepdims
static mx::array std_dev(const mx::array& x, int axis) {
    auto m = mx::mean(x, axis, true);
    return mx::sqrt(mx::mean(mx::square(mx::subtract(x, m)), axis, true));
}

// --- Configuration ---

void from_json(const nlohmann::json& j, Gemma3nTextConfiguration& c) {
    // Handle text_config nesting (VLM compatibility)
    const auto& src = j.contains("text_config") ? j["text_config"] : j;

    c.hidden_size = src.at("hidden_size").get<int>();
    c.num_hidden_layers = src.at("num_hidden_layers").get<int>();
    c.num_attention_heads = src.at("num_attention_heads").get<int>();
    c.head_dim = src.at("head_dim").get<int>();
    c.rms_norm_eps = src.at("rms_norm_eps").get<float>();
    c.vocab_size = src.at("vocab_size").get<int>();
    c.num_key_value_heads = src.at("num_key_value_heads").get<int>();
    c.num_kv_shared_layers = src.at("num_kv_shared_layers").get<int>();
    c.vocab_size_per_layer_input = src.at("vocab_size_per_layer_input").get<int>();
    c.hidden_size_per_layer_input = src.at("hidden_size_per_layer_input").get<int>();
    c.sliding_window = src.at("sliding_window").get<int>();
    c.max_position_embeddings = src.value("max_position_embeddings", 32768);
    c.rope_local_base_freq = src.at("rope_local_base_freq").get<float>();
    c.rope_theta = src.at("rope_theta").get<float>();
    c.final_logit_softcapping = src.at("final_logit_softcapping").get<float>();
    c.altup_num_inputs = src.at("altup_num_inputs").get<int>();
    c.altup_correct_scale = src.at("altup_correct_scale").get<bool>();
    c.altup_active_idx = src.at("altup_active_idx").get<int>();
    c.laurel_rank = src.at("laurel_rank").get<int>();

    // Optional fields
    if (src.contains("query_pre_attn_scalar") && !src["query_pre_attn_scalar"].is_null())
        c.query_pre_attn_scalar = src["query_pre_attn_scalar"].get<float>();
    if (src.contains("altup_coef_clip") && !src["altup_coef_clip"].is_null())
        c.altup_coef_clip = src["altup_coef_clip"].get<float>();
    if (src.contains("sliding_window_pattern") && !src["sliding_window_pattern"].is_null())
        c.sliding_window_pattern = src["sliding_window_pattern"].get<int>();

    // intermediate_size: can be int or array
    if (src["intermediate_size"].is_array()) {
        c.intermediate_size = src["intermediate_size"].get<std::vector<int>>();
    } else {
        int val = src["intermediate_size"].get<int>();
        c.intermediate_size.assign(static_cast<size_t>(c.num_hidden_layers), val);
    }

    // layer_types: optional array of strings
    if (src.contains("layer_types") && !src["layer_types"].is_null()) {
        c.layer_types = src["layer_types"].get<std::vector<std::string>>();
    }

    // activation_sparsity_pattern: optional array of floats
    if (src.contains("activation_sparsity_pattern") && !src["activation_sparsity_pattern"].is_null()) {
        c.activation_sparsity_pattern = src["activation_sparsity_pattern"].get<std::vector<float>>();
    }

    // rope_scaling
    if (src.contains("rope_scaling") && !src["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : src["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }
}

// --- Laurel Block ---

Gemma3nTextLaurelBlock::Gemma3nTextLaurelBlock(const Gemma3nTextConfiguration& config)
    : linear_left_weight_(mx::zeros({config.laurel_rank, config.hidden_size})),
      linear_right_weight_(mx::zeros({config.hidden_size, config.laurel_rank})),
      post_laurel_norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{}

mx::array Gemma3nTextLaurelBlock::operator()(const mx::array& x) {
    auto laurel_x = linear_fwd(x, linear_left_weight_);
    laurel_x = linear_fwd(laurel_x, linear_right_weight_);
    auto normed = mx::fast::rms_norm(laurel_x, post_laurel_norm_weight_, rms_norm_eps_);
    return mx::add(x, normed);
}

std::unordered_map<std::string, mx::array*> Gemma3nTextLaurelBlock::weight_map() {
    return {
        {"linear_left.weight", &linear_left_weight_},
        {"linear_right.weight", &linear_right_weight_},
        {"post_laurel_norm.weight", &post_laurel_norm_weight_},
    };
}

// --- Attention ---

Gemma3nAttention::Gemma3nAttention(const Gemma3nTextConfiguration& config, int layer_idx)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(1.0f),
      rms_norm_eps_(config.rms_norm_eps),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      q_norm_weight_(mx::ones({config.head_dim})),
      k_norm_weight_(mx::ones({config.head_dim})),
      v_norm_eps_(config.rms_norm_eps)
{
    auto lt = gemma3n_resolve_layer_types(config);
    is_sliding_ = (lt[layer_idx] == "sliding_attention");
    rope_base_freq_ = is_sliding_ ? config.rope_local_base_freq : config.rope_theta;

    int first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers;
    is_kv_shared_layer_ = (layer_idx >= first_kv_shared);
}

mx::array Gemma3nAttention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    queries = mx::reshape(queries, {B, L, num_heads_, head_dim_});
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);

    int offset = cache ? cache->offset() : 0;

    mx::array keys(0.0f), values(0.0f);

    if (is_kv_shared_layer_ && cache) {
        auto st = cache->state();
        if (st.size() >= 2) {
            keys = st[0];
            values = st[1];
        } else {
            keys = mx::reshape(linear_fwd(x, wk_weight_), {B, L, num_kv_heads_, head_dim_});
            keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
            keys = mx::transpose(keys, {0, 2, 1, 3});
            keys = mx::fast::rope(keys, head_dim_, false, rope_base_freq_, 1.0f, offset);

            values = mx::reshape(linear_fwd(x, wv_weight_), {B, L, num_kv_heads_, head_dim_});
            values = rms_norm_no_scale(values, v_norm_eps_);
            values = mx::transpose(values, {0, 2, 1, 3});

            if (cache) {
                auto [k, v] = cache->update(keys, values);
                keys = k; values = v;
            }
        }
    } else {
        keys = mx::reshape(linear_fwd(x, wk_weight_), {B, L, num_kv_heads_, head_dim_});
        keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
        keys = mx::transpose(keys, {0, 2, 1, 3});
        keys = mx::fast::rope(keys, head_dim_, false, rope_base_freq_, 1.0f, offset);

        values = mx::reshape(linear_fwd(x, wv_weight_), {B, L, num_kv_heads_, head_dim_});
        values = rms_norm_no_scale(values, v_norm_eps_);
        values = mx::transpose(values, {0, 2, 1, 3});

        if (cache) {
            auto [k, v] = cache->update(keys, values);
            keys = k; values = v;
        }
    }

    queries = mx::transpose(queries, {0, 2, 1, 3});
    queries = mx::fast::rope(queries, head_dim_, false, rope_base_freq_, 1.0f, offset);

    // Adjust mask for shared KV (key seq length may differ).
    // "causal" mode handles variable key lengths internally, so only
    // explicit array masks need adjustment.
    AttentionMask adjusted_mask = mask;
    if (mask.has_array()) {
        int keys_seq_len = keys.shape(-2);
        auto m = mask.as_array();
        if (m.shape(-1) != keys_seq_len) {
            int ndim = m.ndim();
            mx::Shape starts(ndim, 0);
            mx::Shape stops = m.shape();
            stops[ndim - 1] = keys_seq_len;
            adjusted_mask = AttentionMask::from_array(
                mx::astype(mx::slice(m, starts, stops), queries.dtype()));
        } else {
            adjusted_mask = AttentionMask::from_array(
                mx::astype(m, queries.dtype()));
        }
    }

    auto output = sdpa(queries, keys, values, scale_, adjusted_mask);

    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3nAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
}

// --- MLP ---

Gemma3nMLP::Gemma3nMLP(const Gemma3nTextConfiguration& config, int layer_idx)
    : gate_weight_(mx::zeros({config.intermediate_size[layer_idx], config.hidden_size})),
      up_weight_(mx::zeros({config.intermediate_size[layer_idx], config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size[layer_idx]})),
      hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size[layer_idx]),
      activation_sparsity_(0.0f)
{
    if (config.activation_sparsity_pattern.has_value()) {
        activation_sparsity_ = config.activation_sparsity_pattern.value()[layer_idx];
    }
    if (activation_sparsity_ > 0.0f) {
        auto arg = mx::array(2.0f * activation_sparsity_ - 1.0f);
        std_multiplier_ = mx::multiply(mx::sqrt(mx::array(2.0f)), mx::erfinv(arg));
    }
}

mx::array Gemma3nMLP::operator()(const mx::array& x) {
    auto gate_proj = linear_fwd(x, gate_weight_);

    mx::array activations(0.0f);
    if (activation_sparsity_ > 0.0f && std_multiplier_.has_value()) {
        // geluTopK: threshold activations based on mean+std
        auto inputs_mean = mx::mean(gate_proj, -1, true);
        auto inputs_std = std_dev(gate_proj, -1);
        auto cutoff = mx::add(inputs_mean, mx::multiply(inputs_std,
            mx::astype(std_multiplier_.value(), inputs_std.dtype())));
        activations = gelu(mx::maximum(mx::array(0.0f), mx::subtract(gate_proj, cutoff)));
    } else {
        activations = gelu(gate_proj);
    }

    auto up_proj = linear_fwd(x, up_weight_);
    return linear_fwd(mx::multiply(activations, up_proj), down_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3nMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- AltUp ---

Gemma3nAltUp::Gemma3nAltUp(const Gemma3nTextConfiguration& config)
    : altup_num_inputs_(config.altup_num_inputs),
      altup_active_idx_(config.altup_active_idx),
      hidden_size_(config.hidden_size),
      rms_norm_eps_(config.rms_norm_eps),
      altup_coef_clip_(config.altup_coef_clip),
      altup_correct_scale_(config.altup_correct_scale),
      correct_output_scale_weight_(mx::zeros({config.hidden_size})),
      correction_coefs_weight_(mx::zeros({config.altup_num_inputs, config.altup_num_inputs})),
      prediction_coefs_weight_(mx::zeros({config.altup_num_inputs * config.altup_num_inputs, config.altup_num_inputs})),
      modality_router_weight_(mx::zeros({config.altup_num_inputs, config.hidden_size})),
      router_norm_weight_(mx::ones({config.hidden_size})),
      router_input_scale_(mx::array(std::pow(static_cast<float>(config.hidden_size), -1.0f)))
{}

// Compute router modalities from active stream
static mx::array compute_router_modalities(
    const mx::array& x, const mx::array& router_norm_weight,
    const mx::array& router_input_scale, const mx::array& modality_router_weight,
    float rms_norm_eps)
{
    auto normed = mx::fast::rms_norm(x, router_norm_weight, rms_norm_eps);
    auto router_inputs = mx::multiply(normed, mx::astype(router_input_scale, router_norm_weight.dtype()));
    auto routed = mx::astype(linear_fwd(router_inputs, modality_router_weight), mx::float32);
    return mx::tanh(routed);
}

mx::array Gemma3nAltUp::predict(const mx::array& x) {
    int N = altup_num_inputs_;
    // x: [N, B, L, D]
    auto active = mx::squeeze(mx::slice(x, {altup_active_idx_}, {altup_active_idx_ + 1}), 0);

    auto modalities = compute_router_modalities(
        active, router_norm_weight_, router_input_scale_, modality_router_weight_, rms_norm_eps_);

    auto pred_weight = mx::astype(prediction_coefs_weight_, mx::float32);
    if (altup_coef_clip_.has_value()) {
        float clip_val = altup_coef_clip_.value();
        pred_weight = mx::clip(pred_weight, mx::array(-clip_val), mx::array(clip_val));
    }

    // modalities: [B, L, N], pred_weight: [N*N, N] → pred_weight.T: [N, N*N]
    auto raw_output = mx::matmul(modalities, mx::transpose(pred_weight));  // [B, L, N*N]

    // Reshape to [B, L, N, N] then transpose to [B, L, N, N]
    auto mod_shape = modalities.shape();
    mx::Shape coef_shape = {mod_shape[0], mod_shape[1], N, N};
    auto all_coefs = mx::transpose(mx::reshape(raw_output, coef_shape), {0, 1, 3, 2});

    // x: [N, B, L, D] → xPermuted: [B, L, D, N]
    auto x_up = mx::astype(x, mx::float32);
    auto x_permuted = mx::transpose(x_up, {1, 2, 3, 0});

    // matmul([B, L, D, N], [B, L, N, N]) → [B, L, D, N]
    auto predictions = mx::matmul(x_permuted, all_coefs);

    // [B, L, D, N] → [N, B, L, D]
    auto pred_permuted = mx::transpose(predictions, {3, 0, 1, 2});
    return mx::astype(mx::add(pred_permuted, x_up), x.dtype());
}

mx::array Gemma3nAltUp::correct(
    const mx::array& predictions, const mx::array& activated)
{
    auto modalities = compute_router_modalities(
        activated, router_norm_weight_, router_input_scale_, modality_router_weight_, rms_norm_eps_);

    auto corr_weight = mx::astype(correction_coefs_weight_, mx::float32);
    if (altup_coef_clip_.has_value()) {
        float clip_val = altup_coef_clip_.value();
        corr_weight = mx::clip(corr_weight, mx::array(-clip_val), mx::array(clip_val));
    }

    // modalities: [B, L, N], corr_weight: [N, N] → [B, L, N]
    auto all_coefs = mx::add(mx::matmul(modalities, mx::transpose(corr_weight)), mx::array(1.0f));

    // active prediction
    auto active_x = mx::squeeze(
        mx::slice(predictions, {altup_active_idx_}, {altup_active_idx_ + 1}), 0);

    auto innovation = mx::subtract(activated, active_x);  // [B, L, D]

    // all_coefs: [B, L, N] → transposed(2,1,0): [N, L, B]
    auto all_coefs_t = mx::transpose(all_coefs, {2, 1, 0});

    // [1, B, L, D] * [N, 1, L, B] → [N, B, L, D] (broadcasts for B=1)
    auto corrected = mx::multiply(
        mx::expand_dims(innovation, 0),
        mx::expand_dims(all_coefs_t, 1));

    return mx::astype(mx::add(corrected, predictions), activated.dtype());
}

std::pair<mx::array, mx::array>
Gemma3nAltUp::operator()(const mx::array& x, const mx::array& activated) {
    auto predictions = predict(x);
    auto corrected = correct(predictions, activated);
    auto output = mx::squeeze(
        mx::slice(corrected, {altup_active_idx_}, {altup_active_idx_ + 1}), 0);
    if (altup_correct_scale_) {
        output = mx::multiply(output, correct_output_scale_weight_);
    }
    return {corrected, output};
}

std::unordered_map<std::string, mx::array*> Gemma3nAltUp::weight_map() {
    return {
        {"correct_output_scale", &correct_output_scale_weight_},
        {"correction_coefs.weight", &correction_coefs_weight_},
        {"prediction_coefs.weight", &prediction_coefs_weight_},
        {"modality_router.weight", &modality_router_weight_},
        {"router_norm.weight", &router_norm_weight_},
    };
}

// --- Decoder Layer ---

Gemma3nDecoderLayer::Gemma3nDecoderLayer(const Gemma3nTextConfiguration& config, int layer_idx)
    : self_attn_(config, layer_idx),
      mlp_(config, layer_idx),
      altup_(config),
      laurel_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      pre_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({config.hidden_size})),
      post_per_layer_input_norm_weight_(mx::ones({config.hidden_size})),
      per_layer_input_gate_weight_(mx::zeros({config.hidden_size_per_layer_input, config.hidden_size})),
      per_layer_projection_weight_(mx::zeros({config.hidden_size, config.hidden_size_per_layer_input})),
      rms_norm_eps_(config.rms_norm_eps),
      hidden_size_(config.hidden_size),
      sliding_window_(config.sliding_window),
      altup_active_idx_(config.altup_active_idx),
      altup_correct_scale_(config.altup_correct_scale)
{
    auto lt = gemma3n_resolve_layer_types(config);
    is_sliding_ = (lt[layer_idx] == "sliding_attention");
}

mx::array Gemma3nDecoderLayer::operator()(
    const mx::array& x, const AttentionMask& mask,
    KVCache* cache, const mx::array& per_layer_input)
{
    // x: [N, B, L, D]
    // AltUp predict → get active prediction
    auto predictions = altup_.predict(x);
    auto active_pred = mx::squeeze(
        mx::slice(predictions, {altup_active_idx_}, {altup_active_idx_ + 1}), 0);

    // Attention path
    auto normed = mx::fast::rms_norm(active_pred, input_layernorm_weight_, rms_norm_eps_);
    auto laurel_output = laurel_(normed);
    auto attn = self_attn_(normed, mask, cache);
    auto attn_normed = mx::fast::rms_norm(attn, post_attention_layernorm_weight_, rms_norm_eps_);
    auto attn_gated = mx::add(active_pred, attn_normed);
    // Combine attention + laurel with 1/sqrt(2) scaling
    auto attn_laurel = mx::multiply(
        mx::add(attn_gated, laurel_output),
        mx::rsqrt(mx::astype(mx::array(2.0f), active_pred.dtype())));

    // MLP path
    auto ff_normed = mx::fast::rms_norm(attn_laurel, pre_feedforward_layernorm_weight_, rms_norm_eps_);
    auto ffw = mlp_(ff_normed);
    auto ffw_normed = mx::fast::rms_norm(ffw, post_feedforward_layernorm_weight_, rms_norm_eps_);
    auto activated = mx::add(attn_laurel, ffw_normed);

    // AltUp correct
    auto corrected = altup_.correct(predictions, activated);

    auto first_pred = mx::squeeze(
        mx::slice(corrected, {altup_active_idx_}, {altup_active_idx_ + 1}), 0);
    if (altup_correct_scale_) {
        first_pred = mx::multiply(first_pred, altup_.correct_output_scale());
    }

    // Per-layer input gating
    first_pred = gelu(linear_fwd(first_pred, per_layer_input_gate_weight_));
    first_pred = mx::multiply(first_pred, per_layer_input);
    first_pred = linear_fwd(first_pred, per_layer_projection_weight_);
    first_pred = mx::fast::rms_norm(first_pred, post_per_layer_input_norm_weight_, rms_norm_eps_);

    // Add to non-active streams: result[1:] += firstPrediction
    int N = corrected.shape(0);
    if (N > 1) {
        auto first = mx::slice(corrected, {0}, {1});
        auto rest = mx::slice(corrected, {1}, {N});
        rest = mx::add(rest, mx::expand_dims(first_pred, 0));
        return mx::concatenate({first, rest}, 0);
    }
    return corrected;
}

std::unordered_map<std::string, mx::array*> Gemma3nDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    for (auto& [k, v] : altup_.weight_map()) map["altup." + k] = v;
    for (auto& [k, v] : laurel_.weight_map()) map["laurel." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["pre_feedforward_layernorm.weight"] = &pre_feedforward_layernorm_weight_;
    map["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;
    map["post_per_layer_input_norm.weight"] = &post_per_layer_input_norm_weight_;
    map["per_layer_input_gate.weight"] = &per_layer_input_gate_weight_;
    map["per_layer_projection.weight"] = &per_layer_projection_weight_;
    return map;
}

// --- Inner Model ---

Gemma3nModelInner::Gemma3nModelInner(const Gemma3nTextConfiguration& config)
    : config_(config),
      embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      embed_tokens_per_layer_weight_(mx::zeros({config.vocab_size_per_layer_input,
          config.num_hidden_layers * config.hidden_size_per_layer_input})),
      per_layer_model_projection_weight_(mx::zeros({config.num_hidden_layers * config.hidden_size_per_layer_input,
          config.hidden_size})),
      per_layer_projection_norm_weight_(mx::ones({config.hidden_size_per_layer_input})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      hidden_size_(config.hidden_size),
      vocab_size_(config.vocab_size),
      vocab_size_per_layer_input_(config.vocab_size_per_layer_input),
      hidden_size_per_layer_input_(config.hidden_size_per_layer_input),
      num_hidden_layers_(config.num_hidden_layers),
      altup_num_inputs_(config.altup_num_inputs),
      altup_active_idx_(config.altup_active_idx),
      final_logit_softcapping_(config.final_logit_softcapping),
      embed_tokens_scale_(std::sqrt(static_cast<float>(config.hidden_size))),
      embed_tokens_per_layer_scale_(std::sqrt(static_cast<float>(config.hidden_size_per_layer_input)))
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config, i);

    // AltUp projections: (N-1) linear weights
    for (int i = 0; i < config.altup_num_inputs - 1; ++i)
        altup_projections_weights_.push_back(mx::zeros({config.hidden_size, config.hidden_size}));
    for (int i = 0; i < config.altup_num_inputs - 1; ++i)
        altup_unembed_projections_weights_.push_back(mx::zeros({config.hidden_size, config.hidden_size}));

    // Compute layer type mapping
    auto lt = gemma3n_resolve_layer_types(config);
    first_kv_shared_layer_idx_ = config.num_hidden_layers - config.num_kv_shared_layers;

    first_sliding_idx_ = -1;
    first_full_idx_ = -1;
    for (int i = 0; i < static_cast<int>(lt.size()); ++i) {
        if (lt[i] == "sliding_attention" && first_sliding_idx_ < 0) first_sliding_idx_ = i;
        if (lt[i] == "full_attention" && first_full_idx_ < 0) first_full_idx_ = i;
    }

    // Build layer→cache index mapping
    auto concrete_types = std::vector<std::string>(lt.begin(), lt.begin() + first_kv_shared_layer_idx_);

    int shared_full_idx = 0, shared_sliding_idx = 0;
    for (int i = static_cast<int>(concrete_types.size()) - 1; i >= 0; --i) {
        if (concrete_types[i] == "full_attention") { shared_full_idx = i; break; }
    }
    for (int i = static_cast<int>(concrete_types.size()) - 1; i >= 0; --i) {
        if (concrete_types[i] == "sliding_attention") { shared_sliding_idx = i; break; }
    }

    layer_idx_to_cache_idx_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        if (i < first_kv_shared_layer_idx_) {
            layer_idx_to_cache_idx_[i] = i;
        } else if (lt[i] == "full_attention") {
            layer_idx_to_cache_idx_[i] = shared_full_idx;
        } else {
            layer_idx_to_cache_idx_[i] = shared_sliding_idx;
        }
    }
}

mx::array Gemma3nModelInner::get_per_layer_inputs(const mx::array& input_ids) {
    auto valid_mask = mx::logical_and(
        mx::greater_equal(input_ids, mx::array(0)),
        mx::less(input_ids, mx::array(vocab_size_per_layer_input_)));
    auto tokens = mx::where(valid_mask, input_ids, mx::zeros_like(input_ids));
    auto result = mx::take(embed_tokens_per_layer_weight_, tokens, 0);
    result = mx::astype(
        mx::multiply(result, mx::array(embed_tokens_per_layer_scale_, mx::float32)),
        result.dtype());

    // Reshape: [..., num_hidden_layers * hidden_size_per_layer_input] → [..., num_hidden_layers, hidden_size_per_layer_input]
    auto s = input_ids.shape();
    mx::Shape new_shape;
    for (int i = 0; i < input_ids.ndim(); ++i) new_shape.push_back(s[i]);
    new_shape.push_back(num_hidden_layers_);
    new_shape.push_back(hidden_size_per_layer_input_);
    return mx::reshape(result, new_shape);
}

mx::array Gemma3nModelInner::project_per_layer_inputs(
    const mx::array& inputs_embeds, const mx::array& per_layer_inputs)
{
    auto proj = linear_fwd(inputs_embeds, per_layer_model_projection_weight_);
    proj = mx::multiply(proj, mx::astype(mx::array(std::pow(static_cast<float>(hidden_size_), -0.5f)),
        inputs_embeds.dtype()));

    // Reshape: [..., num_hidden_layers * hidden_size_per_layer_input] → [..., num_hidden_layers, hidden_size_per_layer_input]
    auto s = inputs_embeds.shape();
    mx::Shape new_shape;
    for (int i = 0; i < inputs_embeds.ndim() - 1; ++i) new_shape.push_back(s[i]);
    new_shape.push_back(num_hidden_layers_);
    new_shape.push_back(hidden_size_per_layer_input_);
    proj = mx::reshape(proj, new_shape);

    proj = mx::fast::rms_norm(proj, per_layer_projection_norm_weight_, rms_norm_eps_);

    // Combine with per-layer embeddings: (proj + per_layer_inputs) * 1/sqrt(2)
    return mx::multiply(
        mx::add(proj, per_layer_inputs),
        mx::astype(mx::array(std::pow(2.0f, -0.5f)), inputs_embeds.dtype()));
}

mx::array Gemma3nModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    // Embed tokens
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    h = mx::multiply(h, mx::array(embed_tokens_scale_, mx::float32));
    h = mx::astype(h, h.dtype());

    // Per-layer inputs
    auto per_layer_raw = get_per_layer_inputs(inputs);
    auto final_per_layer = project_per_layer_inputs(h, per_layer_raw);

    return forward_embeds(h, final_per_layer, cache);
}

mx::array Gemma3nModelInner::forward_embeds(
    const mx::array& inputs_embeds, const mx::array& per_layer_inputs,
    std::vector<KVCache>* cache)
{
    auto h0 = inputs_embeds;

    // Create masks
    auto lt = gemma3n_resolve_layer_types(config_);
    AttentionMask full_mask, sliding_mask;

    if (first_full_idx_ >= 0) {
        KVCache* fc = (cache && first_full_idx_ < static_cast<int>(cache->size()))
            ? &(*cache)[first_full_idx_] : nullptr;
        full_mask = create_attention_mask(h0, fc);
    }
    if (first_sliding_idx_ >= 0) {
        int sw = config_.sliding_window > 0 ? config_.sliding_window : 4096;
        KVCache* sc = (cache && first_sliding_idx_ < static_cast<int>(cache->size()))
            ? &(*cache)[first_sliding_idx_] : nullptr;
        sliding_mask = create_attention_mask(h0, sc, sw);
    }

    // Magnitude target
    auto target_mag = mx::sqrt(mx::mean(mx::square(h0), -1, true));
    auto eps_t = mx::array(std::numeric_limits<float>::min(), h0.dtype());

    // AltUp initial: create N streams, project non-active ones
    std::vector<mx::array> h_list(altup_num_inputs_, h0);
    for (int i = 1; i < altup_num_inputs_; ++i) {
        h_list[i] = mx::astype(linear_fwd(h_list[i], altup_projections_weights_[i - 1]), h0.dtype());
    }

    auto h = mx::stack(h_list, 0);  // [N, B, L, D]

    // Magnitude normalize non-active streams
    if (altup_num_inputs_ > 1) {
        auto h_rest = mx::slice(h, {1}, {h.shape(0)});
        auto mags = mx::sqrt(mx::mean(mx::square(h_rest), -1, true));
        h_rest = mx::multiply(h_rest, mx::divide(target_mag, mx::maximum(mags, eps_t)));
        h = mx::concatenate({mx::slice(h, {0}, {1}), h_rest}, 0);
    }

    // Layer loop
    for (int i = 0; i < num_hidden_layers_; ++i) {
        // Per-layer input: final_per_layer_inputs[:, :, i, :]
        auto pli_shape = per_layer_inputs.shape();
        auto per_layer_input = mx::squeeze(
            mx::slice(per_layer_inputs,
                {0, 0, i, 0},
                {pli_shape[0], pli_shape[1], i + 1, pli_shape[3]}),
            2);

        bool is_full = (lt[i] == "full_attention");
        const auto& local_mask = is_full ? full_mask : sliding_mask;

        KVCache* layer_cache = nullptr;
        if (cache) {
            int cache_idx = layer_idx_to_cache_idx_[i];
            if (cache_idx < static_cast<int>(cache->size()))
                layer_cache = &(*cache)[cache_idx];
        }

        h = layers_[i](h, local_mask, layer_cache, per_layer_input);
    }

    // AltUp final: unembed non-active streams
    auto target_mag_final = mx::sqrt(mx::mean(
        mx::square(mx::squeeze(mx::slice(h, {0}, {1}), 0)), -1, true));

    for (int i = 1; i < altup_num_inputs_; ++i) {
        auto hi = mx::squeeze(mx::slice(h, {i}, {i + 1}), 0);
        auto proj = mx::astype(linear_fwd(hi, altup_unembed_projections_weights_[i - 1]), h0.dtype());

        // Replace h[i] = proj
        std::vector<mx::array> parts;
        if (i > 0) parts.push_back(mx::slice(h, {0}, {i}));
        parts.push_back(mx::expand_dims(proj, 0));
        if (i + 1 < h.shape(0)) parts.push_back(mx::slice(h, {i + 1}, {h.shape(0)}));
        h = mx::concatenate(parts, 0);
    }

    // Magnitude normalize non-active streams again
    if (altup_num_inputs_ > 1) {
        auto h_rest = mx::slice(h, {1}, {h.shape(0)});
        auto mags = mx::sqrt(mx::mean(mx::square(h_rest), -1, true));
        h_rest = mx::multiply(h_rest, mx::divide(target_mag_final, mx::maximum(mags, eps_t)));
        h = mx::concatenate({mx::slice(h, {0}, {1}), h_rest}, 0);
    }

    // Mean across altup streams
    h = mx::mean(h, 0);  // [B, L, D]

    // Final norm
    auto out = mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);

    // Tied embeddings (embed_tokens as linear)
    out = mx::matmul(out, mx::transpose(embed_tokens_weight_));

    // Logit softcapping (compiled)
    if (final_logit_softcapping_.has_value()) {
        out = logit_softcap(out, final_logit_softcapping_.value());
    }

    return out;
}

std::vector<KVCache> Gemma3nModelInner::new_cache() const {
    auto lt = gemma3n_resolve_layer_types(config_);
    int sw = config_.sliding_window > 0 ? config_.sliding_window : 4096;

    std::vector<KVCache> caches;
    caches.reserve(first_kv_shared_layer_idx_);
    for (int i = 0; i < first_kv_shared_layer_idx_; ++i) {
        if (lt[i] == "full_attention") {
            caches.emplace_back(KVCacheSimple{});
        } else {
            caches.emplace_back(RotatingKVCache(sw, 0));
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array*> Gemma3nModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["embed_tokens_per_layer.weight"] = &embed_tokens_per_layer_weight_;
    map["per_layer_model_projection.weight"] = &per_layer_model_projection_weight_;
    map["per_layer_projection_norm.weight"] = &per_layer_projection_norm_weight_;
    map["norm.weight"] = &norm_weight_;

    for (int i = 0; i < static_cast<int>(altup_projections_weights_.size()); ++i) {
        map["altup_projections." + std::to_string(i) + ".weight"] = &altup_projections_weights_[i];
    }
    for (int i = 0; i < static_cast<int>(altup_unembed_projections_weights_.size()); ++i) {
        map["altup_unembed_projections." + std::to_string(i) + ".weight"] = &altup_unembed_projections_weights_[i];
    }

    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }

    return map;
}

// --- Top-level Model ---

Gemma3nTextModel::Gemma3nTextModel(const Gemma3nTextConfiguration& config)
    : config_(config),
      language_model_(config_)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult Gemma3nTextModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Gemma3nTextModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Gemma3nTextModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache);
}

std::vector<KVCache> Gemma3nTextModel::new_cache_impl(const GenerateParameters&) {
    return language_model_.new_cache();
}

std::unordered_map<std::string, mx::array>
Gemma3nTextModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Strip "model.language_model." prefix → "language_model."
    std::unordered_map<std::string, mx::array> processed;
    bool has_prefix = false;
    for (auto& [key, val] : weights) {
        if (key.find("model.language_model.") == 0) {
            has_prefix = true;
            processed.insert_or_assign("language_model." + key.substr(21), std::move(val));
        }
    }
    if (has_prefix) {
        // Keep non-prefixed weights too
        for (auto& [key, val] : weights) {
            if (key.find("model.language_model.") != 0) {
                processed.insert_or_assign(key, std::move(val));
            }
        }
        weights = std::move(processed);
    }
    return weights;
}

void Gemma3nTextModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Gemma3nTextModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;
    return map;
}

} // namespace mlx_lm
