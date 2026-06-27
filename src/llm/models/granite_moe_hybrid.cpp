// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++
// Port of GraniteMoeHybrid.swift

#include <mlx-lm/llm/models/granite_moe_hybrid.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/ssm_utils.h>
#include <mlx-lm/common/activations.h>
#include <algorithm>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, GraniteMoeHybridConfiguration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.attention_bias = j.at("attention_bias").get<bool>();
    c.embedding_multiplier = j.at("embedding_multiplier").get<float>();
    c.attention_multiplier = j.at("attention_multiplier").get<float>();
    c.logits_scaling = j.at("logits_scaling").get<float>();
    c.residual_multiplier = j.at("residual_multiplier").get<float>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.layer_types = j.at("layer_types").get<std::vector<std::string>>();

    c.mlp_bias = j.value("mlp_bias", false);
    c.position_embedding_type = j.value("position_embedding_type", std::string("rope"));
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);

    // Optional MoE params
    if (j.contains("num_local_experts") && !j["num_local_experts"].is_null())
        c.num_local_experts = j["num_local_experts"].get<int>();
    if (j.contains("num_experts_per_tok") && !j["num_experts_per_tok"].is_null())
        c.num_experts_per_token = j["num_experts_per_tok"].get<int>();
    if (j.contains("shared_intermediate_size") && !j["shared_intermediate_size"].is_null())
        c.shared_intermediate_size = j["shared_intermediate_size"].get<int>();

    // Optional Mamba params
    if (j.contains("mamba_n_heads") && !j["mamba_n_heads"].is_null())
        c.mamba_heads = j["mamba_n_heads"].get<int>();
    if (j.contains("mamba_d_head") && !j["mamba_d_head"].is_null())
        c.mamba_head_dim = j["mamba_d_head"].get<int>();
    if (j.contains("mamba_proj_bias") && !j["mamba_proj_bias"].is_null())
        c.mamba_proj_bias = j["mamba_proj_bias"].get<bool>();
    if (j.contains("mamba_d_state") && !j["mamba_d_state"].is_null())
        c.mamba_state_dim = j["mamba_d_state"].get<int>();
    if (j.contains("mamba_d_conv") && !j["mamba_d_conv"].is_null())
        c.mamba_conv_kernel = j["mamba_d_conv"].get<int>();
    if (j.contains("mamba_n_groups") && !j["mamba_n_groups"].is_null())
        c.mamba_groups = j["mamba_n_groups"].get<int>();
    if (j.contains("mamba_conv_bias") && !j["mamba_conv_bias"].is_null())
        c.mamba_conv_bias = j["mamba_conv_bias"].get<bool>();

    // Time step limits
    if (j.contains("time_step_limit") && !j["time_step_limit"].is_null()) {
        auto tsl = j["time_step_limit"].get<std::vector<float>>();
        if (!tsl.empty()) c.time_step_min = tsl[0];
        if (tsl.size() > 1) c.time_step_max = tsl[1];
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- GraniteMoeHybridRMSNormGated ---

GraniteMoeHybridRMSNormGated::GraniteMoeHybridRMSNormGated(int dims, float eps)
    : weight_(mx::ones({dims})), eps_(eps)
{}

mx::array GraniteMoeHybridRMSNormGated::operator()(
    const mx::array& x, const std::optional<mx::array>& gate) {
    auto states = x;
    if (gate.has_value()) {
        auto g = gate.value();
        states = swiglu(g, states); // silu(gate) * x
    }
    return mx::fast::rms_norm(states, weight_, eps_);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridRMSNormGated::weight_map() {
    return {{"weight", &weight_}};
}

// --- GraniteMoeHybridMamba2Mixer ---

static int compute_granite_intermediate(const GraniteMoeHybridConfiguration& config) {
    return config.mamba_heads.value() * config.mamba_head_dim.value();
}

static int compute_granite_conv_dim(const GraniteMoeHybridConfiguration& config) {
    int intermediate = compute_granite_intermediate(config);
    return intermediate + 2 * config.mamba_groups.value() * config.mamba_state_dim.value();
}

GraniteMoeHybridMamba2Mixer::GraniteMoeHybridMamba2Mixer(const GraniteMoeHybridConfiguration& config)
    : num_heads_(config.mamba_heads.value()),
      hidden_size_(config.hidden_size),
      ssm_state_size_(config.mamba_state_dim.value()),
      conv_kernel_size_(config.mamba_conv_kernel.value()),
      intermediate_size_(compute_granite_intermediate(config)),
      num_groups_(config.mamba_groups.value()),
      head_dim_(config.mamba_head_dim.value()),
      conv_dim_(compute_granite_conv_dim(config)),
      time_step_min_(config.time_step_min),
      time_step_max_(config.time_step_max),
      in_proj_weight_(mx::zeros({compute_granite_intermediate(config) + compute_granite_conv_dim(config) + config.mamba_heads.value(), config.hidden_size})),
      conv1d_weight_(mx::zeros({compute_granite_conv_dim(config), config.mamba_conv_kernel.value(), 1})),
      dt_bias_(mx::ones({config.mamba_heads.value()})),
      A_log_(mx::zeros({config.mamba_heads.value()})),
      D_(mx::ones({config.mamba_heads.value()})),
      out_proj_weight_(mx::zeros({config.hidden_size, compute_granite_intermediate(config)})),
      norm_(compute_granite_intermediate(config), config.rms_norm_eps)
{
    bool use_proj_bias = config.mamba_proj_bias.value_or(false);
    if (use_proj_bias) {
        in_proj_bias_ = mx::zeros({intermediate_size_ + conv_dim_ + num_heads_});
        out_proj_bias_ = mx::zeros({hidden_size_});
    }
    bool use_conv_bias = config.mamba_conv_bias.value_or(false);
    if (use_conv_bias) {
        conv1d_bias_ = mx::zeros({conv_dim_});
    }
}

mx::array GraniteMoeHybridMamba2Mixer::apply_conv(const mx::array& input, MambaCache* mc) {
    int batch = input.shape(0);
    auto dtype = input.dtype();

    int state_len = (conv_kernel_size_ > 1) ? (conv_kernel_size_ - 1) : 0;
    mx::array conv_state = mx::zeros({batch, state_len, conv_dim_}, dtype);
    if (mc && (*mc)[0].has_value()) {
        conv_state = (*mc)[0].value();
    }

    auto padded = mx::concatenate({conv_state, input}, 1);

    if (mc) {
        int end = padded.shape(1);
        int start = std::max(0, end - (conv_kernel_size_ - 1));
        (*mc)[0] = mx::slice(padded, {0, start, 0}, {padded.shape(0), end, padded.shape(2)});
    }

    // Depthwise conv1d: groups = conv_dim_
    auto conv_out = mx::conv1d(padded, conv1d_weight_, 1, 0, 1, conv_dim_);
    if (conv1d_bias_.has_value()) {
        conv_out = mx::add(conv_out, conv1d_bias_.value());
    }

    // silu activation
    return silu(conv_out);
}

mx::array GraniteMoeHybridMamba2Mixer::operator()(
    const mx::array& x,
    const std::optional<mx::array>& mask,
    KVCache* cache) {

    auto projected = linear_fwd(x, in_proj_weight_,
        in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);

    // Split: gate | conv_input | dt
    auto gate = mx::slice(projected, {0, 0, 0},
        {projected.shape(0), projected.shape(1), intermediate_size_});
    auto conv_input = mx::slice(projected, {0, 0, intermediate_size_},
        {projected.shape(0), projected.shape(1), intermediate_size_ + conv_dim_});
    auto dt = mx::slice(projected, {0, 0, intermediate_size_ + conv_dim_},
        {projected.shape(0), projected.shape(1), projected.shape(2)});

    // Apply mask to conv_input if present
    if (mask.has_value()) {
        auto expanded_mask = mx::expand_dims(mask.value(), -1);
        conv_input = mx::where(expanded_mask, conv_input, mx::zeros_like(conv_input));
    }

    auto* mc = cache ? cache->as_mamba() : nullptr;
    auto conv_output = apply_conv(conv_input, mc);

    // Split conv output: hidden | B | C
    int b_end = intermediate_size_ + num_groups_ * ssm_state_size_;
    auto hidden = mx::slice(conv_output, {0, 0, 0},
        {conv_output.shape(0), conv_output.shape(1), intermediate_size_});
    auto B_ssm = mx::slice(conv_output, {0, 0, intermediate_size_},
        {conv_output.shape(0), conv_output.shape(1), b_end});
    auto C_ssm = mx::slice(conv_output, {0, 0, b_end},
        {conv_output.shape(0), conv_output.shape(1), conv_output.shape(2)});

    int b = hidden.shape(0), l = hidden.shape(1);

    // Reshape for SSM
    hidden = mx::reshape(hidden, {b, l, num_heads_, head_dim_});
    B_ssm = mx::reshape(B_ssm, {b, l, num_groups_, ssm_state_size_});
    C_ssm = mx::reshape(C_ssm, {b, l, num_groups_, ssm_state_size_});
    auto dt_reshaped = mx::reshape(dt, {b, l, num_heads_});

    std::optional<mx::array> prev_state;
    if (mc && (*mc)[1].has_value()) {
        prev_state = (*mc)[1].value();
    }

    auto [y, new_state] = ssm_update(
        hidden, A_log_, B_ssm, C_ssm, D_, dt_reshaped, dt_bias_,
        prev_state, time_step_min_, time_step_max_, mask);

    if (mc) {
        (*mc)[1] = new_state;
    }

    // Flatten heads: [B, L, H, Dh] -> [B, L, H*Dh]
    auto flattened_y = mx::reshape(y, {b, l, intermediate_size_});

    // Apply gated norm then output projection
    auto normed = norm_(flattened_y, gate);
    return linear_fwd(normed, out_proj_weight_,
        out_proj_bias_.has_value() ? &out_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridMamba2Mixer::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"in_proj.weight", &in_proj_weight_},
        {"conv1d.weight", &conv1d_weight_},
        {"dt_bias", &dt_bias_},
        {"A_log", &A_log_},
        {"D", &D_},
        {"out_proj.weight", &out_proj_weight_},
    };
    if (in_proj_bias_.has_value()) map["in_proj.bias"] = &in_proj_bias_.value();
    if (conv1d_bias_.has_value()) map["conv1d.bias"] = &conv1d_bias_.value();
    if (out_proj_bias_.has_value()) map["out_proj.bias"] = &out_proj_bias_.value();
    for (auto& [k, v] : norm_.weight_map()) map["norm." + k] = v;
    return map;
}

// --- GraniteMoeHybridAttention ---

GraniteMoeHybridAttention::GraniteMoeHybridAttention(const GraniteMoeHybridConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.hidden_size / config.num_attention_heads),
      scale_(config.attention_multiplier),
      use_rope_(config.position_embedding_type != "nope"),
      wq_weight_(mx::zeros({config.num_attention_heads * (config.hidden_size / config.num_attention_heads), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * (config.hidden_size / config.num_attention_heads), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * (config.hidden_size / config.num_attention_heads), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * (config.hidden_size / config.num_attention_heads)})),
      rope_theta_(config.rope_theta)
{
    if (config.attention_bias) {
        int hd = config.hidden_size / config.num_attention_heads;
        wq_bias_ = mx::zeros({config.num_attention_heads * hd});
        wk_bias_ = mx::zeros({config.num_key_value_heads * hd});
        wv_bias_ = mx::zeros({config.num_key_value_heads * hd});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array GraniteMoeHybridAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache) {

    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_.has_value() ? &wq_bias_.value() : nullptr);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_.has_value() ? &wk_bias_.value() : nullptr);
    auto values = linear_fwd(x, wv_weight_, wv_bias_.has_value() ? &wv_bias_.value() : nullptr);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    if (use_rope_) {
        int offset = cache ? cache->offset() : 0;
        queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
        keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);
    }

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, wo_bias_.has_value() ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_.has_value()) {
        map["q_proj.bias"] = &wq_bias_.value();
        map["k_proj.bias"] = &wk_bias_.value();
    }
    if (wv_bias_.has_value()) {
        map["v_proj.bias"] = &wv_bias_.value();
        map["o_proj.bias"] = &wo_bias_.value();
    }
    return map;
}

// --- GraniteMoeHybridTopKGating ---

GraniteMoeHybridTopKGating::GraniteMoeHybridTopKGating(int input_size, int num_experts, int top_k)
    : num_experts_(num_experts),
      top_k_(top_k),
      layer_weight_(mx::zeros({num_experts, input_size}))
{}

std::pair<mx::array, mx::array> GraniteMoeHybridTopKGating::operator()(const mx::array& x) {
    auto logits = linear_fwd(x, layer_weight_);
    auto neg_logits = mx::negative(logits);
    auto indices = mx::argpartition(neg_logits, top_k_ - 1, -1);
    indices = mx::slice(indices, {0, 0, 0}, {indices.shape(0), indices.shape(1), top_k_});
    auto top_k_logits = mx::take_along_axis(logits, indices, -1);
    auto gates = mx::softmax(top_k_logits, -1);
    return {indices, gates};
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridTopKGating::weight_map() {
    return {{"layer.weight", &layer_weight_}};
}

// --- GraniteMoeHybridMoE ---

GraniteMoeHybridMoE::GraniteMoeHybridMoE(const GraniteMoeHybridConfiguration& config)
    : router_(config.hidden_size, config.num_local_experts.value(), config.num_experts_per_token.value()),
      switch_mlp_(config.hidden_size, config.intermediate_size, config.num_local_experts.value())
{}

mx::array GraniteMoeHybridMoE::operator()(const mx::array& x) {
    auto [indices, gates] = router_(x);
    auto expert_outputs = switch_mlp_(x, indices);
    // expert_outputs: [B, L, k, hidden], gates: [B, L, k]
    auto gates_expanded = mx::expand_dims(gates, -1);
    return mx::sum(mx::multiply(expert_outputs, gates_expanded), -2);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridMoE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : router_.weight_map()) map["router." + k] = v;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    return map;
}

// --- GraniteMoeHybridSharedMLP ---

GraniteMoeHybridSharedMLP::GraniteMoeHybridSharedMLP(const GraniteMoeHybridConfiguration& config)
    : input_linear_weight_(mx::zeros({config.shared_intermediate_size.value() * 2, config.hidden_size})),
      output_linear_weight_(mx::zeros({config.hidden_size, config.shared_intermediate_size.value()})),
      shared_intermediate_size_(config.shared_intermediate_size.value())
{}

mx::array GraniteMoeHybridSharedMLP::operator()(const mx::array& x) {
    auto projected = linear_fwd(x, input_linear_weight_);
    // Split in half along last dimension
    auto first = mx::slice(projected, {0, 0, 0},
        {projected.shape(0), projected.shape(1), shared_intermediate_size_});
    auto second = mx::slice(projected, {0, 0, shared_intermediate_size_},
        {projected.shape(0), projected.shape(1), projected.shape(2)});
    // silu(first) * second
    auto activated = swiglu(first, second);
    return linear_fwd(activated, output_linear_weight_);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridSharedMLP::weight_map() {
    return {
        {"input_linear.weight", &input_linear_weight_},
        {"output_linear.weight", &output_linear_weight_},
    };
}

// --- GraniteMoeHybridMLP ---

GraniteMoeHybridMLP::GraniteMoeHybridMLP(const GraniteMoeHybridConfiguration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size}))
{
    if (config.mlp_bias) {
        gate_bias_ = mx::zeros({config.intermediate_size});
        up_bias_ = mx::zeros({config.intermediate_size});
        down_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array GraniteMoeHybridMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_, gate_bias_.has_value() ? &gate_bias_.value() : nullptr);
    auto u = linear_fwd(x, up_weight_, up_bias_.has_value() ? &up_bias_.value() : nullptr);
    // silu(gate) * up
    auto activated = swiglu(g, u);
    return linear_fwd(activated, down_weight_, down_bias_.has_value() ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
    if (gate_bias_.has_value()) map["gate_proj.bias"] = &gate_bias_.value();
    if (up_bias_.has_value()) map["up_proj.bias"] = &up_bias_.value();
    if (down_bias_.has_value()) map["down_proj.bias"] = &down_bias_.value();
    return map;
}

// --- GraniteMoeHybridLayer ---

GraniteMoeHybridLayer::GraniteMoeHybridLayer(
    const GraniteMoeHybridConfiguration& config,
    const std::string& layer_type)
    : layer_type_(layer_type),
      residual_multiplier_(config.residual_multiplier),
      use_moe_(config.use_moe()),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps)
{
    if (layer_type == "mamba") {
        mamba_.emplace(config);
    } else {
        self_attn_.emplace(config);
    }

    if (use_moe_) {
        block_sparse_moe_.emplace(config);
        shared_mlp_.emplace(config);
    } else {
        mlp_.emplace(config);
    }
}

mx::array GraniteMoeHybridLayer::operator()(
    const mx::array& x,
    const AttentionMask& attn_mask,
    const std::optional<mx::array>& ssm_mask,
    KVCache* cache) {

    auto residual = x;
    auto hidden = mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_);

    if (layer_type_ == "mamba") {
        hidden = (*mamba_)(hidden, ssm_mask, cache);
    } else {
        hidden = (*self_attn_)(hidden, attn_mask, cache);
    }

    hidden = mx::add(residual, mx::multiply(hidden, mx::array(residual_multiplier_)));

    residual = hidden;
    auto normed = mx::fast::rms_norm(hidden, post_attention_layernorm_weight_, rms_norm_eps_);

    mx::array mlp_output = use_moe_
        ? mx::add((*block_sparse_moe_)(normed), (*shared_mlp_)(normed))
        : (*mlp_)(normed);

    return mx::add(residual, mx::multiply(mlp_output, mx::array(residual_multiplier_)));
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;

    if (self_attn_.has_value()) {
        for (auto& [k, v] : self_attn_->weight_map()) map["self_attn." + k] = v;
    }
    if (mamba_.has_value()) {
        for (auto& [k, v] : mamba_->weight_map()) map["mamba." + k] = v;
    }
    if (block_sparse_moe_.has_value()) {
        for (auto& [k, v] : block_sparse_moe_->weight_map()) map["block_sparse_moe." + k] = v;
    }
    if (shared_mlp_.has_value()) {
        for (auto& [k, v] : shared_mlp_->weight_map()) map["shared_mlp." + k] = v;
    }
    if (mlp_.has_value()) {
        for (auto& [k, v] : mlp_->weight_map()) map["mlp." + k] = v;
    }
    return map;
}

// --- GraniteMoeHybridModelInner ---

GraniteMoeHybridModelInner::GraniteMoeHybridModelInner(const GraniteMoeHybridConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      rms_norm_eps_(config.rms_norm_eps),
      embedding_multiplier_(config.embedding_multiplier)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, config.layer_types[i]);
    }

    // Find first attention and mamba indices
    for (size_t i = 0; i < config.layer_types.size(); ++i) {
        if (!first_attention_index_.has_value() && config.layer_types[i] == "attention") {
            first_attention_index_ = static_cast<int>(i);
        }
        if (!first_mamba_index_.has_value() && config.layer_types[i] == "mamba") {
            first_mamba_index_ = static_cast<int>(i);
        }
    }
}

mx::array GraniteMoeHybridModelInner::operator()(
    const mx::array& inputs, std::vector<KVCache>* cache) {

    auto hidden = mx::multiply(
        mx::take(embed_tokens_weight_, inputs, 0),
        mx::array(embedding_multiplier_));

    // Create attention mask using the first attention layer's cache
    AttentionMask attn_mask;
    if (first_attention_index_.has_value()) {
        int idx = first_attention_index_.value();
        KVCache* attn_cache = (cache && idx < static_cast<int>(cache->size()))
            ? &(*cache)[idx] : nullptr;
        attn_mask = create_attention_mask(hidden, attn_cache);
    }

    // SSM mask is always nil (matching Swift's createSSMMask returning nil)
    std::optional<mx::array> ssm_mask;

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        hidden = layers_[i](hidden, attn_mask, ssm_mask, lc);
    }

    return mx::fast::rms_norm(hidden, norm_weight_, rms_norm_eps_);
}

mx::array GraniteMoeHybridModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- GraniteMoeHybridModel ---

GraniteMoeHybridModel::GraniteMoeHybridModel(const GraniteMoeHybridConfiguration& config)
    : config_(config),
      model_(config_),
      logits_scaling_(config.logits_scaling)
{
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult GraniteMoeHybridModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput GraniteMoeHybridModel::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array GraniteMoeHybridModel::forward_impl(
    const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        out = linear_forward(out, lm_head_weight_.value());
    } else {
        out = model_.embed_as_linear(out);
    }
    return mx::divide(out, mx::array(logits_scaling_));
}

std::vector<KVCache> GraniteMoeHybridModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        const auto& lt = config_.layer_types[i];
        if (lt == "mamba") {
            caches.emplace_back(MambaCache());
        } else {
            // attention layer
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
GraniteMoeHybridModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // 1. Remove lm_head.weight if tie_word_embeddings
    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    // 2. Conv1d weight transpose: if last dim != 1, swap axes 1 and 2
    for (auto& [key, value] : weights) {
        if (key.find("conv1d.weight") != std::string::npos && value.shape(-1) != 1) {
            value = mx::swapaxes(value, 1, 2);
        }
    }

    // 3. MoE sanitize: split input_linear into gate_proj + up_proj for SwitchGLU
    if (config_.use_moe()) {
        std::string probe_key = "model.layers.0.block_sparse_moe.input_linear.weight";
        if (weights.find(probe_key) != weights.end()) {
            for (int l = 0; l < config_.num_hidden_layers; ++l) {
                std::string prefix = "model.layers." + std::to_string(l) + ".block_sparse_moe";

                auto il_it = weights.find(prefix + ".input_linear.weight");
                if (il_it == weights.end()) continue;

                auto input_weight = std::move(il_it->second);
                weights.erase(il_it);

                // input_weight: [num_experts, 2*hidden, input]
                int expert_hidden = input_weight.shape(1);
                int half_hidden = expert_hidden / 2;

                // gate_proj = input_weight[:, :half_hidden, :]
                auto gate_proj = mx::slice(input_weight,
                    {0, 0, 0},
                    {input_weight.shape(0), half_hidden, input_weight.shape(2)});
                // up_proj = input_weight[:, half_hidden:, :]
                auto up_proj = mx::slice(input_weight,
                    {0, half_hidden, 0},
                    {input_weight.shape(0), expert_hidden, input_weight.shape(2)});

                weights.insert_or_assign(prefix + ".switch_mlp.gate_proj.weight", gate_proj);
                weights.insert_or_assign(prefix + ".switch_mlp.up_proj.weight", up_proj);

                // Rename output_linear -> switch_mlp.down_proj
                auto ol_it = weights.find(prefix + ".output_linear.weight");
                if (ol_it != weights.end()) {
                    weights.insert_or_assign(prefix + ".switch_mlp.down_proj.weight",
                        std::move(ol_it->second));
                    weights.erase(ol_it);
                }
            }
        }
    }

    // 4. Non-MoE sanitize: split shared_mlp.input_linear into mlp.gate_proj + mlp.up_proj
    if (!config_.use_moe()) {
        std::string probe_key = "model.layers.0.shared_mlp.input_linear.weight";
        if (weights.find(probe_key) != weights.end()) {
            for (int l = 0; l < config_.num_hidden_layers; ++l) {
                std::string shared_prefix = "model.layers." + std::to_string(l) + ".shared_mlp";
                std::string mlp_prefix = "model.layers." + std::to_string(l) + ".mlp";

                auto il_it = weights.find(shared_prefix + ".input_linear.weight");
                if (il_it == weights.end()) continue;

                auto input_weight = std::move(il_it->second);
                weights.erase(il_it);

                // Split in half along axis 0 (2D weight: [2*intermediate, hidden])
                int half = input_weight.shape(0) / 2;
                auto gate_proj = mx::slice(input_weight,
                    {0, 0},
                    {half, input_weight.shape(1)});
                auto up_proj = mx::slice(input_weight,
                    {half, 0},
                    {input_weight.shape(0), input_weight.shape(1)});

                weights.insert_or_assign(mlp_prefix + ".gate_proj.weight", gate_proj);
                weights.insert_or_assign(mlp_prefix + ".up_proj.weight", up_proj);

                // Rename output_linear -> mlp.down_proj
                auto ol_it = weights.find(shared_prefix + ".output_linear.weight");
                if (ol_it != weights.end()) {
                    weights.insert_or_assign(mlp_prefix + ".down_proj.weight",
                        std::move(ol_it->second));
                    weights.erase(ol_it);
                }
            }
        }
    }

    return weights;
}

void GraniteMoeHybridModel::load_weights(
    const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> GraniteMoeHybridModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
