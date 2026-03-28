// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of Qwen35.swift + Qwen35MoE.swift -- Hybrid GatedDeltaNet + Attention + MoE
//
// Qwen3.5 differs from Qwen3Next in several ways:
// - GDN uses 4 separate projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a)
//   instead of 2 combined ones (in_proj_qkvz, in_proj_ba)
// - All layers use MoE (when num_experts > 0), not alternating with dense MLP
// - MoE sanitize splits fused gate_up_proj into gate_proj + up_proj

#include <mlx-lm/llm/models/qwen35_moe.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/quantized_linear.h>
#include <algorithm>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, Qwen35MoEConfiguration& c) {
    // VLM models nest the text config under "text_config"
    const auto& cfg = j.contains("text_config") ? j.at("text_config") : j;
    // All fields use value() with defaults matching Swift's decodeIfPresent
    c.hidden_size = cfg.value("hidden_size", 4096);
    c.num_hidden_layers = cfg.value("num_hidden_layers", 32);
    c.intermediate_size = cfg.value("intermediate_size", 14336);
    c.num_attention_heads = cfg.value("num_attention_heads", 32);
    c.num_key_value_heads = cfg.value("num_key_value_heads", 8);
    c.linear_num_value_heads = cfg.value("linear_num_value_heads", 64);
    c.linear_num_key_heads = cfg.value("linear_num_key_heads", 16);
    c.linear_key_head_dim = cfg.value("linear_key_head_dim", 192);
    c.linear_value_head_dim = cfg.value("linear_value_head_dim", 128);
    c.linear_conv_kernel_dim = cfg.value("linear_conv_kernel_dim", 4);
    c.rms_norm_eps = cfg.value("rms_norm_eps", 1e-6f);
    c.vocab_size = cfg.value("vocab_size", 151936);
    c.rope_theta = cfg.value("rope_theta", 100000.0f);
    c.partial_rotary_factor = cfg.value("partial_rotary_factor", 0.25f);
    c.max_position_embeddings = cfg.value("max_position_embeddings", 131072);
    c.tie_word_embeddings = cfg.value("tie_word_embeddings", false);
    c.attention_bias = cfg.value("attention_bias", false);
    c.full_attention_interval = cfg.value("full_attention_interval", 4);
    c.norm_topk_prob = cfg.value("norm_topk_prob", true);

    // MoE fields (may be absent for pure text model)
    c.num_experts = cfg.value("num_experts", 0);
    c.num_experts_per_tok = cfg.value("num_experts_per_tok", 0);
    c.decoder_sparse_step = cfg.value("decoder_sparse_step", 1);
    c.shared_expert_intermediate_size = cfg.value("shared_expert_intermediate_size", 0);
    c.moe_intermediate_size = cfg.value("moe_intermediate_size", 0);

    if (cfg.contains("head_dim") && !cfg.at("head_dim").is_null()) {
        c.head_dim = cfg.at("head_dim").get<int>();
    }

    // Parse rope_scaling: check rope_parameters first (Qwen3.5 style),
    // then fall back to rope_scaling
    if (cfg.contains("rope_parameters") && !cfg.at("rope_parameters").is_null()) {
        auto& rp = cfg.at("rope_parameters");
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : rp.items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        // Normalize: if "type" missing but "rope_type" present, copy it
        if (scaling.find("type") == scaling.end()) {
            auto it = scaling.find("rope_type");
            if (it != scaling.end()) scaling["type"] = it->second;
        }
        // Extract rope_theta and partial_rotary_factor from rope_parameters
        auto theta_it = scaling.find("rope_theta");
        if (theta_it != scaling.end() && theta_it->second.is_float()) {
            c.rope_theta = theta_it->second.as_float();
        }
        auto prf_it = scaling.find("partial_rotary_factor");
        if (prf_it != scaling.end() && prf_it->second.is_float()) {
            c.partial_rotary_factor = prf_it->second.as_float();
        }
        c.rope_scaling = scaling;
    } else if (cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : cfg.at("rope_scaling").items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }

    // Resolve head_dim if not provided
    if (!c.head_dim.has_value()) {
        c.head_dim = c.hidden_size / c.num_attention_heads;
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const std::optional<mx::array>& bias = std::nullopt) {
    return linear_forward(x, w, bias.has_value() ? &bias.value() : nullptr);
}

// --- Qwen35MoEAttention ---
// Swift: Qwen35Attention -- standard attention with sigmoid gate on q_proj output

Qwen35MoEAttention::Qwen35MoEAttention(const Qwen35MoEConfiguration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(args.resolved_head_dim()), -0.5f)),
      // q_proj outputs 2x head_dim for the sigmoid gate
      q_proj_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim() * 2, args.hidden_size})),
      k_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      v_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      o_proj_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      q_norm_weight_(mx::ones({args.resolved_head_dim()})),
      k_norm_weight_(mx::ones({args.resolved_head_dim()})),
      rms_norm_eps_(args.rms_norm_eps),
      rope_theta_(args.rope_theta),
      rope_dims_(std::max(1, static_cast<int>(args.resolved_head_dim() * args.partial_rotary_factor)))
{
    if (args.attention_bias) {
        q_proj_bias_ = mx::zeros({args.num_attention_heads * args.resolved_head_dim() * 2});
        k_proj_bias_ = mx::zeros({args.num_key_value_heads * args.resolved_head_dim()});
        v_proj_bias_ = mx::zeros({args.num_key_value_heads * args.resolved_head_dim()});
        o_proj_bias_ = mx::zeros({args.hidden_size});
    }
}

mx::array Qwen35MoEAttention::operator()(const mx::array& x,
                                           const AttentionMask& mask,
                                           KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto q_proj_out = linear_fwd(x, q_proj_weight_, q_proj_bias_);
    // Reshape to [B, L, num_heads, 2*head_dim] then split into queries + gate
    q_proj_out = mx::reshape(q_proj_out, {B, L, num_heads_, -1});
    int hd = head_dim_;
    auto queries = mx::slice(q_proj_out, {0, 0, 0, 0}, {B, L, num_heads_, hd});
    auto gate = mx::slice(q_proj_out, {0, 0, 0, hd}, {B, L, num_heads_, 2 * hd});
    gate = mx::reshape(gate, {B, L, -1});

    auto keys = linear_fwd(x, k_proj_weight_, k_proj_bias_);
    auto values = linear_fwd(x, v_proj_weight_, v_proj_bias_);

    // RMSNorm on queries and keys with learned weights
    queries = mx::transpose(
        mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_),
        {0, 2, 1, 3});
    keys = mx::transpose(
        mx::fast::rms_norm(mx::reshape(keys, {B, L, num_kv_heads_, -1}), k_norm_weight_, rms_norm_eps_),
        {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // RoPE with partial rotary factor
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, rope_dims_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, rope_dims_, false, rope_theta_, 1.0f, offset);

    // KV cache update
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // SDPA
    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    // Swift: oProj(sigmoidMultiply(output, gate))
    static auto compiled_gate = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::multiply(inputs[0], mx::sigmoid(inputs[1]))};
        },
        /*shapeless=*/true);
    output = compiled_gate({output, gate})[0];
    return linear_fwd(output, o_proj_weight_, o_proj_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen35MoEAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &q_proj_weight_},
        {"k_proj.weight", &k_proj_weight_},
        {"v_proj.weight", &v_proj_weight_},
        {"o_proj.weight", &o_proj_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
    if (q_proj_bias_.has_value()) {
        map["q_proj.bias"] = &q_proj_bias_.value();
        map["k_proj.bias"] = &k_proj_bias_.value();
        map["v_proj.bias"] = &v_proj_bias_.value();
        map["o_proj.bias"] = &o_proj_bias_.value();
    }
    return map;
}

// --- Qwen35MoEGatedDeltaNet ---
// Swift: Qwen35GatedDeltaNet -- uses 4 separate projections unlike Qwen3Next's 2

Qwen35MoEGatedDeltaNet::Qwen35MoEGatedDeltaNet(const Qwen35MoEConfiguration& args)
    : hidden_size_(args.hidden_size),
      num_v_heads_(args.linear_num_value_heads),
      num_k_heads_(args.linear_num_key_heads),
      head_k_dim_(args.linear_key_head_dim),
      head_v_dim_(args.linear_value_head_dim),
      key_dim_(args.linear_key_head_dim * args.linear_num_key_heads),
      value_dim_(args.linear_value_head_dim * args.linear_num_value_heads),
      conv_kernel_size_(args.linear_conv_kernel_dim),
      conv_dim_(args.linear_key_head_dim * args.linear_num_key_heads * 2
                + args.linear_value_head_dim * args.linear_num_value_heads),
      conv1d_weight_(mx::zeros({conv_dim_, 1, args.linear_conv_kernel_dim})),
      in_proj_qkv_weight_(mx::zeros({key_dim_ * 2 + value_dim_, args.hidden_size})),
      in_proj_z_weight_(mx::zeros({value_dim_, args.hidden_size})),
      in_proj_b_weight_(mx::zeros({args.linear_num_value_heads, args.hidden_size})),
      in_proj_a_weight_(mx::zeros({args.linear_num_value_heads, args.hidden_size})),
      dt_bias_(mx::ones({args.linear_num_value_heads})),
      a_log_(mx::zeros({args.linear_num_value_heads})),
      norm_(args.linear_value_head_dim, args.rms_norm_eps),
      out_proj_weight_(mx::zeros({args.hidden_size, value_dim_}))
{}

mx::array Qwen35MoEGatedDeltaNet::operator()(
    const mx::array& inputs,
    const std::optional<mx::array>& mask,
    MambaCache* cache)
{
    int B = inputs.shape(0), S = inputs.shape(1);

    // 4 separate projections (unlike Qwen3Next which uses 2 combined)
    auto qkv = linear_fwd(inputs, in_proj_qkv_weight_);
    auto z = mx::reshape(linear_fwd(inputs, in_proj_z_weight_), {B, S, num_v_heads_, head_v_dim_});
    auto b_val = linear_fwd(inputs, in_proj_b_weight_);
    auto a_val = linear_fwd(inputs, in_proj_a_weight_);

    // Conv1d processing
    auto dtype = inputs.dtype();
    mx::array conv_state(0.0f);
    if (cache && (*cache)[0].has_value()) {
        conv_state = (*cache)[0].value();
    } else {
        conv_state = mx::zeros({B, conv_kernel_size_ - 1, conv_dim_}, dtype);
    }

    if (mask.has_value()) {
        qkv = mx::where(mx::expand_dims(*mask, -1), qkv, mx::zeros_like(qkv));
    }

    auto conv_input = mx::concatenate({conv_state, qkv}, 1);

    // Save conv state for next step
    if (cache) {
        int start = conv_input.shape(1) - (conv_kernel_size_ - 1);
        (*cache)[0] = mx::slice(conv_input, {0, start, 0}, {B, conv_input.shape(1), conv_dim_});
    }

    // T=1 decode fast path
    if (S == 1 && cache && (*cache)[0].has_value() && conv_input.shape(1) == conv_kernel_size_) {
        // Fused conv1d + silu
        auto w = mx::reshape(
            mx::transpose(mx::reshape(conv1d_weight_, {conv_dim_, conv_kernel_size_})),
            {1, conv_kernel_size_, conv_dim_});
        static auto compiled_conv_silu = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto dot = mx::sum(mx::multiply(inputs[0], inputs[1]), 1, true);
                return {mx::multiply(dot, mx::sigmoid(dot))};
            },
            /*shapeless=*/true);
        auto conv_out = compiled_conv_silu({conv_input, w})[0];

        // Split into q, k, v
        auto q_out = mx::reshape(mx::slice(conv_out, {0, 0, 0}, {B, 1, key_dim_}),
                                  {B, 1, num_k_heads_, head_k_dim_});
        auto k_out = mx::reshape(mx::slice(conv_out, {0, 0, key_dim_}, {B, 1, 2 * key_dim_}),
                                  {B, 1, num_k_heads_, head_k_dim_});
        auto v_out = mx::reshape(mx::slice(conv_out, {0, 0, 2 * key_dim_}, {B, 1, conv_dim_}),
                                  {B, 1, num_v_heads_, head_v_dim_});

        // Q/K norms
        float inv_scale = std::pow(static_cast<float>(head_k_dim_), -0.5f);
        auto q_norm_w = mx::full({head_k_dim_}, inv_scale * inv_scale, dtype);
        auto k_norm_w = mx::full({head_k_dim_}, inv_scale, dtype);
        q_out = mx::fast::rms_norm(q_out, q_norm_w, 1e-6f);
        k_out = mx::fast::rms_norm(k_out, k_norm_w, 1e-6f);

        // Fused GDN decode step
        auto ssm_state = (*cache)[1].has_value() ? (*cache)[1].value()
                           : mx::zeros({B, num_v_heads_, head_v_dim_, head_k_dim_}, dtype);

        int rep = num_v_heads_ / num_k_heads_;
        static auto compiled_decode_step = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto q = inputs[0]; auto k = inputs[1]; auto v = inputs[2];
                auto b = inputs[3]; auto a_log = inputs[4]; auto a = inputs[5];
                auto dt_bias = inputs[6]; auto state = inputs[7];

                int B_ = q.shape(0), Hk_ = q.shape(2), Dk_ = q.shape(3);
                int Hv_ = v.shape(2), Dv_ = v.shape(3);
                int rep_ = Hv_ / Hk_;

                // beta + g fused
                auto beta = mx::sigmoid(b);
                auto a_log_f32 = mx::astype(a_log, mx::float32);
                auto sp = mx::log(mx::add(mx::exp(mx::add(a, dt_bias)), mx::array(1.0f)));
                auto g = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), sp)));
                g = mx::astype(g, a.dtype());

                // Squeeze T=1
                auto qt = mx::reshape(q, {B_, Hk_, Dk_});
                auto kt = mx::reshape(k, {B_, Hk_, Dk_});
                auto vt = mx::reshape(v, {B_, Hv_, Dv_});
                auto gt = mx::reshape(g, {B_, Hv_});
                auto bt = mx::reshape(beta, {B_, Hv_});

                // Repeat q/k heads
                if (rep_ > 1) {
                    qt = mx::reshape(mx::broadcast_to(
                        mx::reshape(qt, {B_, Hk_, 1, Dk_}), {B_, Hk_, rep_, Dk_}), {B_, Hv_, Dk_});
                    kt = mx::reshape(mx::broadcast_to(
                        mx::reshape(kt, {B_, Hk_, 1, Dk_}), {B_, Hk_, rep_, Dk_}), {B_, Hv_, Dk_});
                }

                // GDN recurrence
                auto decay = mx::expand_dims(mx::expand_dims(gt, -1), -1);
                auto s = mx::multiply(state, decay);
                auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(kt, -2)), -1);
                auto delta = mx::multiply(mx::subtract(vt, kv_mem), mx::expand_dims(bt, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(kt, -2), mx::expand_dims(delta, -1)));
                auto y = mx::sum(mx::multiply(s, mx::expand_dims(qt, -2)), -1);

                return {mx::expand_dims(y, 1), s};
            },
            /*shapeless=*/true);

        auto results = compiled_decode_step(
            {q_out, k_out, v_out, b_val, a_log_, a_val, dt_bias_, ssm_state});
        auto out = results[0];
        (*cache)[1] = results[1];

        // Gated norm + output projection
        auto normalized = norm_(out, z);
        return linear_fwd(mx::reshape(normalized, {B, S, -1}), out_proj_weight_);
    }

    // General path: T>1 prefill
    auto conv_out = mx::conv1d(conv_input, conv1d_weight_, 1, 0, 1, conv_dim_);
    conv_out = silu(conv_out);

    auto q_out = mx::reshape(mx::slice(conv_out, {0, 0, 0}, {B, S, key_dim_}),
                              {B, S, num_k_heads_, head_k_dim_});
    auto k_out = mx::reshape(mx::slice(conv_out, {0, 0, key_dim_}, {B, S, 2 * key_dim_}),
                              {B, S, num_k_heads_, head_k_dim_});
    auto v_out = mx::reshape(mx::slice(conv_out, {0, 0, 2 * key_dim_}, {B, S, conv_dim_}),
                              {B, S, num_v_heads_, head_v_dim_});

    float inv_scale = std::pow(static_cast<float>(head_k_dim_), -0.5f);
    auto q_norm_w = mx::full({head_k_dim_}, inv_scale * inv_scale, q_out.dtype());
    auto k_norm_w = mx::full({head_k_dim_}, inv_scale, k_out.dtype());
    q_out = mx::fast::rms_norm(q_out, q_norm_w, 1e-6f);
    k_out = mx::fast::rms_norm(k_out, k_norm_w, 1e-6f);

    std::optional<mx::array> ssm_state;
    if (cache && (*cache)[1].has_value()) {
        ssm_state = (*cache)[1].value();
    }

    auto [out, new_state] = gated_delta_update(
        q_out, k_out, v_out, a_val, b_val, a_log_, dt_bias_, ssm_state, mask);

    if (cache) {
        (*cache)[1] = new_state;
    }

    auto normalized = norm_(out, z);
    return linear_fwd(mx::reshape(normalized, {B, S, -1}), out_proj_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen35MoEGatedDeltaNet::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["conv1d.weight"] = &conv1d_weight_;
    map["in_proj_qkv.weight"] = &in_proj_qkv_weight_;
    map["in_proj_z.weight"] = &in_proj_z_weight_;
    map["in_proj_b.weight"] = &in_proj_b_weight_;
    map["in_proj_a.weight"] = &in_proj_a_weight_;
    map["dt_bias"] = &dt_bias_;
    map["A_log"] = &a_log_;
    for (auto& [k, v] : norm_.weight_map()) map["norm." + k] = v;
    map["out_proj.weight"] = &out_proj_weight_;
    return map;
}

// --- Qwen35MoEMLP ---
// Swift: Qwen3NextMLP reused -- dense MLP with SwiGLU

Qwen35MoEMLP::Qwen35MoEMLP(int dimensions, int hidden_dimensions)
    : gate_proj_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_proj_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_proj_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen35MoEMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_proj_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_proj_weight_)), down_proj_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen35MoEMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_proj_weight_},
        {"down_proj.weight", &down_proj_weight_},
        {"up_proj.weight", &up_proj_weight_},
    };
}

// --- Qwen35MoESparseMoeBlock ---
// Swift: Qwen35SparseMoeBlock -- sparse MoE with shared expert + shared expert gate

Qwen35MoESparseMoeBlock::Qwen35MoESparseMoeBlock(const Qwen35MoEConfiguration& args)
    : norm_topk_prob_(args.norm_topk_prob),
      num_experts_(args.num_experts),
      top_k_(args.num_experts_per_tok),
      gate_weight_(mx::zeros({args.num_experts, args.hidden_size})),
      switch_mlp_(args.hidden_size, args.moe_intermediate_size, args.num_experts),
      shared_expert_(args.hidden_size, args.shared_expert_intermediate_size),
      shared_expert_gate_weight_(mx::zeros({1, args.hidden_size}))
{}

mx::array Qwen35MoESparseMoeBlock::operator()(const mx::array& x) {
    auto gates = mx::softmax(linear_fwd(x, gate_weight_), -1);

    int k = top_k_;
    int kth = gates.shape(-1) - k;
    auto inds = mx::argpartition(gates, kth, -1);
    inds = mx::slice(inds, {0, 0, kth}, {inds.shape(0), inds.shape(1), inds.shape(2)});
    auto scores = mx::take_along_axis(gates, inds, -1);

    if (norm_topk_prob_) {
        static auto compiled_normalize_scores = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return {mx::divide(inputs[0], mx::sum(inputs[0], -1, true))};
            },
            /*shapeless=*/true);
        scores = compiled_normalize_scores({scores})[0];
    }

    auto y = switch_mlp_(x, inds);
    static auto compiled_expert_combine = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::sum(mx::multiply(inputs[0], mx::expand_dims(inputs[1], -1)), -2)};
        },
        /*shapeless=*/true);
    auto combined = compiled_expert_combine({y, scores})[0];

    // Shared expert: sigmoid(gate) * shared_y + combined
    auto shared_y = shared_expert_(x);
    static auto compiled_shared_gate = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto gated = mx::multiply(mx::sigmoid(inputs[0]), inputs[1]);
            return {mx::add(inputs[2], gated)};
        },
        /*shapeless=*/true);
    auto gate_out = linear_fwd(x, shared_expert_gate_weight_);
    return compiled_shared_gate({gate_out, shared_y, combined})[0];
}

std::unordered_map<std::string, mx::array*> Qwen35MoESparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    for (auto& [k, v] : shared_expert_.weight_map()) map["shared_expert." + k] = v;
    map["shared_expert_gate.weight"] = &shared_expert_gate_weight_;
    return map;
}

// --- Qwen35MoEDecoderLayer ---
// Swift: Qwen35DecoderLayer -- linear or standard attn, with MoE or dense MLP

Qwen35MoEDecoderLayer::Qwen35MoEDecoderLayer(const Qwen35MoEConfiguration& args, int layer_idx)
    : is_linear_((layer_idx + 1) % args.full_attention_interval != 0),
      use_moe_(false),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    if (is_linear_) {
        linear_attn_.emplace(args);
    } else {
        self_attn_.emplace(args);
    }

    // Swift: if numExperts > 0, all layers use MoE; otherwise dense MLP
    if (args.num_experts > 0) {
        use_moe_ = true;
        moe_mlp_.emplace(args);
    } else {
        dense_mlp_.emplace(args.hidden_size, args.intermediate_size);
    }
}

mx::array Qwen35MoEDecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& attention_mask,
    const std::optional<mx::array>& ssm_mask,
    KVCache* cache)
{
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_);

    mx::array h(0.0f);
    if (is_linear_) {
        auto* mamba = cache ? cache->as_mamba() : nullptr;
        h = (*linear_attn_)(normed, ssm_mask, mamba);
    } else {
        h = (*self_attn_)(normed, attention_mask, cache);
    }

    auto r = mx::add(x, h);
    auto post_normed = mx::fast::rms_norm(r, post_attention_layernorm_weight_, rms_norm_eps_);

    mx::array mlp_out(0.0f);
    if (use_moe_) {
        mlp_out = (*moe_mlp_)(post_normed);
    } else {
        mlp_out = (*dense_mlp_)(post_normed);
    }

    return mx::add(r, mlp_out);
}

std::unordered_map<std::string, mx::array*> Qwen35MoEDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (is_linear_) {
        for (auto& [k, v] : linear_attn_->weight_map()) map["linear_attn." + k] = v;
    } else {
        for (auto& [k, v] : self_attn_->weight_map()) map["self_attn." + k] = v;
    }
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) map["mlp." + k] = v;
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Qwen35MoEModelInner ---

Qwen35MoEModelInner::Qwen35MoEModelInner(const Qwen35MoEConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      full_attention_interval_(args.full_attention_interval)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args, i);
}

mx::array Qwen35MoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Find the first full-attention index for attention mask
    int fa_idx = full_attention_interval_ - 1;
    if (fa_idx >= static_cast<int>(layers_.size())) fa_idx = 0;

    auto fa_mask = create_attention_mask(
        h, cache && fa_idx < static_cast<int>(cache->size()) ? &(*cache)[fa_idx] : nullptr);

    // SSM mask: always nullopt (no left padding support)
    std::optional<mx::array> ssm_mask;

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        auto attn_mask = layers_[i].is_linear() ? AttentionMask{} : fa_mask;
        h = layers_[i](h, attn_mask, ssm_mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen35MoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen35MoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Qwen35MoEModel ---

Qwen35MoEModel::Qwen35MoEModel(const Qwen35MoEConfiguration& args)
    : config_(args), model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Qwen35MoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Qwen35MoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Qwen35MoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return linear_fwd(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::vector<KVCache> Qwen35MoEModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (const auto& layer : model_.get_layers()) {
        if (layer.is_linear()) {
            caches.emplace_back(MambaCache{});
        } else {
            caches.emplace_back(KVCacheSimple{});
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
Qwen35MoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Remove mtp.* keys
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("mtp.") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    // Qwen35MoE sanitize: remap "model.language_model" -> "model" prefix,
    // and add "model." prefix if not already present.
    // This matches the Swift Qwen35MoEModel.sanitize.
    std::unordered_map<std::string, mx::array> remapped;
    for (auto& [key, value] : weights) {
        // Skip vision weights
        if (key.find("vision_tower") == 0 || key.find("model.visual") == 0) {
            continue;
        }

        std::string new_key = key;
        // Strip "language_model." prefix from VLM wrapper:
        //   "language_model.model.X" -> "model.X"
        //   "language_model.lm_head.X" -> "lm_head.X"
        if (new_key.find("language_model.") == 0) {
            new_key = new_key.substr(std::string("language_model.").size());
        }
        remapped.insert_or_assign(new_key, std::move(value));
    }
    weights = std::move(remapped);

    // Split fused gate_up_proj into separate gate_proj + up_proj
    // Swift: experts.gate_up_proj -> split at mid -> gate_proj + up_proj
    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
        std::string gate_up_key = prefix + "experts.gate_up_proj";
        auto gu_it = weights.find(gate_up_key);
        if (gu_it != weights.end()) {
            auto gate_up = std::move(gu_it->second);
            weights.erase(gu_it);
            // gate_up shape: [num_experts, 2*moe_intermediate, hidden] or similar
            // mid = dim(-2) / 2
            int mid = gate_up.shape(-2) / 2;
            // gate_proj = gate_up[..., :mid, :]
            // up_proj = gate_up[..., mid:, :]
            auto ndim = gate_up.ndim();
            mx::Shape start(ndim, 0);
            mx::Shape stop_gate(gate_up.shape().begin(), gate_up.shape().end());
            mx::Shape start_up(ndim, 0);
            mx::Shape stop_up(gate_up.shape().begin(), gate_up.shape().end());
            stop_gate[ndim - 2] = mid;
            start_up[ndim - 2] = mid;
            weights.insert_or_assign(prefix + "switch_mlp.gate_proj.weight", mx::slice(gate_up, start, stop_gate));
            weights.insert_or_assign(prefix + "switch_mlp.up_proj.weight", mx::slice(gate_up, start_up, stop_up));

            // Move down_proj too
            std::string down_key = prefix + "experts.down_proj";
            auto dp_it = weights.find(down_key);
            if (dp_it != weights.end()) {
                weights.insert_or_assign(prefix + "switch_mlp.down_proj.weight", std::move(dp_it->second));
                weights.erase(dp_it);
            }
        }
    }

    // Stack per-expert weights (and their scales/biases) into SwitchGLU format
    if (weights.find("model.layers.0.mlp.experts.0.up_proj.weight") != weights.end()) {
        for (int l = 0; l < config_.num_hidden_layers; ++l) {
            std::string prefix = "model.layers." + std::to_string(l) + ".mlp.";
            for (const auto& n : {"up_proj", "down_proj", "gate_proj"}) {
                std::string key0 = prefix + "experts.0." + n + ".weight";
                if (weights.find(key0) == weights.end()) continue;

                std::vector<mx::array> w_join, s_join, b_join;
                w_join.reserve(config_.num_experts);
                s_join.reserve(config_.num_experts);
                b_join.reserve(config_.num_experts);
                bool has_biases = false;

                for (int e = 0; e < config_.num_experts; ++e) {
                    std::string ep = prefix + "experts." + std::to_string(e) + "." + n;
                    auto wit = weights.find(ep + ".weight");
                    w_join.push_back(std::move(wit->second));
                    weights.erase(wit);
                    auto sit = weights.find(ep + ".scales");
                    if (sit != weights.end()) {
                        s_join.push_back(std::move(sit->second));
                        weights.erase(sit);
                    }
                    auto bit = weights.find(ep + ".biases");
                    if (bit != weights.end()) {
                        b_join.push_back(std::move(bit->second));
                        weights.erase(bit);
                        has_biases = true;
                    }
                }

                std::string dst = prefix + "switch_mlp." + n;
                auto stacked_w = mx::stack(w_join);
                w_join.clear();  // Release references to individual expert arrays
                mx::eval(stacked_w);
                weights.insert_or_assign(dst + ".weight", std::move(stacked_w));
                if (!s_join.empty()) {
                    auto stacked_s = mx::stack(s_join);
                    s_join.clear();
                    mx::eval(stacked_s);
                    weights.insert_or_assign(dst + ".scales", std::move(stacked_s));
                }
                if (has_biases && !b_join.empty()) {
                    auto stacked_b = mx::stack(b_join);
                    b_join.clear();
                    mx::eval(stacked_b);
                    weights.insert_or_assign(dst + ".biases", std::move(stacked_b));
                }
            }
        }
    }

    // Fix conv1d weight ordering
    for (auto& [key, value] : weights) {
        if (key.find("conv1d.weight") != std::string::npos && value.shape(-1) != 1) {
            value = mx::moveaxis(value, 2, 1);
        }
    }

    return weights;
}

void Qwen35MoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    int loaded = 0, missing = 0;
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
            loaded++;
        } else {
            if (missing < 10) fprintf(stderr, "  MISSING: %s\n", name.c_str());
            missing++;
        }
    }
    fprintf(stderr, "[Qwen35MoE] Loaded %d/%d weights (%d missing)\n",
            loaded, loaded + missing, missing);
}

std::unordered_map<std::string, mx::array*> Qwen35MoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
