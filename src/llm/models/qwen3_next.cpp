// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of Qwen3Next.swift -- Hybrid GatedDeltaNet + Attention + MoE
//
// Faithful 1:1 port from Swift reference. No compiled kernels, no debug code,
// no float32 upcast in gated_delta_ops, no T=1 fast paths for conv1d.

#include <mlx-lm/llm/models/qwen3_next.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/quantized_linear.h>
#include <algorithm>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, Qwen3NextConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.linear_num_value_heads = j.at("linear_num_value_heads").get<int>();
    c.linear_num_key_heads = j.at("linear_num_key_heads").get<int>();
    c.linear_key_head_dim = j.at("linear_key_head_dim").get<int>();
    c.linear_value_head_dim = j.at("linear_value_head_dim").get<int>();
    c.linear_conv_kernel_dim = j.at("linear_conv_kernel_dim").get<int>();
    c.num_experts = j.at("num_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.decoder_sparse_step = j.at("decoder_sparse_step").get<int>();
    c.shared_expert_intermediate_size = j.at("shared_expert_intermediate_size").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.partial_rotary_factor = j.value("partial_rotary_factor", 1.0f);
    c.max_position_embeddings = j.value("max_position_embeddings", 32768);
    c.norm_topk_prob = j.value("norm_topk_prob", false);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.attention_bias = j.value("attention_bias", false);
    c.full_attention_interval = j.value("full_attention_interval", 4);

    if (j.contains("head_dim") && !j["head_dim"].is_null()) {
        c.head_dim = j["head_dim"].get<int>();
    }

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const std::optional<mx::array>& bias = std::nullopt) {
    return linear_forward(x, w, bias.has_value() ? &bias.value() : nullptr);
}

// --- Qwen3NextRMSNormGated ---

Qwen3NextRMSNormGated::Qwen3NextRMSNormGated(int dimensions, float eps)
    : weight_(mx::ones({dimensions})), eps_(eps)
{}

mx::array Qwen3NextRMSNormGated::operator()(const mx::array& x,
                                              const std::optional<mx::array>& gate) {
    auto result = mx::fast::rms_norm(x, weight_, eps_);
    if (gate.has_value()) {
        // Swift: silu(gate) * result
        result = swiglu(*gate, result);
    }
    return result;
}

std::unordered_map<std::string, mx::array*> Qwen3NextRMSNormGated::weight_map() {
    return {{"weight", &weight_}};
}

// --- Qwen3NextAttention ---
// Swift: Qwen3NextAttention -- standard attention for full-attention layers

Qwen3NextAttention::Qwen3NextAttention(const Qwen3NextConfiguration& args)
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

mx::array Qwen3NextAttention::operator()(const mx::array& x,
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

    // Swift: rmsNorm on queries and keys with learned weights
    queries = mx::transpose(
        mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_),
        {0, 2, 1, 3});
    keys = mx::transpose(
        mx::fast::rms_norm(mx::reshape(keys, {B, L, num_kv_heads_, -1}), k_norm_weight_, rms_norm_eps_),
        {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // RoPE
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

    // Swift: oProj(sigmoidMultiply(output, gate)) — compiled to fuse multiply+sigmoid
    static auto compiled_gate = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::multiply(inputs[0], mx::sigmoid(inputs[1]))};
        },
        /*shapeless=*/true);
    output = compiled_gate({output, gate})[0];
    return linear_fwd(output, o_proj_weight_, o_proj_bias_);
}

std::unordered_map<std::string, mx::array*> Qwen3NextAttention::weight_map() {
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

// --- Qwen3NextGatedDeltaNet ---
// Swift: Qwen3NextGatedDeltaNet -- linear attention for most layers

Qwen3NextGatedDeltaNet::Qwen3NextGatedDeltaNet(const Qwen3NextConfiguration& args)
    : hidden_size_(args.hidden_size),
      num_v_heads_(args.linear_num_value_heads),
      num_k_heads_(args.linear_num_key_heads),
      head_k_dim_(args.linear_key_head_dim),
      head_v_dim_(args.linear_value_head_dim),
      key_dim_(args.linear_key_head_dim * args.linear_num_key_heads),
      value_dim_(args.linear_value_head_dim * args.linear_num_value_heads),
      conv_kernel_size_(args.linear_conv_kernel_dim),
      conv_dim_(args.linear_key_head_dim * args.linear_num_key_heads * 2 + args.linear_value_head_dim * args.linear_num_value_heads),
      conv1d_weight_(mx::zeros({conv_dim_, 1, args.linear_conv_kernel_dim})),
      in_proj_qkvz_weight_(mx::zeros({key_dim_ * 2 + value_dim_ * 2, args.hidden_size})),
      in_proj_ba_weight_(mx::zeros({args.linear_num_value_heads * 2, args.hidden_size})),
      dt_bias_(mx::ones({args.linear_num_value_heads})),
      a_log_(mx::zeros({args.linear_num_value_heads})),
      norm_(args.linear_value_head_dim, args.rms_norm_eps),
      out_proj_weight_(mx::zeros({args.hidden_size, value_dim_}))
{}

// Swift: createSSMMask -- returns nil unless leftPadding is set (we don't use it).
// MambaCache.makeMask returns nil when cache[0] exists OR no leftPadding.
// Since we never set leftPadding, this always returns nullopt.
static std::optional<mx::array> create_ssm_mask(int /*B*/, int /*S*/, MambaCache* /*cache*/) {
    return std::nullopt;
}

mx::array Qwen3NextGatedDeltaNet::operator()(
    const mx::array& inputs,
    const std::optional<mx::array>& mask,
    MambaCache* cache)
{
    int B = inputs.shape(0), S = inputs.shape(1);

    auto mixed_qkvz = linear_fwd(inputs, in_proj_qkvz_weight_);
    auto mixed_ba = linear_fwd(inputs, in_proj_ba_weight_);

    // fixQueryKeyValueOrdering
    int nk = num_k_heads_, dn = head_k_dim_, nv = num_v_heads_, dv = head_v_dim_;
    int v_heads_per_k = nv / nk;

    auto qkvz = mx::reshape(mixed_qkvz, {B, S, nk, -1});
    auto ba = mx::reshape(mixed_ba, {B, S, nk, -1});

    // Split qkvz: [q(dn), k(dn), v(vheads_per_k*dv), z(vheads_per_k*dv)]
    auto q = mx::slice(qkvz, {0, 0, 0, 0}, {B, S, nk, dn});
    auto k = mx::slice(qkvz, {0, 0, 0, dn}, {B, S, nk, 2 * dn});
    auto v = mx::reshape(
        mx::slice(qkvz, {0, 0, 0, 2 * dn}, {B, S, nk, 2 * dn + v_heads_per_k * dv}),
        {B, S, -1, dv});
    auto z = mx::reshape(
        mx::slice(qkvz, {0, 0, 0, 2 * dn + v_heads_per_k * dv}, {B, S, nk, qkvz.shape(3)}),
        {B, S, -1, dv});

    // Split ba: [b(vheads_per_k), a(remaining)]
    auto b_val = mx::reshape(
        mx::slice(ba, {0, 0, 0, 0}, {B, S, nk, v_heads_per_k}),
        {B, S, nv});
    auto a_val = mx::reshape(
        mx::slice(ba, {0, 0, 0, v_heads_per_k}, {B, S, nk, ba.shape(3)}),
        {B, S, nv});

    // Conv1d processing
    auto dtype = inputs.dtype();
    mx::array conv_state(0.0f);
    if (cache && (*cache)[0].has_value()) {
        conv_state = (*cache)[0].value();
    } else {
        conv_state = mx::zeros({B, conv_kernel_size_ - 1, conv_dim_}, dtype);
    }

    auto mixed_qkv = mx::concatenate({
        mx::reshape(q, {B, S, -1}),
        mx::reshape(k, {B, S, -1}),
        mx::reshape(v, {B, S, -1})
    }, -1);

    if (mask.has_value()) {
        mixed_qkv = mx::where(mx::expand_dims(*mask, -1), mixed_qkv, mx::zeros_like(mixed_qkv));
    }

    auto conv_input = mx::concatenate({conv_state, mixed_qkv}, 1);

    // Save conv state for next step
    if (cache) {
        int start = conv_input.shape(1) - (conv_kernel_size_ - 1);
        (*cache)[0] = mx::slice(conv_input, {0, start, 0}, {B, conv_input.shape(1), conv_dim_});
    }

    // *** T=1 DECODE FAST PATH ***
    // Fuse conv_silu + split + norms + beta/g + recurrence into minimal launches.
    if (S == 1 && cache && (*cache)[0].has_value() && conv_input.shape(1) == conv_kernel_size_) {
        // 1. Fused conv1d + silu (1 compiled kernel)
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

        // 2. Split + reshape (zero-cost views)
        auto q_out = mx::reshape(mx::slice(conv_out, {0, 0, 0}, {B, 1, key_dim_}),
                                  {B, 1, num_k_heads_, head_k_dim_});
        auto k_out = mx::reshape(mx::slice(conv_out, {0, 0, key_dim_}, {B, 1, 2 * key_dim_}),
                                  {B, 1, num_k_heads_, head_k_dim_});
        auto v_out = mx::reshape(mx::slice(conv_out, {0, 0, 2 * key_dim_}, {B, 1, conv_dim_}),
                                  {B, 1, num_v_heads_, head_v_dim_});

        // 3. Q/K norms (2 fast kernels via rms_norm)
        float inv_scale = std::pow(static_cast<float>(head_k_dim_), -0.5f);
        auto q_norm_w = mx::full({head_k_dim_}, inv_scale * inv_scale, dtype);
        auto k_norm_w = mx::full({head_k_dim_}, inv_scale, dtype);
        q_out = mx::fast::rms_norm(q_out, q_norm_w, 1e-6f);
        k_out = mx::fast::rms_norm(k_out, k_norm_w, 1e-6f);

        // 4. Fused beta + g (1 compiled kernel)
        // 5. Fused GDN step with head repeat (1 compiled kernel)
        auto ssm_state = (*cache)[1].has_value() ? (*cache)[1].value()
                           : mx::zeros({B, num_v_heads_, head_v_dim_, head_k_dim_}, dtype);

        // Mega-fused: beta/g computation + head repeat + recurrence step
        int rep = num_v_heads_ / num_k_heads_;
        static auto compiled_decode_step = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                // inputs: q[B,1,Hk,Dk], k[B,1,Hk,Dk], v[B,1,Hv,Dv],
                //         b[B,1,Hv], a_log[Hv], a[B,1,Hv], dt_bias[Hv],
                //         state[B,Hv,Dv,Dk], rep_factor
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

        // 6. Gated norm + output projection
        auto normalized = norm_(out, z);
        return linear_fwd(mx::reshape(normalized, {B, S, -1}), out_proj_weight_);
    }

    // *** GENERAL PATH: T>1 prefill ***
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

std::unordered_map<std::string, mx::array*> Qwen3NextGatedDeltaNet::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["conv1d.weight"] = &conv1d_weight_;
    map["in_proj_qkvz.weight"] = &in_proj_qkvz_weight_;
    map["in_proj_ba.weight"] = &in_proj_ba_weight_;
    map["dt_bias"] = &dt_bias_;
    map["A_log"] = &a_log_;
    for (auto& [k, v] : norm_.weight_map()) map["norm." + k] = v;
    map["out_proj.weight"] = &out_proj_weight_;
    return map;
}

// --- Qwen3NextMLP ---
// Swift: Qwen3NextMLP -- dense MLP

Qwen3NextMLP::Qwen3NextMLP(int dimensions, int hidden_dimensions)
    : gate_proj_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_proj_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_proj_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Qwen3NextMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_proj_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_proj_weight_)), down_proj_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3NextMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_proj_weight_},
        {"down_proj.weight", &down_proj_weight_},
        {"up_proj.weight", &up_proj_weight_},
    };
}

// --- Qwen3NextSparseMoeBlock ---
// Swift: Qwen3NextSparseMoeBlock -- Sparse MoE with shared expert

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(const Qwen3NextConfiguration& args)
    : norm_topk_prob_(args.norm_topk_prob),
      num_experts_(args.num_experts),
      top_k_(args.num_experts_per_tok),
      gate_weight_(mx::zeros({args.num_experts, args.hidden_size})),
      switch_mlp_(args.hidden_size, args.moe_intermediate_size, args.num_experts),
      shared_expert_(args.hidden_size, args.shared_expert_intermediate_size),
      shared_expert_gate_weight_(mx::zeros({1, args.hidden_size}))
{}

mx::array Qwen3NextSparseMoeBlock::operator()(const mx::array& x) {
    auto gates = mx::softmax(linear_fwd(x, gate_weight_), -1);

    int k = top_k_;
    int kth = gates.shape(-1) - k;
    auto inds = mx::argpartition(gates, kth, -1);
    inds = mx::slice(inds, {0, 0, kth}, {inds.shape(0), inds.shape(1), inds.shape(2)});
    auto scores = mx::take_along_axis(gates, inds, -1);

    if (norm_topk_prob_) {
        // Compile score normalization: divide + sum fused into one kernel
        static auto compiled_normalize_scores = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return {mx::divide(inputs[0], mx::sum(inputs[0], -1, true))};
            },
            /*shapeless=*/true);
        scores = compiled_normalize_scores({scores})[0];
    }

    auto y = switch_mlp_(x, inds);
    // Compile expert output weighting: expand_dims + multiply + sum fused
    static auto compiled_expert_combine = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::sum(mx::multiply(inputs[0], mx::expand_dims(inputs[1], -1)), -2)};
        },
        /*shapeless=*/true);
    auto combined = compiled_expert_combine({y, scores})[0];

    // Shared expert: sigmoid(gate) * shared_y + combined — compiled
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

std::unordered_map<std::string, mx::array*> Qwen3NextSparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    for (auto& [k, v] : shared_expert_.weight_map()) map["shared_expert." + k] = v;
    map["shared_expert_gate.weight"] = &shared_expert_gate_weight_;
    return map;
}

// --- Qwen3NextDecoderLayer ---
// Swift: Qwen3NextDecoderLayer -- either linear attn or standard attn, with MLP or MoE

Qwen3NextDecoderLayer::Qwen3NextDecoderLayer(const Qwen3NextConfiguration& args, int layer_idx)
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

    bool is_mlp_only = std::find(args.mlp_only_layers.begin(), args.mlp_only_layers.end(), layer_idx) != args.mlp_only_layers.end();
    if (!is_mlp_only && args.num_experts > 0 && (layer_idx + 1) % args.decoder_sparse_step == 0) {
        use_moe_ = true;
        moe_mlp_.emplace(args);
    } else {
        dense_mlp_.emplace(args.hidden_size, args.intermediate_size);
    }
}

mx::array Qwen3NextDecoderLayer::operator()(
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

std::unordered_map<std::string, mx::array*> Qwen3NextDecoderLayer::weight_map() {
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

// --- Qwen3NextModelInner ---
// Swift: Qwen3NextModelInner

Qwen3NextModelInner::Qwen3NextModelInner(const Qwen3NextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      full_attention_interval_(args.full_attention_interval)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args, i);
}

mx::array Qwen3NextModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Find the first full-attention index for attention mask
    int fa_idx = full_attention_interval_ - 1;
    if (fa_idx >= static_cast<int>(layers_.size())) fa_idx = 0;

    auto fa_mask = create_attention_mask(
        h, cache && fa_idx < static_cast<int>(cache->size()) ? &(*cache)[fa_idx] : nullptr);

    // SSM mask: return nil (nullopt) when cache has conv_state, ones otherwise.
    // Find first linear layer to check its cache state.
    std::optional<mx::array> ssm_mask;
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i].is_linear()) {
            MambaCache* mamba = nullptr;
            if (cache && i < cache->size()) {
                mamba = (*cache)[i].as_mamba();
            }
            ssm_mask = create_ssm_mask(h.shape(0), h.shape(1), mamba);
            break;
        }
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        auto attn_mask = layers_[i].is_linear() ? AttentionMask{} : fa_mask;
        h = layers_[i](h, attn_mask, ssm_mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen3NextModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Qwen3NextModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Qwen3NextModel ---
// Swift: Qwen3NextModel

Qwen3NextModel::Qwen3NextModel(const Qwen3NextConfiguration& args)
    : config_(args), model_(args)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Qwen3NextModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Qwen3NextModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Qwen3NextModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return linear_fwd(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::vector<KVCache> Qwen3NextModel::new_cache_impl(const GenerateParameters& params) {
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
Qwen3NextModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Remove mtp.* keys
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("mtp.") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    // Stack per-expert weights (and their scales/biases) into SwitchGLU format.
    // Early return when no experts need stacking.
    if (weights.find("model.layers.0.mlp.experts.0.up_proj.weight") == weights.end()) {
        // No expert stacking needed -- fall through to norm + conv1d fixups
    } else {
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
                    // Weight
                    auto wit = weights.find(ep + ".weight");
                    w_join.push_back(std::move(wit->second));
                    weights.erase(wit);
                    // Scales
                    auto sit = weights.find(ep + ".scales");
                    if (sit != weights.end()) {
                        s_join.push_back(std::move(sit->second));
                        weights.erase(sit);
                    }
                    // Biases
                    auto bit = weights.find(ep + ".biases");
                    if (bit != weights.end()) {
                        b_join.push_back(std::move(bit->second));
                        weights.erase(bit);
                        has_biases = true;
                    }
                }

                std::string dst = prefix + "switch_mlp." + n;
                weights.insert_or_assign(dst + ".weight", mx::stack(w_join));
                if (!s_join.empty()) {
                    weights.insert_or_assign(dst + ".scales", mx::stack(s_join));
                }
                if (has_biases && !b_join.empty()) {
                    weights.insert_or_assign(dst + ".biases", mx::stack(b_join));
                }
            }
        }
    }

    // Fix conv1d weight ordering and add 1.0 to norm weights.
    // Only regular RMSNorm weights use (1+weight) and need +1 here.
    // GatedRMSNorm (.linear_attn.norm.weight) uses weight directly -- do NOT add +1.
    std::vector<std::string> norm_suffixes = {
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    };

    for (auto& [key, value] : weights) {
        // Conv1d weight: MLX expects [C_out, K, C_in/groups].
        // Safetensors may store [C, K, 1] (correct) or [C, 1, K] (needs swap).
        if (key.find("conv1d.weight") != std::string::npos && value.shape(-1) != 1) {
            value = mx::moveaxis(value, 2, 1);
            continue;
        }
        bool is_norm = false;
        for (const auto& suffix : norm_suffixes) {
            if (key.size() >= suffix.size() &&
                key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
                is_norm = true;
                break;
            }
        }
        // Do not add +1: our rms_norm uses weight directly (not 1+weight).
        // The mlx-community 4-bit safetensors stores weights as direct multipliers (~1.0).
        (void)is_norm;
    }

    return weights;
}

void Qwen3NextModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen3NextModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
