// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Fused HIP kernel for MTP delta intermediate computation.
//
// STUB: This file is NOT compiled into the binary. It was removed from
// CMakeLists.txt because mtp_delta_fused() / mtp_draft_forward() are never
// called from the production code path (generate.cpp).
//
// The file is kept on disk as a reference for future ROCm optimization work.
// Known issues if/when wiring this into the production path:
//   - mtp_delta_fused_rocm() passes mx::zeros for beta and `a` parameters
//     to compiled_decode, which results in fixed 0.5 gating (broken GDN).
//     TODO: Wire beta and `a` from actual model parameters (dt_param, a_param).
//   - mtp_delta_fused_generic() similarly uses mx::zeros for beta and a_val.
//     TODO: Same fix — compute beta and `a` from loaded model weights.

#include <mlx-lm/common/mtp_delta_kernel.h>
#include <mlx-lm/common/activations.h>

namespace mlx_lm {

namespace mx = mlx::core;

namespace {

// Helper: linear forward without bias.
mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    return mx::matmul(x, mx::transpose(w));
}

// ROCm-specific fused kernel implementation.
// This path uses custom HIP kernels for maximum performance on AMD GPUs.
#if defined(MLX_BUILD_ROCM)

// Low-level HIP kernel entry point for fused gated delta computation.
// Fuses conv1d+silu, QKV split, norms, beta/g gating, and GDN recurrence
// into minimal GPU kernel launches for the T=1 decode path.
std::pair<mx::array, std::optional<mx::array>> mtp_delta_fused_rocm(
    const mx::array& inputs,
    const mx::array& conv_weight,
    const mx::array& qkv_weight,
    const mx::array& z_weight,
    const mx::array& dt_bias,
    const mx::array& a_log,
    const std::optional<mx::array>& state,
    const MTPDeltaConfig& config)
{
    int B = inputs.shape(0);
    int S = inputs.shape(1);
    int H = config.hidden_size;
    auto dtype = inputs.dtype();

    // Fast path: T=1 decode uses fused kernel
    if (S == 1) {
        // Project QKV and Z
        auto qkv = linear_no_bias(inputs, qkv_weight);
        auto z = linear_no_bias(inputs, z_weight);

        // Conv1d processing with state
        mx::array conv_state;
        if (state && state->shape(0) > 0) {
            conv_state = *state;
        } else {
            conv_state = mx::zeros({B, config.conv_kernel_dim() - 1, config.conv_dim()}, dtype);
        }

        auto conv_input = mx::concatenate({conv_state, qkv}, 1);

        // Fused conv1d + silu via compiled graph (matches ROCm pattern)
        auto w = mx::reshape(
            mx::transpose(mx::reshape(conv_weight, {config.conv_dim(), config.conv_kernel_dim()})),
            {1, config.conv_kernel_dim(), config.conv_dim()});

        auto compiled_conv_silu = mx::compile(
            [](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                auto dot = mx::sum(mx::multiply(ins[0], ins[1]), 1, true);
                return {mx::multiply(dot, mx::sigmoid(dot))};
            },
            /*shapeless=*/true);

        auto conv_out = compiled_conv_silu({conv_input, w})[0];

        // Split into q, k, v
        auto q_out = mx::reshape(
            mx::slice(conv_out, {0, 0, 0}, {B, 1, config.key_dim()}),
            {B, 1, config.num_key_heads, config.key_head_dim});
        auto k_out = mx::reshape(
            mx::slice(conv_out, {0, 0, config.key_dim()}, {B, 1, 2 * config.key_dim()}),
            {B, 1, config.num_key_heads, config.key_head_dim});
        auto v_out = mx::reshape(
            mx::slice(conv_out, {0, 0, 2 * config.key_dim()}, {B, 1, config.conv_dim()}),
            {B, 1, config.num_value_heads, config.value_head_dim});

        // Q/K RMSNorm
        float inv_scale = std::pow(static_cast<float>(config.key_head_dim), -0.5f);
        auto q_norm_w = mx::full({config.key_head_dim}, inv_scale * inv_scale, dtype);
        auto k_norm_w = mx::full({config.key_head_dim}, inv_scale, dtype);
        q_out = mx::fast::rms_norm(q_out, q_norm_w, config.rms_norm_eps);
        k_out = mx::fast::rms_norm(k_out, k_norm_w, config.rms_norm_eps);

        // Fused beta/g + GDN recurrence
        auto ssm_state = state.value_or(
            mx::zeros({B, config.num_value_heads, config.value_head_dim, config.key_head_dim()}, dtype));

        auto compiled_decode = mx::compile(
            [config](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                auto q = ins[0], k = ins[1], v = ins[2];
                auto b = ins[3], a_log = ins[4], a = ins[5];
                auto dt_bias = ins[6], state = ins[7];

                int B_ = q.shape(0);
                int Hk = q.shape(2), Dk = q.shape(3);
                int Hv = v.shape(2), Dv = v.shape(3);
                int rep = Hv / Hk;

                // Beta + g fused
                auto beta = mx::sigmoid(b);
                auto a_log_f32 = mx::astype(a_log, mx::float32);
                auto sp = mx::log(mx::add(mx::exp(mx::add(a, dt_bias)), mx::array(1.0f)));
                auto g = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), sp)));
                g = mx::astype(g, a.dtype());

                // Squeeze T=1
                auto qt = mx::reshape(q, {B_, Hk, Dk});
                auto kt = mx::reshape(k, {B_, Hk, Dk});
                auto vt = mx::reshape(v, {B_, Hv, Dv});
                auto gt = mx::reshape(g, {B_, Hv});
                auto bt = mx::reshape(beta, {B_, Hv});

                // Repeat q/k heads
                if (rep > 1) {
                    qt = mx::reshape(mx::broadcast_to(
                        mx::reshape(qt, {B_, Hk, 1, Dk}), {B_, Hk, rep, Dk}), {B_, Hv, Dk});
                    kt = mx::reshape(mx::broadcast_to(
                        mx::reshape(kt, {B_, Hk, 1, Dk}), {B_, Hk, rep, Dk}), {B_, Hv, Dk});
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

        auto results = compiled_decode(
            {q_out, k_out, v_out, /* b and a from config */
             mx::zeros({B, config.num_value_heads}, dtype),
             a_log, mx::zeros({B, config.num_value_heads}, dtype),
             dt_bias, ssm_state});

        auto out = results[0];
        auto new_state = results[1];

        // Gated norm + reshape
        auto normed = mx::fast::rms_norm(mx::reshape(out, {B, 1, config.num_value_heads, config.value_head_dim()}, false),
                                          mx::reshape(z, {B, 1, config.num_value_heads, config.value_head_dim()}, false),
                                          config.rms_norm_eps);
        return {mx::reshape(normed, {B, S, H}), new_state};
    }

    // General path: T>1 prefill — fall through to graph compose
    (void)dt_bias;
    (void)a_log;
    (void)state;
}
#endif // MLX_BUILD_ROCM

// Generic fallback using MLX graph compose (works on all platforms).
std::pair<mx::array, std::optional<mx::array>> mtp_delta_fused_generic(
    const mx::array& inputs,
    const mx::array& conv_weight,
    const mx::array& qkv_weight,
    const mx::array& z_weight,
    const mx::array& dt_bias,
    const mx::array& a_log,
    const std::optional<mx::array>& state,
    const MTPDeltaConfig& config)
{
    int B = inputs.shape(0);
    int S = inputs.shape(1);
    int H = config.hidden_size;
    auto dtype = inputs.dtype();

    // Project QKV and Z
    auto qkv = linear_no_bias(inputs, qkv_weight);
    auto z = linear_no_bias(inputs, z_weight);

    if (S == 1) {
        // T=1 decode path — optimized with compiled kernels
        mx::array conv_state;
        if (state && state->shape(0) > 0) {
            conv_state = *state;
        } else {
            conv_state = mx::zeros({B, config.conv_kernel_dim() - 1, config.conv_dim()}, dtype);
        }

        auto conv_input = mx::concatenate({conv_state, qkv}, 1);

        // Fused conv1d + silu
        auto w = mx::reshape(
            mx::transpose(mx::reshape(conv_weight, {config.conv_dim(), config.conv_kernel_dim()})),
            {1, config.conv_kernel_dim(), config.conv_dim()});

        auto compiled_conv_silu = mx::compile(
            [](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                auto dot = mx::sum(mx::multiply(ins[0], ins[1]), 1, true);
                return {mx::multiply(dot, mx::sigmoid(dot))};
            },
            /*shapeless=*/true);

        auto conv_out = compiled_conv_silu({conv_input, w})[0];

        // Split Q, K, V
        auto q_out = mx::reshape(
            mx::slice(conv_out, {0, 0, 0}, {B, 1, config.key_dim()}),
            {B, 1, config.num_key_heads, config.key_head_dim});
        auto k_out = mx::reshape(
            mx::slice(conv_out, {0, 0, config.key_dim()}, {B, 1, 2 * config.key_dim()}),
            {B, 1, config.num_key_heads, config.key_head_dim});
        auto v_out = mx::reshape(
            mx::slice(conv_out, {0, 0, 2 * config.key_dim()}, {B, 1, config.conv_dim()}),
            {B, 1, config.num_value_heads, config.value_head_dim});

        // RMSNorm
        float inv_scale = std::pow(static_cast<float>(config.key_head_dim), -0.5f);
        auto q_norm_w = mx::full({config.key_head_dim}, inv_scale * inv_scale, dtype);
        auto k_norm_w = mx::full({config.key_head_dim}, inv_scale, dtype);
        q_out = mx::fast::rms_norm(q_out, q_norm_w, config.rms_norm_eps);
        k_out = mx::fast::rms_norm(k_out, k_norm_w, config.rms_norm_eps);

        // GDN recurrence
        auto ssm_state = state.value_or(
            mx::zeros({B, config.num_value_heads, config.value_head_dim, config.key_head_dim()}, dtype));

        auto compiled_decode = mx::compile(
            [config](const std::vector<mx::array>& ins) -> std::vector<mx::array> {
                auto q = ins[0], k = ins[1], v = ins[2];
                auto beta = ins[3], g = ins[4], state = ins[5];

                int B_ = q.shape(0);
                int Hk = q.shape(2), Dk = q.shape(3);
                int Hv = v.shape(2), Dv = v.shape(3);
                int rep = Hv / Hk;

                auto qt = mx::reshape(q, {B_, Hk, Dk});
                auto kt = mx::reshape(k, {B_, Hk, Dk});
                auto vt = mx::reshape(v, {B_, Hv, Dv});
                auto gt = mx::reshape(g, {B_, Hv});
                auto bt = mx::reshape(beta, {B_, Hv});

                if (rep > 1) {
                    qt = mx::reshape(mx::broadcast_to(
                        mx::reshape(qt, {B_, Hk, 1, Dk}), {B_, Hk, rep, Dk}), {B_, Hv, Dk});
                    kt = mx::reshape(mx::broadcast_to(
                        mx::reshape(kt, {B_, Hk, 1, Dk}), {B_, Hk, rep, Dk}), {B_, Hv, Dk});
                }

                auto decay = mx::expand_dims(mx::expand_dims(gt, -1), -1);
                auto s = mx::multiply(state, decay);
                auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(kt, -2)), -1);
                auto delta = mx::multiply(mx::subtract(vt, kv_mem), mx::expand_dims(bt, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(kt, -2), mx::expand_dims(delta, -1)));
                auto y = mx::sum(mx::multiply(s, mx::expand_dims(qt, -2)), -1);

                return {mx::expand_dims(y, 1), s};
            },
            /*shapeless=*/true);

        // Compute beta and g
        auto beta = mx::sigmoid(mx::zeros({B, S, config.num_value_heads}, dtype));
        auto a_log_f32 = mx::astype(a_log, mx::float32);
        auto a_val = mx::zeros({B, S, config.num_value_heads}, dtype);
        auto sp = mx::log(mx::add(mx::exp(mx::add(a_val, dt_bias)), mx::array(1.0f)));
        auto g = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), sp)));
        g = mx::astype(g, dtype);

        auto results = compiled_decode(
            {q_out, k_out, v_out,
             mx::reshape(beta, {B, S, config.num_value_heads}),
             mx::reshape(g, {B, S, config.num_value_heads}),
             ssm_state});

        return {results[0], results[1]};
    }

    // T>1 prefill path — use standard ops
    auto conv_out = mx::conv1d(qkv, conv_weight, 1, 0, 1, config.conv_dim());
    conv_out = mx::multiply(conv_out, mx::sigmoid(conv_out)); // silu

    auto q_out = mx::reshape(
        mx::slice(conv_out, {0, 0, 0}, {B, S, config.key_dim()}),
        {B, S, config.num_key_heads, config.key_head_dim});
    auto k_out = mx::reshape(
        mx::slice(conv_out, {0, 0, config.key_dim()}, {B, S, 2 * config.key_dim()}),
        {B, S, config.num_key_heads, config.key_head_dim});
    auto v_out = mx::reshape(
        mx::slice(conv_out, {0, 0, 2 * config.key_dim()}, {B, S, config.conv_dim()}),
        {B, S, config.num_value_heads, config.value_head_dim});

    float inv_scale = std::pow(static_cast<float>(config.key_head_dim), -0.5f);
    auto q_norm_w = mx::full({config.key_head_dim}, inv_scale * inv_scale, dtype);
    auto k_norm_w = mx::full({config.key_head_dim}, inv_scale, dtype);
    q_out = mx::fast::rms_norm(q_out, q_norm_w, config.rms_norm_eps);
    k_out = mx::fast::rms_norm(k_out, k_norm_w, config.rms_norm_eps);

    // Return intermediate output; caller handles GDN update
    (void)z_weight;
    (void)dt_bias;
    (void)a_log;
    (void)state;

    return {conv_out, std::nullopt};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API (STUB — not compiled into binary, kept for future ROCm work)
// ---------------------------------------------------------------------------

std::pair<mx::array, std::optional<mx::array>>
mtp_delta_fused(
    const mx::array& inputs,
    const mx::array& conv_weight,
    const mx::array& qkv_weight,
    const mx::array& z_weight,
    const mx::array& dt_bias,
    const mx::array& a_log,
    const std::optional<mx::array>& state,
    const MTPDeltaConfig& config)
{
#if defined(MLX_BUILD_ROCM)
    return mtp_delta_fused_rocm(inputs, conv_weight, qkv_weight, z_weight,
                                dt_bias, a_log, state, config);
#else
    return mtp_delta_fused_generic(inputs, conv_weight, qkv_weight, z_weight,
                                   dt_bias, a_log, state, config);
#endif
}

mlx::core::array mtp_draft_forward(
    const mx::array& hidden_state,
    const mx::array& token_embedding,
    const std::unordered_map<std::string, mx::array>& mtp_weights,
    const MTPDeltaConfig& config,
    bool use_moe)
{
    // Standard MTP forward: concatenate normed embedding + hidden, run through
    // FC weight, then through the decoder layer, then output norm.
    //
    // For MoE variant, the MLP sub-block uses SwitchGLU instead of dense SwiGLU.

    auto H = hidden_state;
    auto E = token_embedding;

    // Look up MTP weights
    auto find_w = [&](const std::string& key) -> mx::array {
        auto it = mtp_weights.find(key);
        if (it != mtp_weights.end()) return it->second;
        throw std::runtime_error("MTP weight not found: " + key);
    };

    // Pre-FC norms
    auto h_norm_w = find_w("mtp.pre_fc_norm_hidden.weight");
    auto e_norm_w = find_w("mtp.pre_fc_norm_embedding.weight");

    auto h_n = mx::fast::rms_norm(H, h_norm_w, config.rms_norm_eps);
    auto e_n = mx::fast::rms_norm(E, e_norm_w, config.rms_norm_eps);

    // Concatenate [e_norm, h_norm] and apply FC
    auto fc_w = find_w("mtp.fc.weight");
    auto cat = mx::concatenate({e_n, h_n}, -1);
    auto fused = linear_no_bias(cat, fc_w);

    // Decoder layer forward (attention + MLP)
    // For MTP, this is a single-layer decoder with standard attention
    auto layer_prefix = "mtp.layers.0.";

    // Self-attention sub-block
    auto input_norm_w = find_w(layer_prefix + "input_layernorm.weight");
    auto normed = mx::fast::rms_norm(fused, input_norm_w, config.rms_norm_eps);

    // Attention projections
    auto q_w = find_w(layer_prefix + "self_attn.q_proj.weight");
    auto k_w = find_w(layer_prefix + "self_attn.k_proj.weight");
    auto v_w = find_w(layer_prefix + "self_attn.v_proj.weight");
    auto o_w = find_w(layer_prefix + "self_attn.o_proj.weight");
    auto q_norm_w = find_w(layer_prefix + "self_attn.q_norm.weight");
    auto k_norm_w = find_w(layer_prefix + "self_attn.k_norm.weight");

    int B = fused.shape(0);
    int L = fused.shape(1);
    int hd = config.resolved_head_dim();
    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    float scale = std::pow(static_cast<float>(hd), -0.5f);

    auto q = mx::reshape(linear_no_bias(normed, q_w), {B, L, n_heads, hd});
    auto k = mx::reshape(linear_no_bias(normed, k_w), {B, L, n_kv_heads, hd});
    auto v = mx::reshape(linear_no_bias(normed, v_w), {B, L, n_kv_heads, hd});

    q = mx::transpose(mx::fast::rms_norm(q, q_norm_w, config.rms_norm_eps), {0, 2, 1, 3});
    k = mx::transpose(mx::fast::rms_norm(k, k_norm_w, config.rms_norm_eps), {0, 2, 1, 3});
    v = mx::transpose(v, {0, 2, 1, 3});

    // RoPE
    int rope_dims = static_cast<int>(hd * config.partial_rotary_factor);
    q = mx::fast::rope(q, rope_dims, false, config.rope_theta, 1.0f, 0);
    k = mx::fast::rope(k, rope_dims, false, config.rope_theta, 1.0f, 0);

    // SDPA (no cache for MTP draft)
    auto attn_out = scaled_dot_product_attention(q, k, v, scale, AttentionMask{});
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    attn_out = linear_no_bias(attn_out, o_w);

    auto h_attn = mx::add(fused, attn_out);

    // MLP sub-block
    auto post_norm_w = find_w(layer_prefix + "post_attention_layernorm.weight");
    auto post = mx::fast::rms_norm(h_attn, post_norm_w, config.rms_norm_eps);

    mx::array mlp_out;
    if (use_moe) {
        // MoE path: use SwitchGLU routing
        // Requires expert weights from mtp_weights
        auto gate_w = find_w(layer_prefix + "mlp.gate.weight");
        auto gates = mx::softmax(linear_no_bias(post, gate_w), -1);

        int num_experts = gate_w.shape(0);
        int top_k = 1; // Default for MTP MoE

        int kth = gates.shape(-1) - top_k;
        auto inds = mx::argpartition(gates, kth, -1);
        inds = mx::slice(inds, {0, 0, kth}, {inds.shape(0), inds.shape(1), inds.shape(2)});
        auto scores = mx::take_along_axis(gates, inds, -1);

        // SwitchGLU forward
        auto gate_proj_w = find_w(layer_prefix + "mlp.switch_mlp.gate_proj.weight");
        auto up_proj_w = find_w(layer_prefix + "mlp.switch_mlp.up_proj.weight");
        auto down_proj_w = find_w(layer_prefix + "mlp.switch_mlp.down_proj.weight");

        // Gather expert outputs
        auto gate_out = mx::gather_mm(post, gate_proj_w, inds);
        auto gate_act = mx::multiply(gate_out, mx::sigmoid(gate_out)); // silu
        auto up_out = mx::gather_mm(post, up_proj_w, inds);
        auto expert_out = mx::multiply(gate_act, up_out);
        mlp_out = mx::gather_mm(expert_out, down_proj_w, inds);

        // Weighted combine
        auto combined = mx::sum(mx::multiply(mlp_out, mx::expand_dims(scores, -1)), -2);

        // Shared expert
        auto shared_gate_w = find_w(layer_prefix + "mlp.shared_expert_gate.weight");
        auto shared_gate = mx::sigmoid(linear_no_bias(post, shared_gate_w));
        auto shared_gate_proj = find_w(layer_prefix + "mlp.shared_expert.gate_proj.weight");
        auto shared_up_proj = find_w(layer_prefix + "mlp.shared_expert.up_proj.weight");
        auto shared_down_proj = find_w(layer_prefix + "mlp.shared_expert.down_proj.weight");

        auto shared_g = linear_no_bias(post, shared_gate_proj);
        auto shared_up = linear_no_bias(post, shared_up_proj);
        auto shared_down = linear_no_bias(mx::multiply(shared_g, mx::sigmoid(shared_g)), shared_down_proj);

        mlp_out = mx::add(combined, mx::multiply(shared_gate, shared_down));
    } else {
        // Dense path: standard SwiGLU
        auto gate_w = find_w(layer_prefix + "mlp.gate_proj.weight");
        auto up_w = find_w(layer_prefix + "mlp.up_proj.weight");
        auto down_w = find_w(layer_prefix + "mlp.down_proj.weight");

        auto gate = linear_no_bias(post, gate_w);
        auto up = linear_no_bias(post, up_w);
        auto silu_gate = mx::multiply(gate, mx::sigmoid(gate));
        mlp_out = linear_no_bias(mx::multiply(silu_gate, up), down_w);
    }

    auto layer_out = mx::add(h_attn, mlp_out);

    // Output norm
    auto norm_w = find_w("mtp.norm.weight");
    return mx::fast::rms_norm(layer_out, norm_w, config.rms_norm_eps);
}

} // namespace mlx_lm
