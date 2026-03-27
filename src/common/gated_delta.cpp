// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of GatedDelta.swift -- Gated Delta Net recurrence for linear attention.
//
// Optimized: compiled kernels for fusion, T=1 decode fast path.

#include <mlx-lm/common/gated_delta.h>

namespace mx = mlx::core;

namespace mlx_lm {

// ---------------------------------------------------------------------------
// computeGatedDeltaG — compiled to fuse ~9 element-wise ops into ~2 kernels
// Swift: decay = exp(-exp(aLog.float32) * softplus(a + dtBias))
// ---------------------------------------------------------------------------
mx::array compute_gated_delta_g(
    const mx::array& a_log, const mx::array& a, const mx::array& dt_bias)
{
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto a_log_f32 = mx::astype(inputs[0], mx::float32);
            auto softplus_val = mx::log(mx::add(mx::exp(mx::add(inputs[1], inputs[2])), mx::array(1.0f)));
            auto decay = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), softplus_val)));
            return {mx::astype(decay, inputs[1].dtype())};
        },
        /*shapeless=*/true);
    return compiled({a_log, a, dt_bias})[0];
}

// ---------------------------------------------------------------------------
// Compiled beta + g: fuses sigmoid(b) and compute_gated_delta_g into one kernel
// ---------------------------------------------------------------------------
static auto compiled_beta_and_g = mx::compile(
    [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto beta = mx::sigmoid(inputs[0]);
        auto a_log_f32 = mx::astype(inputs[1], mx::float32);
        auto softplus_val = mx::log(mx::add(mx::exp(mx::add(inputs[2], inputs[3])), mx::array(1.0f)));
        auto g = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), softplus_val)));
        g = mx::astype(g, inputs[1].dtype());
        return {beta, g};
    },
    /*shapeless=*/true);

// ---------------------------------------------------------------------------
// Compiled gated delta step: fuses ~9 ops into ~2 kernels for decode
// Inputs: [q, k, v, g, beta, state] → Outputs: [y, s_new]
// q: [B, Hv, Dk], k: [B, Hv, Dk], v: [B, Hv, Dv], g: [B, Hv],
// beta: [B, Hv], state: [B, Hv, Dv, Dk]
// ---------------------------------------------------------------------------
static auto compiled_gated_delta_step = mx::compile(
    [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto& q = inputs[0];
        auto& k = inputs[1];
        auto& v = inputs[2];
        auto& g = inputs[3];
        auto& beta = inputs[4];
        auto& state = inputs[5];

        auto decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
        auto s = mx::multiply(state, decay);
        auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
        auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
        s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
        auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);

        return {y, s};
    },
    /*shapeless=*/true);

// ---------------------------------------------------------------------------
// gatedDeltaStepOps — single timestep recurrence
// Uses compiled kernel for common decode case (no mask, g.ndim==2)
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_step_ops(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state,
    const std::optional<mx::array>& mask)
{
    // Fast path: no mask, g.ndim==2 (common decode case)
    if (!mask.has_value() && g.ndim() == 2) {
        auto results = compiled_gated_delta_step({q, k, v, g, beta, state});
        return {results[0], results[1]};
    }

    // Fallback for masked or unusual-rank cases
    auto old_state = state;

    mx::array decay(0.0f);
    if (g.ndim() == 2) {
        decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
    } else if (g.ndim() == 3) {
        decay = mx::expand_dims(g, -2);
    }

    auto s = mx::multiply(state, decay);
    auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
    auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
    s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
    auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);

    if (mask.has_value()) {
        mx::array expanded_mask(0.0f);
        if (mask->ndim() == 1) {
            expanded_mask = mx::expand_dims(mx::expand_dims(mx::expand_dims(*mask, -1), -1), -1);
        } else if (mask->ndim() == 2) {
            expanded_mask = mx::expand_dims(mx::expand_dims(*mask, -1), -1);
        } else if (mask->ndim() == 3) {
            expanded_mask = mx::expand_dims(*mask, -1);
        }
        s = mx::where(expanded_mask, s, old_state);
    }

    return {y, s};
}

// ---------------------------------------------------------------------------
// Repeat q/k from Hk heads to Hv heads using reshape + broadcast.
// ---------------------------------------------------------------------------
static mx::array repeat_heads(const mx::array& x, int repeat_factor) {
    if (repeat_factor <= 1) return x;
    int B = x.shape(0), T = x.shape(1), Hk = x.shape(2), D = x.shape(3);
    auto expanded = mx::reshape(x, {B, T, Hk, 1, D});
    auto tiled = mx::broadcast_to(expanded, {B, T, Hk, repeat_factor, D});
    return mx::reshape(tiled, {B, T, Hk * repeat_factor, D});
}

// ---------------------------------------------------------------------------
// gatedDeltaOps — loop over T timesteps
// For T=1 decode: skip slice/squeeze overhead, use compiled step directly
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_ops(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask)
{
    int B = q.shape(0), T = q.shape(1);
    int Hk = q.shape(2), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    int repeat_factor = Hv / Hk;
    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    // *** FAST PATH: T=1 decode — no loop, no slice/squeeze/concatenate ***
    if (T == 1 && !mask.has_value()) {
        auto q_t = mx::reshape(q, {B, Hk, Dk});
        auto k_t = mx::reshape(k, {B, Hk, Dk});
        auto v_t = mx::reshape(v, {B, Hv, Dv});
        auto g_t = mx::reshape(g, {B, g.shape(2)});
        auto beta_t = mx::reshape(beta, {B, beta.shape(2)});

        // Repeat q/k heads if needed
        if (repeat_factor > 1) {
            q_t = mx::reshape(mx::broadcast_to(
                mx::reshape(q_t, {B, Hk, 1, Dk}), {B, Hk, repeat_factor, Dk}),
                {B, Hv, Dk});
            k_t = mx::reshape(mx::broadcast_to(
                mx::reshape(k_t, {B, Hk, 1, Dk}), {B, Hk, repeat_factor, Dk}),
                {B, Hv, Dk});
        }

        auto [y, new_s] = gated_delta_step_ops(q_t, k_t, v_t, g_t, beta_t, s, std::nullopt);
        return {mx::expand_dims(y, 1), new_s};
    }

    // *** GENERAL PATH: T>1 prefill ***
    auto q_work = repeat_heads(q, repeat_factor);
    auto k_work = repeat_heads(k, repeat_factor);

    std::vector<mx::array> ys;
    ys.reserve(T);

    for (int t = 0; t < T; ++t) {
        auto q_t = mx::squeeze(mx::slice(q_work, {0, t, 0, 0}, {B, t + 1, q_work.shape(2), Dk}), 1);
        auto k_t = mx::squeeze(mx::slice(k_work, {0, t, 0, 0}, {B, t + 1, k_work.shape(2), Dk}), 1);
        auto v_t = mx::squeeze(mx::slice(v,      {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
        auto g_t = mx::squeeze(mx::slice(g,      {0, t, 0},    {B, t + 1, g.shape(2)}), 1);
        auto beta_t = mx::squeeze(mx::slice(beta, {0, t, 0},   {B, t + 1, beta.shape(2)}), 1);

        std::optional<mx::array> mask_t;
        if (mask.has_value()) {
            mask_t = mx::squeeze(mx::slice(*mask, {0, t}, {B, t + 1}), 1);
        }

        auto [y, new_s] = gated_delta_step_ops(q_t, k_t, v_t, g_t, beta_t, s, mask_t);
        ys.push_back(mx::expand_dims(y, 1));
        s = new_s;
    }

    return {mx::concatenate(ys, 1), s};
}

// ---------------------------------------------------------------------------
// gatedDeltaUpdate — compute beta+g fused, init state, call gatedDeltaOps
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_update(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& a,
    const mx::array& b, const mx::array& a_log,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask)
{
    // Fused beta + g computation
    auto bg = compiled_beta_and_g({b, a_log, a, dt_bias});
    auto& beta = bg[0];
    auto& g = bg[1];

    int B = q.shape(0), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    return gated_delta_ops(q, k, v, g, beta, s, mask);
}

} // namespace mlx_lm
