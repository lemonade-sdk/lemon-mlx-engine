// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of GatedDelta.swift -- Gated Delta Net recurrence for linear attention.
//
// Faithful 1:1 port. No compiled kernels, no float32 upcast, no fast paths.

#include <mlx-lm/common/gated_delta.h>

namespace mx = mlx::core;

namespace mlx_lm {

// ---------------------------------------------------------------------------
// computeGatedDeltaG
// Swift: decay = exp(-exp(aLog.float32) * softplus(a + dtBias))
//        return decay cast to a.dtype
// ---------------------------------------------------------------------------
mx::array compute_gated_delta_g(
    const mx::array& a_log, const mx::array& a, const mx::array& dt_bias)
{
    auto a_log_f32 = mx::astype(a_log, mx::float32);
    auto softplus_val = mx::log(mx::add(mx::exp(mx::add(a, dt_bias)), mx::array(1.0f)));
    auto decay = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), softplus_val)));
    return mx::astype(decay, a.dtype());
}

// ---------------------------------------------------------------------------
// gatedDeltaStepOps — single timestep recurrence
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_step_ops(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state,
    const std::optional<mx::array>& mask)
{
    auto old_state = state;

    // expand_dims(g) based on ndim:
    //   2 -> axes [2, 3] (two expand_dims at -1)
    //   3 -> axis [-2]   (one expand_dims at -2)
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
    // x: [B, T, Hk, D] -> [B, T, Hk, 1, D] -> broadcast -> [B, T, Hk*rep, D]
    int B = x.shape(0), T = x.shape(1), Hk = x.shape(2), D = x.shape(3);
    auto expanded = mx::reshape(x, {B, T, Hk, 1, D});
    auto tiled = mx::broadcast_to(expanded, {B, T, Hk, repeat_factor, D});
    return mx::reshape(tiled, {B, T, Hk * repeat_factor, D});
}

// ---------------------------------------------------------------------------
// gatedDeltaOps — loop over T timesteps
// Swift: no float32 upcast, state = state ?? zeros([B, Hv, Dv, Dk], q.dtype)
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

    // Repeat q,k from Hk to Hv heads if needed
    auto q_work = repeat_heads(q, repeat_factor);
    auto k_work = repeat_heads(k, repeat_factor);

    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

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

    // Stack outputs along axis 1
    return {mx::concatenate(ys, 1), s};
}

// ---------------------------------------------------------------------------
// gatedDeltaUpdate — compute beta, g, init state, call gatedDeltaOps
// Swift: beta = sigmoid(b), g = computeGatedDeltaG(aLog, a, dtBias)
//        state = state ?? zeros([B, Hv, Dv, Dk], q.dtype)
//        No Metal kernel on ROCm — just call gatedDeltaOps directly.
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_update(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& a,
    const mx::array& b, const mx::array& a_log,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask)
{
    auto beta = mx::sigmoid(b);
    auto g = compute_gated_delta_g(a_log, a, dt_bias);

    int B = q.shape(0), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    return gated_delta_ops(q, k, v, g, beta, s, mask);
}

} // namespace mlx_lm
