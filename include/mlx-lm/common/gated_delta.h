// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of GatedDelta.swift -- Gated Delta Net recurrence for linear attention.
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <utility>

namespace mlx_lm {

// computeGatedDeltaG: decay = exp(-exp(aLog.float32) * softplus(a + dtBias))
// Returns result in a_log's dtype.
mlx::core::array compute_gated_delta_g(
    const mlx::core::array& a_log,
    const mlx::core::array& a,
    const mlx::core::array& dt_bias);

// Single-step recurrence for Gated Delta Net.
// q: [B, Hv, Dk], k: [B, Hv, Dk], v: [B, Hv, Dv], g: [B, Hv] or [B, Hv, Dv],
// beta: [B, Hv], state: [B, Hv, Dv, Dk]
// Returns (y, new_state).
std::pair<mlx::core::array, mlx::core::array> gated_delta_step_ops(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& g,
    const mlx::core::array& beta, const mlx::core::array& state,
    const std::optional<mlx::core::array>& mask = std::nullopt);

// Loop over T timesteps calling gated_delta_step_ops.
// q,k: [B, T, Hk, Dk], v: [B, T, Hv, Dv], g: [B, T, Hv], beta: [B, T, Hv]
// Returns (output [B, T, Hv, Dv], final_state [B, Hv, Dv, Dk]).
std::pair<mlx::core::array, mlx::core::array> gated_delta_ops(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& g,
    const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state = std::nullopt,
    const std::optional<mlx::core::array>& mask = std::nullopt);

// Full gated delta update: computes beta, g, initializes state, calls gated_delta_ops.
std::pair<mlx::core::array, mlx::core::array> gated_delta_update(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& a,
    const mlx::core::array& b, const mlx::core::array& a_log,
    const mlx::core::array& dt_bias,
    const std::optional<mlx::core::array>& state = std::nullopt,
    const std::optional<mlx::core::array>& mask = std::nullopt);

} // namespace mlx_lm
