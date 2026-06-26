// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of GatedDelta.swift -- Gated Delta Net recurrence for linear attention.
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <tuple>
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
    const std::optional<mlx::core::array>& mask = std::nullopt,
    // When true, the fused decode kernel writes the new SSM state in place.
    bool inplace_state = false);

// Full gated delta update: computes beta, g, initializes state, calls gated_delta_ops.
std::pair<mlx::core::array, mlx::core::array> gated_delta_update(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& a,
    const mlx::core::array& b, const mlx::core::array& a_log,
    const mlx::core::array& dt_bias,
    const std::optional<mlx::core::array>& state = std::nullopt,
    const std::optional<mlx::core::array>& mask = std::nullopt,
    bool inplace_state = false);

// In-place write of `src` into `dst`'s device buffer (same total size required).
mlx::core::array inplace_write(const mlx::core::array& dst,
                              const mlx::core::array& src);

// In-place KV-cache slice write: writes new_kv [B,H,N,D] into cache [B,H,ALLOC,D]
// at [:,:,offset:offset+N,:]. The output ALIASES the cache buffer, so no copy is
// made (replaces slice_update, whose donation fails under the async pipeline and
// copies the whole cache → variable copy count → non-replayable decode graph).
mlx::core::array kv_inplace_update(
    const mlx::core::array& cache, const mlx::core::array& new_kv, int offset);

// FlashQLA-style fused GDN decode step (T=1): folds q/k-RMSNorm + beta/g +
// the delta recurrence into ONE kernel (replaces rms_norm(q)+rms_norm(k)+
// compiled beta/g + gated_delta_step). q,k: [B,1,Hk,Dk], v: [B,1,Hv,Dv],
// a,b: [B,1,Hv], a_log,dt_bias: [Hv], q_norm_w,k_norm_w: [Dk],
// state: [B,Hv,Dv,Dk]. Returns (y [B,1,Hv,Dv], new_state). The output gate
// (norm_, reduces over Dv) stays separate. MLX_GDN_FUSED2_MXOPS=1 falls back.
std::pair<mlx::core::array, mlx::core::array> gdn_fused_decode(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& a,
    const mlx::core::array& b, const mlx::core::array& a_log,
    const mlx::core::array& dt_bias,
    const mlx::core::array& q_norm_w, const mlx::core::array& k_norm_w,
    const mlx::core::array& state);

// Fused GDN conv1d decode step: causal depthwise conv (KS taps) + silu + state
// shift in one kernel. conv_state [B,KS-1,CD], qkv [B,1,CD], weight [CD,1,KS].
// Returns (conv_out [B,1,CD] silu'd, new_state [B,KS-1,CD]). Replaces the
// concatenate+slice+conv+silu op chain (kills their copy kernels).
std::pair<mlx::core::array, mlx::core::array> gdn_conv_step(
    const mlx::core::array& conv_state,
    const mlx::core::array& qkv,
    const mlx::core::array& weight);

// Fused residual-add + RMSNorm. Returns (sum = a+b, normed = rmsnorm(sum)*weight)
// in one kernel — eliminates the standalone residual add and keeps the sum
// on-chip for the norm. a,b: [..., H]; weight: [H].
std::pair<mlx::core::array, mlx::core::array> add_rms_norm(
    const mlx::core::array& a,
    const mlx::core::array& b,
    const mlx::core::array& weight,
    float eps);

// Fused gated RMSNorm (GDN/attention output gate): silu(gate) * rmsnorm(x) *
// weight in one kernel. Replaces rms_norm + sigmoid + multiply. x,gate: [..,H],
// weight: [H]. MLX_FUSED_NORM_MXOPS=1 falls back to the op chain.
mlx::core::array gated_rms_norm(
    const mlx::core::array& x, const mlx::core::array& gate,
    const mlx::core::array& weight, float eps);

// Fused MoE router (norm_topk_prob): top-k of the router logits + softmax over
// just those k, in one kernel (replaces argpartition+slice+take_along+softmax).
// Returns (indices [.., k] uint32, scores [.., k]). ROCm fast path needs
// E % 32 == 0 and k <= 16; otherwise falls back to argpartition.
std::pair<mlx::core::array, mlx::core::array> moe_route(
    const mlx::core::array& logits,
    int k);

// Speculative-decoding variant of gated_delta_ops: also returns the per-token
// SSM state stack `state_seq` [B, T, Hv, Dv, Dk] (state after each token).
// Returns (output [B, T, Hv, Dv], final_state [B, Hv, Dv, Dk], state_seq).
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> gated_delta_ops_seq(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& g,
    const mlx::core::array& beta,
    const std::optional<mlx::core::array>& state = std::nullopt,
    const std::optional<mlx::core::array>& mask = std::nullopt);

// Like gated_delta_update, but also returns the per-token state stack (see
// gated_delta_ops_seq). Used by speculative decoding verification.
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array> gated_delta_update_seq(
    const mlx::core::array& q, const mlx::core::array& k,
    const mlx::core::array& v, const mlx::core::array& a,
    const mlx::core::array& b, const mlx::core::array& a_log,
    const mlx::core::array& dt_bias,
    const std::optional<mlx::core::array>& state = std::nullopt,
    const std::optional<mlx::core::array>& mask = std::nullopt);

} // namespace mlx_lm
