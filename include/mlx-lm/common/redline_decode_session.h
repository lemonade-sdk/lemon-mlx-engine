// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2/P5/P6/P7/P8 research: optional Redline session (exp/redline-kernel-launch).
// Default OFF — only when MLX_REDLINE_DECODE=1. Does not replace product forward.
//
// P5 micro-op (opt-in HSACO): retained-PM4 load+patch+replay correctness.
// P6 graph_decode bind: stable input/pos VRAM ptrs; bake pos as micro acc.
// P7 sidecar (MLX_REDLINE_SIDECAR=1): arm retained IB after micro; L=1 ticks
// patch+replay alongside product call_fn (instrumentation). NOT gen t/s.
// P8 small-op (MLX_REDLINE_SMALL_OP=1): engine-owned L=1 op that writes/reads
// live graph_decode_input VRAM and drives retained PM4 from product token ids.
// Does not replace call_fn. NOT gen t/s.
// P12 OWN_RMSNORM (MLX_REDLINE_OWN_RMSNORM=1): replace packed product RMSNorm
// HIP launches with Redline retained PM4 (multi-instance non-qmm family).
// Default OFF. Mid-eval uses HIP stream sync for ordering (documented tax).
// P12b: MLX_REDLINE_POST_SYNC=device|stream|off (default device) — post-replay
// fence policy for dual-queue tax A/B. stream/off may race; research only.
// P12c: set_k-before-pre-sync overlap; MLX_REDLINE_PRE_SYNC=stream|device|off
// (default stream); MLX_REDLINE_RMS_PROFILE=1 host-phase timers. Default OFF.

#pragma once

#include <cstdint>
#include <string>

namespace mlx {
namespace core {
class array;
} // namespace core
} // namespace mlx

// C ABI for MLX weak-hook (libmlx.a → chat resolves strong symbol).
// dtype: 0=f32, 1=f16, 2=bf16. hip_stream may be null (device sync).
// Returns true if Redline handled the launch (caller must not HIP-launch).
extern "C" bool mlx_redline_try_own_rmsnorm(
    const void* x,
    const void* w,
    void* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride,
    uint32_t n_rows,
    int dtype_code,
    void* hip_stream);

namespace mlx_lm {

// Result of one-shot session probe (process lifetime).
enum class RedlineSessionState {
    Disabled,   // env not exact "1" (or non-ROCm)
    XorEager,   // REDLINE + pure-graph both set → fail-closed
    Ready,      // dlopen + abi + gpu_new smoke OK (micro optional)
    Failed,     // dlopen/symbol/init failed; product path remains
};

// Lazy once: if env MLX_REDLINE_DECODE!=1, returns Disabled without loading.
// When =1: dlopen redline-capi; gpu_new smoke; if MLX_REDLINE_HSACO set, P5/P6
// PM4 micro; if also SIDECAR=1 or SMALL_OP=1 and micro PASS, arm retained IB.
// Never enables HIP graphs. Never claims gen t/s. Does not change call_fn.
RedlineSessionState redline_session_ensure_init();

// One-shot stderr banner for P0/P2/P5–P9 (safe every step).
void maybe_log_redline_session_status();

// P6: one-shot probe of stable graph_decode_input/pos buffer pointers
// (device VRAM ptrs). Safe every L=1 step; no-ops unless MLX_REDLINE_DECODE=1.
// Does not change product forward. Logs [redline] gd_bind PASS|FAIL once.
void maybe_probe_redline_graph_decode_bind();

// P7: optional L=1 retained-PM4 patch+replay tick (synthetic n=1,2,...).
// No-op unless DECODE=1 + SIDECAR=1 and armed. Skipped when SMALL_OP=1
// (product-buffer-driven ticks own the retained IB). Does not replace call_fn.
void maybe_redline_sidecar_l1();

// P7b/P8: D2H side_acc vs host expected after L=1 ticks (full-gen verify).
// SIDECAR-only: triangular sum. SMALL_OP: sum of product token ids from
// graph_decode_input VRAM. Logs PASS|FAIL once per call. NOT gen t/s.
// Does not replace call_fn. Safe from TokenIterator destructor / end of gen.
void maybe_redline_sidecar_verify();

// P8: engine-owned L=1 small op consuming live graph_decode_* VRAM.
// No-op unless DECODE=1 + SMALL_OP=1 and armed. Writes previous token into
// graph_decode_input, D2H via device ptr, retained PM4 patch+replay.
// Does not set graph_external_pos (product RoPE stays host-offset).
// Does not replace call_fn. NOT gen t/s.
void maybe_redline_small_op_l1(mlx::core::array& previous_token);

// P9: own product decode glue launches (replace mlx HIP for these ops only).
// When MLX_REDLINE_DECODE=1 and MLX_REDLINE_OWN_GLUE=1 and glue CO armed,
// set_graph_decode_pos / advance / set_graph_decode_input_from route here
// instead of gpu_kv_pos_* / gpu_scalar_copy_i32. Returns true if Redline
// handled the launch (caller must not fall back). Default OFF. NOT gen t/s.
bool redline_try_own_pos_set(mlx::core::array& pos, int v);
bool redline_try_own_pos_inc(mlx::core::array& pos, int delta);
bool redline_try_own_scalar_copy_i32(mlx::core::array& dst, mlx::core::array& src);

// P12: own packed product RMSNorm (see mlx_redline_try_own_rmsnorm C ABI).
// Product route is via weak hook in MLX rms_norm.hip.
bool redline_try_own_rmsnorm_packed(
    const void* x,
    const void* w,
    void* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride,
    uint32_t n_rows,
    int dtype_code,
    void* hip_stream);

// Human-readable last error / status detail (empty if none / disabled).
const std::string& redline_session_last_error();

} // namespace mlx_lm
