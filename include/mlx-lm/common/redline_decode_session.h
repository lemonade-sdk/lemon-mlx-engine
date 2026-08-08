// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2/P5/P6/P7 research: optional Redline session (exp/redline-kernel-launch).
// Default OFF — only when MLX_REDLINE_DECODE=1. Does not replace product forward.
//
// P5 micro-op (opt-in HSACO): retained-PM4 load+patch+replay correctness.
// P6 graph_decode bind: stable input/pos VRAM ptrs; bake pos as micro acc.
// P7 sidecar (MLX_REDLINE_SIDECAR=1): arm retained IB after micro; L=1 ticks
// patch+replay alongside product call_fn (instrumentation). NOT gen t/s.

#pragma once

#include <string>

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
// PM4 micro; if also MLX_REDLINE_SIDECAR=1 and micro PASS, arm P7 retained IB.
// Never enables HIP graphs. Never claims gen t/s. Does not change call_fn.
RedlineSessionState redline_session_ensure_init();

// One-shot stderr banner for P0/P2/P5/P6/P7 (safe every step).
void maybe_log_redline_session_status();

// P6: one-shot probe of stable graph_decode_input/pos buffer pointers
// (device VRAM ptrs). Safe every L=1 step; no-ops unless MLX_REDLINE_DECODE=1.
// Does not change product forward. Logs [redline] gd_bind PASS|FAIL once.
void maybe_probe_redline_graph_decode_bind();

// P7: optional L=1 retained-PM4 patch+replay tick. No-op unless
// MLX_REDLINE_DECODE=1 and MLX_REDLINE_SIDECAR=1 and session armed.
// Does not replace call_fn. NOT gen t/s.
void maybe_redline_sidecar_l1();

// P7b: D2H side_acc vs host triangular sum after L=1 ticks (full-gen verify).
// No-op unless DECODE=1 + SIDECAR=1 + armed and at least one L=1 tick ran.
// Logs [redline] sidecar L1 fullgen PASS|FAIL once per call. NOT gen t/s.
// Does not replace call_fn. Safe from TokenIterator destructor / end of gen.
void maybe_redline_sidecar_verify();

// Human-readable last error / status detail (empty if none / disabled).
const std::string& redline_session_last_error();

} // namespace mlx_lm
