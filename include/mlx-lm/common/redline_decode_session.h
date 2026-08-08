// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2 research: optional Redline session init (exp/redline-kernel-launch).
// Default OFF — only when MLX_REDLINE_DECODE=1. Does not replace product forward.

#pragma once

#include <string>

namespace mlx_lm {

// Result of one-shot session probe (process lifetime).
enum class RedlineSessionState {
    Disabled,   // env not exact "1" (or non-ROCm)
    XorEager,   // REDLINE + pure-graph both set → fail-closed
    Ready,      // dlopen + abi + gpu_new smoke OK
    Failed,     // dlopen/symbol/init failed; product path remains
};

// Lazy once: if env MLX_REDLINE_DECODE!=1, returns Disabled without loading.
// When =1: attempt dlopen of redline-capi (libredline_dispatch.so), resolve
// rl_abi_version + rl_gpu_new/rl_gpu_free, create ordinal-0 GPU, free it.
// Never enables HIP graphs. Never claims gen t/s. Does not change call_fn.
RedlineSessionState redline_session_ensure_init();

// One-shot stderr banner for P0/P2 (safe every step).
void maybe_log_redline_session_status();

// Human-readable last error (empty if none / disabled).
const std::string& redline_session_last_error();

} // namespace mlx_lm
