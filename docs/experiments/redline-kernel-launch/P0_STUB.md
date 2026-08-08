# P0 — `MLX_REDLINE_DECODE` stub (shipped, default OFF)

**Date:** 2026-08-07  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **GREEN** on host ROCm / gfx1150 smoke  
**Prereqs:** E0–E4 design/measure

---

## Goal

Implement the E4 §5 / §10 **P0** deliverable only:

| Item | Result |
|------|--------|
| Env `MLX_REDLINE_DECODE` | Exact `"1"` only; default unset/other → silent product eager |
| Banner | One-shot stderr when enabled; text says **session not implemented** |
| Product path | Unchanged (still `context_.call_fn` / eager HIP) |
| CMake | `MLX_LM_WITH_REDLINE` **OFF** by default; notes for optional redline-capi link |
| HIP graphs | **Not** enabled; XOR with `MLX_DECODE_GRAPH_PURE=1` → fail-closed eager |

---

## Code sites

| Site | Change |
|------|--------|
| [`src/common/generate.cpp`](../../../src/common/generate.cpp) | `redline_decode_env_enabled_()`, `maybe_log_redline_p0_stub_()` under `#if defined(MLX_BUILD_ROCM)`; called from `TokenIterator::step` (L=1) and `next()` (XOR pure path) |
| [`CMakeLists.txt`](../../../CMakeLists.txt) | `option(MLX_LM_WITH_REDLINE ... OFF)` + comment block (no forced link) |

**Not done (P1+):** redline-capi link, retained AQL session, HSACO load, any gen t/s claim.

---

## Smoke evidence (this GPU)

Host: rebuild `chat` exit 0. Model: `mlx-community/Qwen3.5-0.8B-4bit` local snapshot.  
Logs under [`logs/`](logs/):

| Case | Env | Expected | Log |
|------|-----|----------|-----|
| default | unset | 0× `[redline]` | [`logs/p0-off-20260807-215209.err`](logs/p0-off-20260807-215209.err) |
| opt-in | `MLX_REDLINE_DECODE=1` | **1×** not-implemented banner | [`logs/p0-on-20260807-215209.err`](logs/p0-on-20260807-215209.err) |
| XOR | `=1` + `MLX_DECODE_GRAPH_PURE=1` | **1×** fail-closed XOR banner; pure disabled | [`logs/p0-xor-pure-20260807-215209.err`](logs/p0-xor-pure-20260807-215209.err) |
| non-exact | `MLX_REDLINE_DECODE=true` | 0× `[redline]` | [`logs/p0-true-20260807-215209.err`](logs/p0-true-20260807-215209.err) |

Banner line (opt-in):

```text
[redline] MLX_REDLINE_DECODE=1: session not implemented (P0 stub — product eager path unchanged; see docs/experiments/redline-kernel-launch/E4_DESIGN.md)
```

CMake cache: `MLX_LM_WITH_REDLINE:BOOL=OFF`.

---

## Honesty

- **No gen t/s** reported for Redline (session is no-op).  
- Smoke tokens/s in stderr (if any) are **product eager** only — do not attribute to Redline.  
- E1 1.91× / E2 1.59× remain microbench floors only.

---

## Next (P1)

Tiny fixed-kernel retained AQL replay (toy / JIT HSACO from E3) vs HIP wall on gfx1150 — out-of-process harness OK; still no product-default-on.
