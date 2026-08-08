# Quality review — P2b engine session init

**Date:** 2026-08-08  
**Scope:** dlopen Redline C-API session + chat smoke  
**Verdict:** **PASS** (with residual)

| Check | Result |
|-------|--------|
| Default OFF | **PASS** — 0× `[redline]` when env unset |
| Exact `"1"` only | **PASS** — session path gated |
| One-shot banner | **PASS** — count=1 on on-log |
| No forward change | **PASS** — still product eager |
| XOR pure-graph | **PASS** — fail-closed |
| No HIP-graph enable | **PASS** |
| CMake default OFF | **PASS** — dlopen, not hard link |
| Build green | **PASS** — `chat` exit 0 |
| No gen t/s claim | **PASS** |
| Honest residual | **PASS** — documents `gpu_new=null` in MLX-linked process |
| Naming | **PASS** — P2b (session) ≠ P2 N-sweep |

## Residual

- In-process `rl_gpu_new(0)` null after linking MLX/HIP even pre-load; standalone C smoke OK.  
- P3 must not assume in-process ROCr bind without further isolation work.

**Reviewer notes:** ship P2b as init smoke; do not product-default-on; P3 design doc separate.
