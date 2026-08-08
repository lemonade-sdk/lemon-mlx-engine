# Quality review — P0 stub

**Date:** 2026-08-07  
**Scope:** `MLX_REDLINE_DECODE` stub + CMake notes + gfx1150 smoke  
**Verdict:** **PASS**

| Check | Result |
|-------|--------|
| Default OFF | **PASS** — unset / `true` → 0× `[redline]` in logs |
| Exact `"1"` only | **PASS** — banner only for `MLX_REDLINE_DECODE=1` |
| Banner once | **PASS** — count=1 in on-log |
| No path change | **PASS** — still product eager; message says not implemented |
| XOR pure-graph | **PASS** — fail-closed banner; pure disabled when both set |
| No HIP-graph enable | **PASS** — does not set USE_HIP_GRAPHS / HIP_GRAPH_DECODE |
| CMake default OFF | **PASS** — `MLX_LM_WITH_REDLINE:BOOL=OFF` |
| Build green | **PASS** — `chat` link exit 0 |
| No gen t/s claim | **PASS** — P0_STUB + MASTER honesty |
| Logs present | **PASS** — `logs/p0-{off,on,xor-pure,true}-20260807-215209.err` |

## Residual / next

- ~~P1 not green: single-dispatch batch rejected (`InvalidBatchShape`).~~ **Resolved** — P1 PASS with n≥2 ([`P1_LOAD.md`](P1_LOAD.md)).  
- Banner uses non-atomic `static bool` (matches other generate.cpp env logs).  
- XOR only documents pure-graph env, not `MLX_USE_HIP_GRAPHS` / `MLX_HIP_GRAPH_DECODE` in the pure_enabled gate (banner comment still forbids coupling).

**Reviewer notes:** ship P0; do not product-default-on; P1/P2 measured later — see P1_LOAD / P2_NSWEEP.
