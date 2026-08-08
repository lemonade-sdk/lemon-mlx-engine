# P2 — N-sweep / multi-run retained AQL host wall (gfx1150)

**Date:** 2026-08-07  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **GREEN** (measure)  
**Loop mapping:** continuous-loop **P2** = multi-seed / N-sweep (this doc).  
E4 “engine session init” sketch lives in [`P2_PLAN.md`](P2_PLAN.md) and is **not** claimed done here.

---

## Goal

Sweep dispatch count **N** and fence **policy** on the best retained path from P1 (floor CO + `SingleQueueBatchGraph`), with **independent process runs** as multi-run variance (“seeds”). Record **host µs/replay only** — **not** model gen t/s.

---

## Method

| Item | Value |
|------|--------|
| Harness | [`harness/p2_nsweep.sh`](harness/p2_nsweep.sh) → binary `p1_load_hsaco` |
| CO | [`logs/floor_kernel-gfx1150.co`](logs/floor_kernel-gfx1150.co) `floor_k.kd` |
| N | 2, 4, 8, 16, 32, 64 |
| Policies | BoundarySerialized, SystemEveryDispatch |
| Runs | 3 independent processes per (N, policy) |
| Iters / warmup | 40 / 10 |
| Device | gfx1150 |
| Log | [`logs/p2-nsweep-20260807-215606.log`](logs/p2-nsweep-20260807-215606.log) |

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
REDLINE_P1_HSACO=docs/experiments/redline-kernel-launch/logs/floor_kernel-gfx1150.co \
  ./docs/experiments/redline-kernel-launch/harness/p2_nsweep.sh
```

---

## Results (median-of-medians host µs/replay)

| N | BoundarySerialized | SystemEveryDispatch | BS vs Sys |
|---|-------------------:|--------------------:|----------:|
| 2 | 8.722 | 9.803 | **1.12×** |
| 4 | 10.660 | 14.001 | **1.31×** |
| 8 | 14.562 | 22.377 | **1.54×** |
| 16 | 22.422 | 43.657 | **1.95×** |
| 32 | 43.682 | 80.366 | **1.84×** |
| 64 | 81.979 | 147.743 | **1.80×** |

**us_per_dispatch (BoundarySerialized):** falls from ~4.36 (N=2) toward ~1.28 (N=64) — retained batch amortizes.

**Variance:** min/max of the 3 process medians are tight (typically <2% relative) except occasional dips (e.g. N=32 BS min 38.2 vs med 43.7).

**Cross-check E2:** E2 N=64 BoundarySerialized host wall ~**75 µs** vs HIP eager ~**120 µs**. This P2 N=64 BS ~**82 µs** (same order; different harness/iters/day). P2 **does not** re-measure HIP eager this fire.

---

## Honesty bans

| Claim | Status |
|-------|--------|
| Model gen t/s | **None** |
| Product enable | **None** |
| E1 1.91× as gen win | **Forbidden** — fence µs only |
| Engine session / dlopen | **Not this doc** — see P2_PLAN |

**Gen-adjacent:** product `MLX_REDLINE_DECODE=1` still P0 no-op banner only (no session).

---

## Conclusion

- Best retained path for v0 decode-shaped chains remains **BoundarySerialized**.  
- Benefit vs system-every grows with N (~1.1× at N=2 → ~1.8× at N=64).  
- Multi-run stable enough for research decisions.  
- **Next (loop P3):** document `graph_decode_*` stable buffers + kernarg patch integration (stop A).
