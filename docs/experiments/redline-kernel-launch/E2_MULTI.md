# E2 — Toy multi-kernel retained AQL vs HIP host wall (gfx1150)

**Date:** 2026-08-02  
**Branch:** `exp/redline-kernel-launch`  
**Host:** gfx**1150** · ROCm Core **7.13.0** · HIP **7.13.99004**  
**Verdict:** **MEASURED** — retained AQL **BoundarySerialized** host wall is faster than **HIP eager** N×launch for no-op chains; **hipGraph ≈ eager** (no win) on this micro shape.

**Not claimed:** gen t/s, 35B MoE decode, product HIP-graph enablement, PM4 IB.

---

## Design

| Arm | What | Timing |
|-----|------|--------|
| **HIP_eager** | N× `floor_k<<<1,1>>>` on a stream + `hipStreamSynchronize` | host wall (median of M) |
| **HIP_graph** | Capture N launches → `hipGraphLaunch` + sync | host wall |
| **AQL SystemEveryDispatch** | Retained `SingleQueueBatchGraph` of N no-ops, system fence every dispatch | host wall + GPU span |
| **AQL BoundarySerialized** | Same retained batch, Redline decode-safe fence policy | host wall + GPU span |
| **AQL BoundaryIndependent** | Aggressive independent policy (no-op only correctness) | host wall + GPU span |

Same kernel body class as E1 (`floor_k` no-op). HIP arms compile the kernel in-process; AQL arms load `floor_kernel-gfx1150.co`.

**Parameters:** M=100 timed, warmup=20; N ∈ {64, 256}.

---

## Harness

| File | Role |
|------|------|
| [`harness/e2_hip_multi_wall.hip`](harness/e2_hip_multi_wall.hip) | HIP eager + hipGraph host wall |
| [`harness/e2_aql_host_wall.rs`](harness/e2_aql_host_wall.rs) | Redline AQL host + GPU span (source of record; also built as example under local warpfront clone) |
| [`logs/e2-multi-kernel-wall-20260802-143256.log`](logs/e2-multi-kernel-wall-20260802-143256.log) | Full run log |

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
hipcc --offload-arch=gfx1150 harness/e2_hip_multi_wall.hip -O2 -o /tmp/e2_hip_multi_wall
# AQL example built via local /tmp/redline-warpfront + CARGO_TARGET_DIR
export REDLINE_FLOOR_HSACO=/tmp/redline-warpfront-hsaco/floor_kernel-gfx1150.co
export REDLINE_FLOOR_N=64 REDLINE_FLOOR_M=100 REDLINE_FLOOR_WARMUP=20
```

---

## Results (this GPU, this log)

### N=64

| Arm | median host us/replay | us/disp (host) | notes |
|-----|----------------------:|---------------:|-------|
| HIP_eager | **119.605** | 1.8688 | baseline “HIP wall” |
| HIP_graph | 120.120 | 1.8769 | **~1.0× vs eager** (no graph win) |
| AQL SystemEveryDispatch | 148.359 | 2.3181 | retained but heavy fences; **slower than HIP eager** |
| AQL **BoundarySerialized** | **75.117** | **1.1737** | **1.59× vs HIP_eager** host wall |
| AQL BoundaryIndependent | 73.418 | 1.1472 | 1.63× vs HIP_eager |

GPU-span (AQL only, for cross-check with E1): SystemEvery **131.0** us (2.05 us/d); BoundarySerialized **63.7** us (1.00 us/d).

### N=256

| Arm | median host us/replay | us/disp (host) | notes |
|-----|----------------------:|---------------:|-------|
| HIP_eager | **441.254** | 1.7236 | |
| HIP_graph | 452.414 | 1.7672 | still no graph win |
| AQL SystemEveryDispatch | 562.391 | 2.1968 | again slower than HIP eager |
| AQL **BoundarySerialized** | **289.278** | **1.1300** | **1.53× vs HIP_eager** |
| AQL BoundaryIndependent | 261.822 | 1.0227 | 1.69× vs HIP_eager |

---

## Honest reading

1. **E2 pass bar met:** measurable host-wall delta on gfx1150 for multi-kernel no-op chains.  
2. **Win driver is fence policy (BoundarySerialized), not “graphs exist.”** Retained AQL with **SystemEveryDispatch** was **worse** than HIP eager host wall.  
3. **hipGraph did not beat eager HIP** for this tiny no-op chain on this stack — consistent with our product stance that decode HIP graphs are not free wins.  
4. **Do not equate** E2 host-wall ratios with E1 GPU-span fence ratios (~1.91×) or with gen t/s.  
5. Real MLX kernels (HSACO ownership, MoE data-dependent experts) remain **E3+**.

---

## Board

| Gate | Status |
|------|--------|
| Toy multi-kernel retained vs HIP wall | **MEASURED** |
| Fail log if broken | N/A (all arms EXIT 0) |
| Gen t/s | **not claimed** |
