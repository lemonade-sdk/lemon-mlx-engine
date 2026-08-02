# MASTER — redline-kernel-launch

| Field | Value |
|-------|--------|
| **Branch** | `exp/redline-kernel-launch` |
| **Parent** | `fix/mtp-stream-p0` @ `875a39d` |
| **Sibling** | `exp/mtp-t1-lmhead-graph` (same parent) |
| **Project** | Redline (warpfront upstream / pwilkin fork) |
| **Loop status** | **STOPPED** — E0–E2 green + E4 design landed (stop rule 1) |

## Board

| Item | Status |
|------|--------|
| Identity (pwilkin vs warpfront) | **DONE** — pwilkin fork; upstream warpfront |
| Architecture / integration map | **DONE** — RESEARCH + subagent docs |
| E0 build on host ROCm 7.13 | **BUILD_OK** — warpfront `b505a72`; log + HSACO; 7.14 not hard compile gate |
| E1 floor bench gfx1150 | **AQL MEASURED** — ~2.04 vs ~1.07 µs/disp (1.91× BoundarySerialized); PM4 example tail FAIL gfx12 |
| E2 toy multi-kernel | **MEASURED** — N=64 host BoundarySerialized **75µs** vs HIP_eager **120µs** (~1.59×); hipGraph ≈ eager |
| E3 MLX HSACO inventory | **DONE** — qmm AOT **not** drop-in; JIT `.hsaco` on disk; see [`E3_HSACO.md`](E3_HSACO.md) |
| E4 design hook | **DONE** — [`E4_DESIGN.md`](E4_DESIGN.md) (`MLX_REDLINE_DECODE` default OFF) |
| Engine wire | **NOT STARTED** (design only; future P0+) |

## Fire log

### 2026-08-02 — E4 design + STOP

- **Primary E-step:** E4.  
- Clear Thought: sequentialthinking, decisionframework (arch A vs B/C/D), metacognitivemonitoring, collaborative critique.  
- Design: opt-in `MLX_REDLINE_DECODE=1` → redline-capi / AQL **BoundarySerialized** fixed small-op subgraph; **qmm stays HIP**; no HIP-graph product path; phases P0–P4; kill criteria vs eager only.  
- Evidence: [`E4_DESIGN.md`](E4_DESIGN.md).  
- **Stop rule (1):** E0–E2 gfx1150 evidence + E4 design → **scheduler_delete**.  
- **Not shipped:** product stub in binary; gen t/s claims.

### 2026-08-02 — E3 MLX HSACO inventory

- **Primary E-step:** E3 (hot op = quantized matmul / qmm).  
- **AOT qmm:** pointer `hipLaunchKernel` — drop-in Redline load **NOT FEASIBLE**.  
- **JIT:** `/tmp/mlx/0.32.0/hsaco/gfx1150/` format-feasible.  
- Evidence: [`E3_HSACO.md`](E3_HSACO.md).

### 2026-08-02 — E2 multi-kernel HIP wall vs AQL

- **N=64 host:** HIP_eager **119.6µs**; BoundarySerialized **75.1µs** (~1.59×); hipGraph ≈ eager.  
- Evidence: [`logs/e2-multi-kernel-wall-20260802-143256.log`](logs/e2-multi-kernel-wall-20260802-143256.log).

### 2026-08-02 — E1 dispatch_floor gfx1150

- AQL fence spectrum measured; PM4 example tail gfx12 mismatch EXIT 1.  
- Evidence: [`logs/e1-dispatch-floor-gfx1150-20260802-142850.log`](logs/e1-dispatch-floor-gfx1150-20260802-142850.log).

### 2026-08-02 — E0 host build

- Redline warpfront release build OK on ROCm 7.13 / gfx1150.  
- Evidence: [`logs/e0-build-warpfront-20260802-142519.log`](logs/e0-build-warpfront-20260802-142519.log).

### 2026-08-02 — research branch open

- Architecture docs + identity (warpfront / pwilkin).
