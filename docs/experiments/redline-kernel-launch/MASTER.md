# MASTER — redline-kernel-launch

| Field | Value |
|-------|--------|
| **Branch** | `exp/redline-kernel-launch` |
| **Parent** | `fix/mtp-stream-p0` @ `875a39d` |
| **Sibling** | `exp/mtp-t1-lmhead-graph` (same parent) |
| **Project** | Redline (warpfront upstream / pwilkin fork) |
| **Loop status** | **IMPLEMENTING P0–P4** — scheduler `019fdfb3d185` · **P0 GREEN · P1 GREEN** (P3 doc still open for stop A) |

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
| **P0 env stub** | **GREEN** — [`P0_STUB.md`](P0_STUB.md); code + CMake OFF + gfx1150 chat smoke logs |
| **P1 AQL HSACO load** | **GREEN** — n=2 floor CO load+replay; host_median **8.455 µs** ([`P1_LOAD.md`](P1_LOAD.md)); **not** gen t/s |
| **P2 session init** | **NOT STARTED** |
| **P3 micro-op / graph_decode doc** | **NOT STARTED** (needed for stop A with quality PASS) |
| **P4 MoE multipath design** | **NOT STARTED** |
| Engine product wire / default ON | **FORBIDDEN until measured** |

## Fire log

### 2026-08-07 — P1 load+replay GREEN (same fire window)

- **Secondary:** land P1 measure after dual-dispatch fix (`REDLINE_P1_N` default 2).  
- **Log:** [`logs/p1-load-hsaco-20260807-215318.log`](logs/p1-load-hsaco-20260807-215318.log) — `P1_OK` n=2 host_median_us=8.455.  
- **Doc:** [`P1_LOAD.md`](P1_LOAD.md).  
- **Not claimed:** model gen t/s; MLX JIT HSACO; product wire.  
- **Next:** P3 `graph_decode` kernarg-patch integration doc (and/or P2 session) toward stop A.

### 2026-08-07 — P0 smoke GREEN (continuous loop)

- **Primary P-step:** P0 complete.  
- Clear Thought: sequentialthinking, decisionframework (P0 first), metacognitivemonitoring.  
- **Code:** `src/common/generate.cpp` stub + `CMakeLists.txt` `MLX_LM_WITH_REDLINE=OFF` notes.  
- **Build:** `cmake --build build --target chat` exit 0.  
- **Smoke (gfx1150, Qwen3.5-0.8B-4bit):**  
  - off → 0× `[redline]` [`logs/p0-off-20260807-215209.err`](logs/p0-off-20260807-215209.err)  
  - `=1` → 1× not-implemented banner [`logs/p0-on-20260807-215209.err`](logs/p0-on-20260807-215209.err)  
  - XOR pure → fail-closed banner [`logs/p0-xor-pure-20260807-215209.err`](logs/p0-xor-pure-20260807-215209.err)  
  - `=true` → silent [`logs/p0-true-20260807-215209.err`](logs/p0-true-20260807-215209.err)  
- **Evidence:** [`P0_STUB.md`](P0_STUB.md).  
- **Not claimed:** gen t/s; product enable; P1 green.  
- **Next fire:** P1 dual-dispatch retained AQL (fix single-dispatch InvalidBatchShape).

### 2026-08-08 — P0 implement + P1 scaffold (continuous loop)

- **Primary P-step:** P0 code + P1 harness scaffold.  
- **Code:** `generate.cpp` env parse + banner; `harness/p1_load_hsaco.rs`.  
- **P1 attempt:** Executable load OK; FAIL `InvalidBatchShape` (≥2 dispatches required) — see p1 log.  
- **Loop:** continue until stop A/B/C.

### 2026-08-02 — E4 design + STOP (design loop closed)

- **Primary E-step:** E4.  
- Clear Thought: sequentialthinking, decisionframework (arch A vs B/C/D), metacognitivemonitoring, collaborative critique.  
- Design: opt-in `MLX_REDLINE_DECODE=1` → redline-capi / AQL **BoundarySerialized** fixed small-op subgraph; **qmm stays HIP**; no HIP-graph product path; phases P0–P4; kill criteria vs eager only.  
- Evidence: [`E4_DESIGN.md`](E4_DESIGN.md).  
- **Stop rule (1):** E0–E2 gfx1150 evidence + E4 design → **scheduler_delete** (design loop only).  
- **Not shipped (then):** product stub in binary; gen t/s claims.

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
