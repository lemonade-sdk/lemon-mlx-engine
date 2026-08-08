# MASTER — redline-kernel-launch

| Field | Value |
|-------|--------|
| **Branch** | `exp/redline-kernel-launch` |
| **Parent** | `fix/mtp-stream-p0` @ `875a39d` |
| **Sibling** | `exp/mtp-t1-lmhead-graph` (same parent) |
| **Project** | Redline (warpfront upstream / pwilkin fork) |
| **Loop status** | **ACTIVE (post–Stop A)** — P5+P6 PASS; gen t/s A/B still **not** run; product default ON **forbidden** |

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
| **P2 N-sweep multi-run** | **GREEN** — N=2..64 × BS/Sys × 3 runs; N=64 BS **81.98 µs** vs Sys **147.7 µs** (~**1.80×**) ([`P2_NSWEEP.md`](P2_NSWEEP.md)) |
| **P2b engine session init** | **GREEN** — dlopen + abi + **`gpu_new=ok`** after RUNPATH/core-first RPATH fix ([`P2_INIT.md`](P2_INIT.md)) |
| **P3 graph_decode / kernarg doc** | **PASS (design)** — [`P3_GRAPH_DECODE.md`](P3_GRAPH_DECODE.md) + [`QUALITY_REVIEW_P3.md`](QUALITY_REVIEW_P3.md) |
| **P3 measured micro-op** | **PASS** — out-of-process AQL patch+replay; correctness 4160; host_median **8.796 µs** ([`P3_MICRO_OP.md`](P3_MICRO_OP.md)); **not** gen t/s |
| **P4 MoE multipath design** | **SKETCH** — [`P4_MOE_MULTIPATH.md`](P4_MOE_MULTIPATH.md) |
| **P5 in-process C-API micro** | **PASS** — chat session `micro=PASS` 2080/2080; host_total_us labeled NOT gen t/s ([`P5_INPROC_MICRO.md`](P5_INPROC_MICRO.md)) |
| **P6 graph_decode bind** | **PASS** — VRAM ptrs stable + **bake pos as PM4 acc** micro 2080/2080 ([`P6_GRAPH_DECODE_BIND.md`](P6_GRAPH_DECODE_BIND.md)) |
| Engine product wire / default ON | **FORBIDDEN until measured** gen A/B |
| Gen t/s A/B (same-build eager) | **NOT RUN** |

## Fire log

### 2026-08-08 — P6 measured product-buffer bake PASS

- **Primary:** Measure P6 bake: `graph_decode_pos` VRAM ptr as retained-PM4 `acc_k` accumulator (product buffer, not hipMalloc).  
- Clear Thought: sequentialthinking, decisionframework (A1 bake vs A2/B/E), metacognitivemonitoring, scientificmethod (H-p6-gd-bind).  
- **Code (on tip):** `graph_decode_device_data_ptr`; `try_micro_op` gd_bind+bake; `maybe_probe_redline_graph_decode_bind`.  
- **Smoke:** off 0×; skip; **`gd_bind=PASS gd_post=stable micro=PASS observed=2080 expected=2080`**; xor fail-closed.  
- **Logs:** `logs/p6-{off,on-skip,on-micro,xor}-20260808-113412.err`.  
- **Doc:** [`P6_GRAPH_DECODE_BIND.md`](P6_GRAPH_DECODE_BIND.md) + [`QUALITY_REVIEW_P6.md`](QUALITY_REVIEW_P6.md) **PASS**.  
- **Not claimed:** gen t/s; default ON; call_fn replace.  
- **Next:** L=1 sidecar without call_fn replace; gen A/B only after product-path ownership.

### 2026-08-08 — P6 graph_decode bind PASS (probe land)

- **Primary:** Stable `graph_decode_input` / `graph_decode_pos` device buffer pointers after in-place mutate.  
- **Code:** `maybe_probe_redline_graph_decode_bind()`; wired from `generate.cpp` L=1/`next` + `chat` post-load.  
- **Smoke:** off silent; on **`gd_bind PASS stable=1`**; xor no gd_bind — `logs/p6-*-20260808-113247.err`.  
- **Follow-up same window:** product-buffer bake measure (fire entry above).

### 2026-08-08 — P5 in-process micro-op PASS (post–Stop A)

- **Primary:** Product-adjacent engine session micro — retained PM4 load + kernarg patch + replay correctness behind `MLX_REDLINE_DECODE=1` + opt-in `MLX_REDLINE_HSACO`.  
- Clear Thought: sequentialthinking, decisionframework (A vs B/E), metacognitivemonitoring, scientificmethod (H-p5-inproc-micro).  
- **Code:** `src/common/redline_decode_session.cpp` + header; `try_micro_op` (dlsym C-API; HIP owns accumulator).  
- **Precondition:** standalone `decode_kernargs` gfx1150 **PASS** 2080/2080.  
- **Engine smoke:** off 0×; on `micro=skip`; on+HSACO **`micro=PASS observed=2080 expected=2080`**; xor fail-closed.  
- **Logs:** `logs/p5-{off,on-skip,on-micro,xor}-20260808-112653.err`.  
- **Doc:** [`P5_INPROC_MICRO.md`](P5_INPROC_MICRO.md) + [`QUALITY_REVIEW_P5.md`](QUALITY_REVIEW_P5.md) **PASS**.  
- **Not claimed:** gen t/s; product default ON; `call_fn` / qmm replace; TokenIterator partial wire.  
- **Next:** graph_decode_* bind or real small engine-owned op; gen A/B only after product-path change.

### 2026-08-07 — P3 micro-op PASS + STOP A (measured clause)

- **Primary P-step:** P3 fixed micro-op — retained AQL + `patch_kernarg_u32` + correctness.  
- Clear Thought: sequentialthinking, decisionframework (P3 measure vs empty/P4), metacognitivemonitoring.  
- **Code/harness:** [`harness/p3_kernarg_patch.rs`](harness/p3_kernarg_patch.rs); CO [`logs/acc_kernel-gfx1150.co`](logs/acc_kernel-gfx1150.co).  
- **Measure:** n=2, T=64, observed=expected=4160; host_median_us **8.796** (NOT gen t/s).  
- **Log:** [`logs/p3-kernarg-patch-20260807-221119.log`](logs/p3-kernarg-patch-20260807-221119.log).  
- **Doc:** [`P3_MICRO_OP.md`](P3_MICRO_OP.md) + [`QUALITY_REVIEW_P3_MICRO.md`](QUALITY_REVIEW_P3_MICRO.md) **PASS**.  
- **Stop A (measured):** P0+P1 green + P3 design PASS + P3 micro-op PASS + P2b `gpu_new=ok` → **met** → **scheduler_delete** `019fdfc642e5`.  
- **Not claimed:** gen t/s; product default ON; TokenIterator product wire.

### 2026-08-07 — P3 doc hardened + stop A (continuous loop)

- **Primary P-step:** P3 integration doc (file:line citations for `graph_decode.cpp` / pure-path `generate.cpp`).  
- Clear Thought: sequentialthinking, decisionframework (P3 vs P2b/P4), metacognitivemonitoring.  
- **Evidence:** [`P3_GRAPH_DECODE.md`](P3_GRAPH_DECODE.md) + [`QUALITY_REVIEW_P3.md`](QUALITY_REVIEW_P3.md) **PASS**.  
- **Stop A:** P0+P1 gfx1150 logs + P3 doc + quality PASS → **met** → **scheduler_delete** `019fdfb3d185`.  
- **Not claimed:** gen t/s; product default ON; in-process `rl_gpu_new` full bind (residual).  
- Optional P4 sketch already on branch ([`P4_MOE_MULTIPATH.md`](P4_MOE_MULTIPATH.md)).

### 2026-08-08 — P2b residual CLOSED (`gpu_new=ok`)

- **Root cause:** chat **DT_RPATH** miniforge-first → Redline bound conda HSA missing `hsa_amd_counted_queue_acquire`.  
- **Fix:** CMake `--enable-new-dtags` + rpath `/opt/rocm/core/lib` first on chat/server.  
- **Smoke:** `session READY ... gpu_new=ok` — [`logs/p2b-rpathfix-20260807-220731.err`](logs/p2b-rpathfix-20260807-220731.err).  
- **Not claimed:** gen t/s; product default ON.

### 2026-08-08 — P4 MoE multipath sketch

- Optional post-stop-A design: multipath vs re-record vs HIP-only MoE.  
- Evidence: [`P4_MOE_MULTIPATH.md`](P4_MOE_MULTIPATH.md). No code; no gen t/s.

### 2026-08-08 — P2b session init GREEN + P3 design PASS

- **Primary:** P2b engine dlopen session + P3 graph_decode design doc (stop A).  
- **Code:** `redline_decode_session.{h,cpp}`; `generate.cpp` + early `chat.cpp` probe; CMake `CMAKE_DL_LIBS`.  
- **Smoke:** off silent; on `session READY abi=1 gpu_new=null...`; XOR fail-closed — `logs/p2-*-20260807-215745.err`.  
- **Residual:** in-process `rl_gpu_new` null when MLX-linked; standalone C smoke OK.  
- **P3:** kernarg-patch design + quality PASS (no product wire).  
- **Stop A:** P0+P1 green + P3 doc quality PASS — **met**. Optional: P4 MoE multipath.

### 2026-08-07 — P2 N-sweep GREEN (continuous loop)

- **Primary P-step:** P2 multi-run N-sweep (loop sequence).  
- Clear Thought: sequentialthinking, decisionframework (P2 vs P3-first), scientificmethod.  
- **Harness:** [`harness/p2_nsweep.sh`](harness/p2_nsweep.sh).  
- **Measure:** BoundarySerialized vs SystemEveryDispatch, N∈{2,4,8,16,32,64}, 3 process runs, host µs only.  
- **Headline:** N=64 BS med_of_med **81.979 µs** vs Sys **147.743 µs** (**1.80×**); ratio rises with N.  
- **Log:** [`logs/p2-nsweep-20260807-215606.log`](logs/p2-nsweep-20260807-215606.log).  
- **Doc:** [`P2_NSWEEP.md`](P2_NSWEEP.md).  
- **Not claimed:** gen t/s; HIP eager re-bench; engine session (P2b).  
- **Next:** P3 `graph_decode` integration doc → stop A with quality PASS.

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
