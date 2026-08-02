# MASTER — redline-kernel-launch

| Field | Value |
|-------|--------|
| **Branch** | `exp/redline-kernel-launch` |
| **Parent** | `fix/mtp-stream-p0` @ `875a39d` |
| **Sibling** | `exp/mtp-t1-lmhead-graph` (same parent) |
| **Project** | Redline (warpfront upstream / pwilkin fork) |

## Board

| Item | Status |
|------|--------|
| Identity (pwilkin vs warpfront) | **DONE** — pwilkin fork; upstream warpfront |
| Architecture / integration map | **DONE** — RESEARCH + subagent docs |
| E0 build on host ROCm 7.13 | **BUILD_OK** — warpfront `b505a72`; log + HSACO; 7.14 not hard compile gate |
| E1 floor bench gfx1150 | **AQL MEASURED** — ~2.04 vs ~1.07 µs/disp (1.91× BoundarySerialized); PM4 example tail FAIL gfx12 |
| E2 toy multi-kernel | **MEASURED** — N=64 host BoundarySerialized **75µs** vs HIP_eager **120µs** (~1.59×); hipGraph ≈ eager |
| E3 MLX HSACO inventory | **PENDING** |
| E4 design hook | **PENDING** (E1 green → design allowed next) |
| Engine wire | **NOT STARTED** (design only) |

## Fire log

### 2026-08-02 — E2 multi-kernel HIP wall vs AQL

- **Primary E-step:** E2.  
- Clear Thought: sequentialthinking, scientificmethod (experiment), metacognitivemonitoring, collaborative critique.  
- Harness: [`harness/e2_hip_multi_wall.hip`](harness/e2_hip_multi_wall.hip) + [`harness/e2_aql_host_wall.rs`](harness/e2_aql_host_wall.rs).  
- **N=64 host wall (median M=100):** HIP_eager **119.6µs**; HIP_graph **120.1µs** (~no win); AQL BoundarySerialized **75.1µs** (**~1.59×** vs eager); AQL SystemEveryDispatch **148.4µs** (worse than eager).  
- **N=256:** HIP_eager **441µs**; BoundarySerialized **289µs** (**~1.53×**).  
- Evidence: [`logs/e2-multi-kernel-wall-20260802-143256.log`](logs/e2-multi-kernel-wall-20260802-143256.log), [`E2_MULTI.md`](E2_MULTI.md).  
- **Not claimed:** gen t/s; conflating E1 GPU-span 1.91× with E2 host wall.  
- **Next:** E3 MLX HSACO inventory **or** E4 design sketch (E1 unlocked).

### 2026-08-02 — E1 dispatch_floor gfx1150

- **Primary E-step:** E1.  
- Clear Thought: sequentialthinking, scientificmethod (hypothesis→analysis), metacognitivemonitoring, collaborative critique.  
- Ran `dispatch_floor` with `REDLINE_FLOOR_HSACO=floor_kernel-gfx1150.co`, N=64 M=200 warmup=20 on ROCm **7.13** / **gfx1150**.  
- **AQL fence spectrum (GPU-timed):** SystemEveryDispatch **2.0388** µs/disp; BoundarySerialized **1.0676** µs/disp (**1.910×**); BoundaryIndependent 0.9480 µs/disp (2.151×).  
- Evidence: [`logs/e1-dispatch-floor-gfx1150-20260802-142850.log`](logs/e1-dispatch-floor-gfx1150-20260802-142850.log), [`E1_FLOOR.md`](E1_FLOOR.md).  
- Process **EXIT 1** after table: PM4 IB `ArchitectureMismatch { required: "gfx12", actual: "gfx1150" }` (example hardcodes Gfx12; library has gfx11 path unused by example).  
- **Not claimed:** gen t/s, 35B win, PM4 IB numbers on 890M.  
- **Next:** E2 toy multi-kernel retained vs HIP wall.

### 2026-08-02 — E0 host build



- **Primary E-step:** E0.  
- Clear Thought: sequentialthinking, scientificmethod (observation), decisionframework, metacognitivemonitoring, collaborative critique.  
- Built **warpfront** Redline @ `b505a72` against TheRock **ROCm Core 7.13.0** / HIP **7.13.99004** / **gfx1150**.  
- `cargo build --release -p redline-dispatch -p redline-capi -p redline-hipgraph` → **Finished 8.45s EXIT 0**.  
- Evidence: [`logs/e0-build-warpfront-20260802-142519.log`](logs/e0-build-warpfront-20260802-142519.log), [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md).  
- Floor kernel: `hipcc --genco --offload-arch=gfx1150` → [`logs/floor_kernel-gfx1150.co`](logs/floor_kernel-gfx1150.co).  
- **Revision:** README “≥7.14” is **not** a hard *compile* blocker here; optional 7.14-only FFI may still fail at use. Upgrade notes: [`INSTALL_UPGRADE.md`](INSTALL_UPGRADE.md).  
- **Not claimed:** any dispatch µs, gen t/s, or engine integration.  
- **Next:** E1 `dispatch_floor` with `REDLINE_FLOOR_HSACO=…/floor_kernel-gfx1150.co`.

### 2026-08-02 — research branch open

- Clear Thought + explore subagents (ROCm dispatch, engine launch map).  
- Documented retained-PM4 vs HIP floor, C ABI path, gfx1150 caveats.  
- No code wire; no fake speed claims.
