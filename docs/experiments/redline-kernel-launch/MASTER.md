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
| E1 floor bench gfx1150 | **PENDING** — CO ready; no µs/dispatch yet |
| E2 toy multi-kernel | **PENDING** |
| E3 MLX HSACO inventory | **PENDING** |
| E4 design hook | **PENDING** |
| Engine wire | **NOT STARTED** (design only) |

## Fire log

### 2026-08-02 — E0 host build (this fire)

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
