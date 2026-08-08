# exp/redline-kernel-launch

**Sibling of** `exp/mtp-t1-lmhead-graph` · **Parent** `fix/mtp-stream-p0` @ `875a39d`

Research how **[Redline](https://github.com/warpfront/redline)** (user URL typo `pwilikin` → actual **`pwilkin/redline`** fork of **warpfront/redline**) can speed **kernel dispatch / launch floor** for ROCm decode — complementary to lm_head two-stage work.

→ **Start here:** [`ROADMAP.md`](ROADMAP.md) · [`RESEARCH.md`](RESEARCH.md)

### Status
- **E0 BUILD_OK** on host ROCm Core **7.13.0** / gfx1150. [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md).  
- **E1 AQL MEASURED:** ~**2.04** vs ~**1.07** µs/disp (GPU-span fence **~1.91×**). [`E1_FLOOR.md`](E1_FLOOR.md).  
- **E2 MEASURED:** multi no-op host wall — BoundarySerialized **~1.5–1.6×** vs HIP eager; hipGraph **≈** eager (no win). [`E2_MULTI.md`](E2_MULTI.md).  
- **E3 DONE:** qmm is AOT pointer-launch (not drop-in HSACO); JIT `.hsaco` exists on disk. [`E3_HSACO.md`](E3_HSACO.md).  
- **E4 DONE (design):** [`E4_DESIGN.md`](E4_DESIGN.md) — `MLX_REDLINE_DECODE` default **OFF**; AQL fixed small-op subgraph; qmm stays HIP.  
- **P0 GREEN:** env stub + CMake `MLX_LM_WITH_REDLINE=OFF` + gfx1150 chat smoke — [`P0_STUB.md`](P0_STUB.md).  
- **P1 GREEN:** floor CO load+replay n=2 BoundarySerialized; host µs only — [`P1_LOAD.md`](P1_LOAD.md).  
- **P2 GREEN:** N-sweep multi-run host wall — N=64 BS ~**82 µs** vs Sys ~**148 µs** (~**1.80×**) — [`P2_NSWEEP.md`](P2_NSWEEP.md).  
- **P2b GREEN:** engine dlopen session READY (abi smoke; residual gpu_new) — [`P2_INIT.md`](P2_INIT.md).  
- **P3 DESIGN PASS:** graph_decode kernarg-patch integration — [`P3_GRAPH_DECODE.md`](P3_GRAPH_DECODE.md).  
- **Stop A met** (P0+P1+P3 doc). Optional P4. No gen t/s claim. No product default ON. Decode HIP graphs remain **product OFF**.
