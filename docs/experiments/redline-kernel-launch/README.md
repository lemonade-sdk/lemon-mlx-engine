# exp/redline-kernel-launch

**Sibling of** `exp/mtp-t1-lmhead-graph` · **Parent** `fix/mtp-stream-p0` @ `875a39d`

Research how **[Redline](https://github.com/warpfront/redline)** (user URL typo `pwilikin` → actual **`pwilkin/redline`** fork of **warpfront/redline**) can speed **kernel dispatch / launch floor** for ROCm decode — complementary to lm_head two-stage work.

→ **Start here:** [`RESEARCH.md`](RESEARCH.md)

### Status
- **E0 BUILD_OK** on host ROCm Core **7.13.0** / gfx1150. [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md).  
- **E1 AQL MEASURED:** ~**2.04** vs ~**1.07** µs/disp (GPU-span fence **~1.91×**). [`E1_FLOOR.md`](E1_FLOOR.md).  
- **E2 MEASURED:** multi no-op host wall — BoundarySerialized **~1.5–1.6×** vs HIP eager; hipGraph **≈** eager (no win). [`E2_MULTI.md`](E2_MULTI.md).  
- **E3 DONE:** qmm is AOT pointer-launch (not drop-in HSACO); JIT `.hsaco` exists on disk. [`E3_HSACO.md`](E3_HSACO.md).  
- **No gen t/s claim.** No product wire. Decode HIP graphs remain **product OFF**.
