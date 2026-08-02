# exp/redline-kernel-launch

**Sibling of** `exp/mtp-t1-lmhead-graph` · **Parent** `fix/mtp-stream-p0` @ `875a39d`

Research how **[Redline](https://github.com/warpfront/redline)** (user URL typo `pwilikin` → actual **`pwilkin/redline`** fork of **warpfront/redline**) can speed **kernel dispatch / launch floor** for ROCm decode — complementary to lm_head two-stage work.

→ **Start here:** [`RESEARCH.md`](RESEARCH.md)

### Status
- **E0 BUILD_OK** on host ROCm Core **7.13.0** / gfx1150 (warpfront build + floor HSACO). See [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md).  
- **E1 AQL MEASURED** on gfx1150: ~**2.04** vs ~**1.07** µs/disp (BoundarySerialized **~1.91×**). PM4 example tail needs gfx12. See [`E1_FLOOR.md`](E1_FLOOR.md).  
- Redline README **≥ 7.14** is **not** a hard *compile* gate here ([`INSTALL_UPGRADE.md`](INSTALL_UPGRADE.md)).  
- **No gen t/s claim.** No product wire. Decode HIP graphs remain **product OFF**.
