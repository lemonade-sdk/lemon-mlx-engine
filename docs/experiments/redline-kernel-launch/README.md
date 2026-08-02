# exp/redline-kernel-launch

**Sibling of** `exp/mtp-t1-lmhead-graph` · **Parent** `fix/mtp-stream-p0` @ `875a39d`

Research how **[Redline](https://github.com/warpfront/redline)** (user URL typo `pwilikin` → actual **`pwilkin/redline`** fork of **warpfront/redline**) can speed **kernel dispatch / launch floor** for ROCm decode — complementary to lm_head two-stage work.

→ **Start here:** [`RESEARCH.md`](RESEARCH.md)

### Status
- **E0 BUILD_OK** on host ROCm Core **7.13.0** / gfx1150 (warpfront build + floor HSACO). See [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md).  
- Redline README **≥ 7.14** is **not** a hard *compile* gate here; optional 7.14-only FFI / product cert still prefer upgrade ([`INSTALL_UPGRADE.md`](INSTALL_UPGRADE.md)).  
- **E1** floor µs/dispatch **not** measured yet. No product wire.  
- Decode HIP graphs remain **product OFF** (separate from Redline retained-PM4).
