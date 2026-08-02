# exp/redline-kernel-launch

**Sibling of** `exp/mtp-t1-lmhead-graph` · **Parent** `fix/mtp-stream-p0` @ `875a39d`

Research how **[Redline](https://github.com/warpfront/redline)** (user URL typo `pwilikin` → actual **`pwilkin/redline`** fork of **warpfront/redline**) can speed **kernel dispatch / launch floor** for ROCm decode — complementary to lm_head two-stage work.

→ **Start here:** [`RESEARCH.md`](RESEARCH.md)

### Status
- Docs + subagent maps only (no product wire yet).  
- Local ROCm **7.13** vs Redline **≥ 7.14** is a first-class gate.  
- Decode HIP graphs remain **product OFF** (separate from Redline retained-PM4).
