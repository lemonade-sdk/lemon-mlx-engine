# Quality review — E1 floor numbers

**Date:** 2026-08-02  
**Method:** Clear Thought collaborative (Quality + ROCm) + metacognitive + log cross-check.

## Verdict: **PASS** (with required caveats)

| Claim | Allowed? | Notes |
|-------|----------|--------|
| AQL us/disp table on gfx1150 | **YES** | Log shows full 5-policy table before error |
| BoundarySerialized **1.910×** vs SystemEveryDispatch | **YES** | From same run; quote log path |
| “dispatch_floor fully green EXIT 0” | **NO** | EXIT 1 after PM4 gfx12 mismatch |
| “PM4 retained IB measured on 890M” | **NO** | ArchitectureMismatch |
| Gen t/s / 35B | **NO** | Hard ban; no-op only |
| Transfer of historical 1.8× as this result | **NO** | Say “consistent in class,” cite **this** log |

## Required wording (applied in `E1_FLOOR.md` / MASTER)

- **E1 status:** `AQL MEASURED` not unqualified green.  
- Always attach: N=64, M=200, warmup=20, no-op `floor_k`, ROCm 7.13, gfx1150.  
- Explain EXIT 1: example hardcodes `Gfx12Pm4CommandBuffer`; AQL table still valid.

## Residual risks

1. Single-run; APU concurrent graphics load not controlled.  
2. No HIP *launch* wall comparison of independent hipLaunchKernel chain (E2).  
3. Decode-shaped multi-kernel + real HSACO not measured.

**Sign-off:** Safe to publish as floor microbench evidence on this GPU.
