# Branch freeze — gen-path ownership on `exp/redline-kernel-launch`

**Date:** 2026-08-08  
**Decision:** **No further product-ownership work on this branch is expected to fix OWN_RMSNORM gen t/s** without **Redline PR-A** (HIP stream/event bridge).

## Certainty

| Work on lemon-mlx only | Gen effect |
|------------------------|------------|
| More PRE/POST knobs | Exhausted (P12b–d) |
| Own strided RMS / CustomKernel still dual-queue | Likely **more** PRE tax |
| Sidecar / all-flags | Already **slower** |
| Wire stream-bridge symbols | **Blocked** until `libredline_dispatch` exports them |

PRE tax is **host join of product HIP producers** before a **different** Redline queue. That ordering cannot be expressed without a Redline (and possibly ROCm) API.

## What this branch already completed

- OWN_GLUE + OWN_RMSNORM (research, default OFF)  
- Measure B0/B1/B2; no ≥2% win  
- P13 contract: [`P13_STREAM_BRIDGE_PR.md`](P13_STREAM_BRIDGE_PR.md)

## Next venue

**Redline repo** branch for **PR-A** → install new `.so` → lemon-mlx **PR-B** (small wire) → gen A/B.

Until then: product defaults stay **OFF**; optional OWN_GLUE for ownership without big gen loss.
