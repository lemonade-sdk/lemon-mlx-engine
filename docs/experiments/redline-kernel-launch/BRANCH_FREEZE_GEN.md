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

| Item | Value |
|------|--------|
| **Redline fork (use this)** | https://github.com/antmikinka/redline |
| **Base** | https://github.com/pwilkin/redline (~8 commits behind warpfront; OK for `redline-capi`) |
| **PR-A branch** | `exp/hip-stream-bridge` (phase1 landed; install `.so` → lemon PR-B already wires it) |
| **Upstream reference only** | https://github.com/warpfront/redline |

```bash
cd /home/antmi/redline   # origin = antmikinka/redline
git checkout exp/hip-stream-bridge
cargo build -p redline-capi --release
cp -a target/release/libredline_dispatch.so /tmp/redline-warpfront-target/release/
# lemon-mlx: MLX_REDLINE_LIB=... expect log bridge=yes used
```

Product defaults stay **OFF**; optional OWN_GLUE for ownership without big gen loss.
