# Loop10 — pin mlx SHA + GDN same-dtype cast skip

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**Supervisors:** Clear Thought collaborative + decision; CI babysit subagent  

## Code

| Change | Why |
|--------|-----|
| `CMakeLists.txt` FetchContent `GIT_TAG` → `0dadb703d77301af29405cf7e12627efb88a6d0f` | Reproducible mlx pin matching exclusive/dual-load evidence packs |
| `gdn_fused_decode`: skip `astype(…, f32)` when already f32 | Residual mid-forward copy hygiene after load-time materialize |

**Not done:** re-fetch rebuild of mlx tree this loop (pin takes effect on next clean FetchContent / configure that updates dep). Local build still links existing `_deps/mlx` @ same SHA already.

## Live spot-check (exclusive server already up)

| Gate | Result |
|------|--------|
| S1 live `2+2` → `4` | **PASS** (`raw/S1-live.json`) |

No dual-load; no MTP; no pure-graph.

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Engine CMake pins verified mlx SHA | **Yes** |
| Binary automatically rebuilt mlx from git this loop | **No** — pin is for future/configure; tip already 0dadb703 |
| Full loop6 matrix re-run | **No** (not required) |
