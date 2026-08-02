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
| E0 build on host ROCm 7.13 | **PENDING** |
| E1 floor bench gfx1150 | **PENDING** (may need ROCm 7.14) |
| Engine wire | **NOT STARTED** (design only) |

## Fire log

### 2026-08-02 — research branch open

- Clear Thought + explore subagents (ROCm dispatch, engine launch map).  
- Documented retained-PM4 vs HIP floor, C ABI path, gfx1150 caveats.  
- No code wire; no fake speed claims.
