# F1 results summary

**Date:** 2026-08-01  
**task_id:** 019fbfad5fe8  
**tip:** 09940a8 (+ local mlx opt-in patch; engine banner commit)  
**PASS/FAIL:** **FAIL** (≥10% pp/s bar not met)  
**schedule stop?** **n** (fire 1/3 of “not viable” ladder; path runs but small win)

## Idea

Restore env-gated `use_hip_graphs()` for prefill (`MLX_HIP_GRAPH_PREFILL=1` + `MLX_GRAPH_PREFILL_REPLAY=1`) and measure multi-chunk prefill vs eager on 35B gfx1150.

## Baseline vs experiment (required)

| | prefill_s | pp/s |
|--|-----------|------|
| **Baseline** (eager, mean r1/r2) | ~14.93 | **~110.7** |
| **Experiment** (graph, mean r1/r2) | ~14.54 | **~113.7** |
| **Δ** | −2.6% time | **+~2.7% pp/s** |
| Long re-prefill turn2 | 31.00 → 29.02 | 109.4 → 116.9 (**+6.8%**) |

prompt_tokens single-turn: **1653** · step **128** · gen short ~29–30 t/s either way · graph peak mem **+2.5 GB**

## Code paths

- **mlx (local):** `device.cpp` `use_hip_graphs` env opt-in — patch file only, not upstream default  
- **engine:** `generate.cpp` prefill-graph env banner when profiling/flags set  
