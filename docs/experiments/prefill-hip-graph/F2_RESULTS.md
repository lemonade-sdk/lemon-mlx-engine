# F2 results summary

**Date:** 2026-08-01  
**task_id:** 019fbfad5fe8  
**tip base:** `c2eba69` (+ this fire’s engine commit)  
**PASS/FAIL:** **FAIL** (≥10% pp/s bar not met)  
**schedule stop?** **n** (fire **2/3** of “not viable without mlx PrefillArena / true build-once” ladder)

## Idea (NEW vs F1)

**Whole-chunk single HIP graph for prepare:** `MLX_PREFILL_ONE_GRAPH=1` sets `gpu_set_graph_decode_mode(true)` only during `prepare_fn` so multi-token chunk forwards do **not** mid-commit (F1 left decode-mode off → graphs still fragmented by op/MB caps). After prepare, mode restored false. Needs mlx graphs on for decode-mode gate (`MLX_USE_HIP_GRAPHS=1` or `MLX_HIP_GRAPH_DECODE=1`) plus optional `MLX_GRAPH_PREFILL_REPLAY=1`.

## Baseline vs experiment (1653 tok, step=128, no MTP, pure OFF)

| Probe | prefill_s | pp/s | gen t/s | peak mem |
|-------|-----------|------|---------|----------|
| baseline_eager | 14.763 | **111.97** | 30.67 | 18.3 GB |
| one_graph (USE_HIP_GRAPHS=1 + REPLAY + ONE_GRAPH) | 14.298 | **115.61** | 29.63 | 20.8 GB |
| one_graph_decode_flag only (DECODE=1 + REPLAY + ONE_GRAPH) | 15.305 | **108.00** | 19.27 | 18.3 GB |
| r2 one_graph | 14.318 | **115.45** | 29.68 | 20.8 GB |
| r2 baseline_eager | 14.879 | **111.10** | 30.43 | 18.3 GB |

**Mean:** eager **~111.5 pp/s** · one_graph (USE) **~115.5 pp/s** → **~+3.6%** (still ≪10%)  
**DECODE-only gate:** **regression** (~−3.5% pp/s, gen t/s collapsed) — do not use.

## Code paths

- `src/common/generate.cpp` — `MLX_PREFILL_ONE_GRAPH`, banner fields, restore decode-mode after prepare  
- mlx local: same F1 `use_hip_graphs` opt-in patch (not product default)

## Interpretation

Whole-chunk graphs do not unlock a product-scale prefill win on gfx1150 APU. Delta remains in the few-percent band (F1 split ~+2.7%, F2 one-graph ~+3.6%). Remaining headroom likely needs **PrefillArena + true build-once** in mlx, not more engine flag combos.

## Logs

`F2_baseline_eager.txt`, `F2_one_graph.txt`, `F2_one_graph_decode_flag.txt`, `F2_r2_*.txt`
