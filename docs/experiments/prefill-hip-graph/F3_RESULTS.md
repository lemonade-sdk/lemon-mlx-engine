# F3 results summary

**Date:** 2026-08-01  
**task_id:** 019fbfad5fe8  
**tip base:** `56d131b` (+ this fire’s commit)  
**PASS/FAIL:** **FAIL** (long-prompt ≥10% pp/s bar not met; F3 slightly regresses)  
**schedule stop?** **y** — **third consecutive fire** concludes prefill HIP graphs **not viable on gfx1150 without mlx PrefillArena / true build-once**

## Idea (NEW vs F1/F2)

**`MLX_PREFILL_ABSORB_TAIL=1`:** `llm_default_prepare` keeps chunking while `n > 1`, taking `min(step, n-1)`, so all multi-token prefill runs under prepare-time graph mode; only the last token goes to `step()` for first-sample. Combined with F2 `ONE_GRAPH` + `USE_HIP_GRAPHS` + `PREFILL_REPLAY`. Also remeasured **step=512** and a **short** prompt.

## Baseline vs experiment

### Long prompt (1653 tokens, fixed recipe)

| Probe | prefill_s | pp/s | Δ vs matched baseline |
|-------|-----------|------|------------------------|
| s128 baseline eager | 14.754 | **112.04** | — |
| s128 absorb+one_graph | 14.859 | **111.25** | **−0.7%** |
| s512 baseline eager | 12.496 | **132.28** | — |
| s512 absorb+one_graph | 12.597 | **131.23** | **−0.8%** |

peak mem: eager ~18.3 GB · graph ~20.8 GB · gen ~30 t/s either way

### Short prompt (22 tokens) — noise / not the product bar

| Probe | prefill_s | pp/s |
|-------|-----------|------|
| short baseline | 1.174 | 18.74 |
| short absorb+one_graph | 0.790 | 27.86 |

Short Δ looks large but is dominated by `gpu_exec` accounting on tiny N; **not** a fixed-long ≥10% claim. Long path is the contract metric.

## Code paths

- `include/mlx-lm/llm/llm_model.h` — `MLX_PREFILL_ABSORB_TAIL` (default off)
- `src/common/generate.cpp` — banner includes ABSORB_TAIL
- mlx local opt-in patch still required for graphs (F1)

## Three-fire ladder (honest)

| Fire | Approach | Long pp/s Δ |
|------|----------|-------------|
| F1 | split prefill graphs + ExecUpdate | ~**+2.7%** |
| F2 | whole-chunk ONE_GRAPH | ~**+3.6%** |
| F3 | absorb tail + ONE_GRAPH (s128/s512) | ~**−0.7%** |

**Conclusion:** engine-side flag combos cannot deliver a product-scale prefill graph win on this APU. See `PREFILL_ARENA_DESIGN.md`.
