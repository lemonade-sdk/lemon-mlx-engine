# Lever 4 — Graph decode MoE+GDN 35B (inventory)

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Status:** INVENTORY — field T₁ kill probe not run this fire (temp×think matrix primary).

## Existing machinery (code)

| Mechanism | Location | Notes |
|-----------|----------|-------|
| `gpu_set_graph_decode_mode(L==1)` | `generate.cpp` decode path | T=1 decode can enter graph mode |
| Fixed `graph_decode_input()` buffer | `graph_decode.h` | Capture/replay address stability |
| `MLX_HIP_GRAPH_DECODE` / `MLX_USE_HIP_GRAPHS` | env (mlx side) | Required for HIP graph capture |
| Prefill graphs | `MLX_PREFILL_ONE_GRAPH`, F1–F3 | **Missed ≥10% pp/s bar** on this stack |
| MTP sequential verify | default | Uses graph-decode mode for T=1 trunk |
| MTP batch verify | `MLX_MTP_BATCH_VERIFY=1` | **S4 KILL** on gfx1150 |

## Hypothesis

If launch overhead dominates T₁, full HIP graph capture of MoE+GDN T=1 could cut T₁ 38→28–32 ms.

## Blockers

1. **MoE data-dependent expert routing** — different experts per token may prevent stable graph topology.  
2. **GDN recurrence** — sequential state; multi-token batch already failed S4.  
3. Prior pure-graph work **flat** on smaller model (historical).  

## Measure criteria (when probing)

See [`../NOTABLE_WINS.md`](../NOTABLE_WINS.md): **any** logged gen t/s / T₁ improvement is **notable**.

- Measure eager T₁ (or gen t/s) with graphs on vs off, temp=0 **and** temp=0.7 no-think at minimum (think cell optional).  
- **Record** all deltas including +1–4%.  
- **FUND** multi-day polish if gain is sustained and ≥~5% **or** cheap default-on; still ship notes for smaller notables.  
- **KILL path only if:** capture fails / SEGV / thrash / **regress**. Flat (~0%) = no win, not “ignore noise only” without n≥2.

## Next probe (after temp×think baselines)

```bash
# baseline already in T_E0_*
MLX_USE_HIP_GRAPHS=1 MLX_HIP_GRAPH_DECODE=1 MLX_ENABLE_QUANT_FUSE=1 \
  ./build/chat MODEL --temperature 0 --no-think --max-tokens 128 ...
# compare Generation t/s
```
