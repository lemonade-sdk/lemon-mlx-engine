# H-KV instrumented thrash (2026-07-30)

## Question

If attention KV does not advance correctly, decode can lock into repetition and garbage.  
Is that what happens during our multi-turn thrash?

## Method

```bash
printf '%s\n' \
  'Explain how a phased array radar steers a beam without moving antennas.' \
  'Now simplify for a non-expert.' \
  'What fails if phase synchronization drifts?' \
  'who are you?' \
  'quit' \
| env MLX_KV_OFFSET_LOG=1 MLX_KV_OFFSET_EVERY=16 \
    ./build/chat mlx-community/Qwen3.5-0.8B-4bit \
    --temperature 0 --max-tokens 400
```

- Tip: `f715043` (includes `MLX_KV_OFFSET_LOG`)
- Path: **default** (no fused2)
- Log: `logs/TIP_0.8B_radar_KVLOG.txt`

## Results

| Metric | Value |
|--------|--------|
| EXIT | 0 |
| HISTORY | OK 25 → 442 → 860 → 1274 |
| Gens | 400/400/400/400 (hit budget) |
| Content thrash | **Yes** — classic wrong-fact CoT loop (*“standard phased array radar uses moving antennas…”* ×31+) |
| `[kv]` log lines | 100 |
| **STALL markers** | **0** |
| non-advance pairs (prev ≥ off) | **0** |
| Offset sample turn 1 | tok 16→400: max_offset **41 → 425** (+1 per step) |
| Offset sample turn 4 end | max_offset **1674** (still advancing) |

## Conclusion

**During active content thrash, attention KV max_offset advanced every decode step.**

Therefore:

1. **H-KV (stuck attention offset) is REFUTED as the mechanism of this thrash class** — not only via P1/P2 ladder, but with **direct instrumentation under thrash**.
2. Thrash here is **logit/CoT self-reinforcement** (and/or residual GDN numerics on hybrid layers), **not** frozen attention KV.
3. Cross-turn re-appearance of the same thrash phrase is consistent with **history text re-seed** (assistant thrash saved → next prefill), not leftover KV (fresh cache each turn).

## What is still necessary

| Item | Why |
|------|-----|
| Keep `MLX_KV_OFFSET_LOG` for any **future 35B** thrash repro | Confirm same non-STALL on product model if thrash returns |
| Do **not** re-prioritize KV advance as primary fix | Evidence against |
| Residual: 0.8B thinking CoT; rare 35B default@0.7 thrash | GDN residual / sampling / history text — not H-KV |
| Human merge PR #74 f32 SSM stack | Product path |

## Do not

- Re-enable permanent `MLX_SYNC_DECODE` for this class  
- Treat LoopBrake as fix  
- Spin uninstrumented “KV is broken” arguments  
