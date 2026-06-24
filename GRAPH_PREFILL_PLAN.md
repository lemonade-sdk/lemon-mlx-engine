# Graph prefill — plan & status

Decode is now one HIP graph per token (decode-mode: one `hipGraphExecUpdate` +
one launch, beats eager on gfx1151/gfx1201). This branch tracks extending the
same approach to **prefill** and hardening the decode path.

## Goal

Batch the prompt-prefill forward into HIP graph(s) the way decode is, without
OOMing on long prompts. Prefill currently runs as hundreds of small commits
(one-time per prompt) — fine for short prompts, wasteful for long ones.

## Why prefill is different from decode

- Prefill processes T>1 tokens at once → intermediates are T× larger, so a single
  whole-forward graph can blow peak memory (the per-graph op/byte caps exist to
  bound this; decode-mode disables them only because decode's live set is small).
- Prefill shapes change with prompt length, so a captured/instantiated prefill
  graph is not reused across prompts the way the decode graph is across tokens.

## Approach options (to evaluate)

1. **Bounded graph chunks + ExecUpdate**: keep committing prefill in capped
   chunks (current behavior) but ensure each chunk is a clean kernel-only graph
   (child-graph flatten already landed) and reused via ExecUpdate when shapes
   recur. Lower risk; partial win.
2. **Chunked-prefill one-graph-per-chunk**: process the prompt in fixed-size
   token windows, each window = one graph, sized so peak memory stays under a
   budget. Bigger win, needs a memory model.
3. **Static-shape prefill graph** keyed by padded length buckets, reused across
   prompts. Most reuse, most complex.

## Tasks

- [ ] Measure prefill commit/launch count and peak memory vs prompt length.
- [ ] Decide chunk sizing from a memory budget (leave headroom over the model).
- [ ] Wire a prefill graph mode analogous to decode-mode (per-chunk, bounded).
- [ ] Validate bit-identical to eager across prompt lengths, both GPUs.
- [ ] Long-prompt OOM guard test on the 32GB R9700.

## Hardening carried over from decode-mode (do before/with prefill)

- [ ] 1000-token coherence soak (decode validated only to ~120 tokens so far).
- [ ] Sampling coverage: decode-mode validated at temp 0 / no processor only;
      verify temp>0 and repetition-penalty paths (the sampler ops are in-graph).
- [ ] R9700 longer-run stability (no intermittent graph faults).
- [ ] Remove leftover diagnostic knobs once not needed (eu_min/eu_max bisection
      gate; arghash is already stats-gated).

## Defaults (current, for reference)

- `MLX_USE_HIP_GRAPHS` — graphs ON by default (opt-out `=0`).
- `MLX_GRAPH_DECODE` — decode-mode ON by default (opt-out `=0`); active only on
  single-token steps.
- `MLX_DECODE_GRAPH` — old capture-once/arena path, OFF (opt-in).
- `MLX_GRAPH_EXECUPDATE` / `MLX_NO_CONCAT_SPLIT` — global opt-ins; decode-mode
  forces both for decode regardless.
