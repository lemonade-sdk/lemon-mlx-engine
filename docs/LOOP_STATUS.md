# Loop status — fix/eager-no-mtp-correctness

**Scope:** No MTP · eager only (no pure-graph as bug path)

## Shipped this loop

| Item | Status |
|------|--------|
| ChatSession multi-turn | **Fixed** — fresh KV + full re-prefill each turn; re-hydration folds into `messages_` |
| Request `stop` strings | **Honored** in chat + completions (block + stream) |
| Multi-id EOS | **Merge** template EOS; do not replace multi-id set |
| Pure-graph | **Opt-in** via `MLX_DECODE_GRAPH_PURE=1` (default eager) |
| Unit tests | `test_chat_session` multi-turn + re-hydrate (91 asserts) |

## Deferred

- MTP / Stream(cpu,0)
- Pure-graph quality work (operator not using graph)

## Next loop focus

- Thinking budget / max_tokens product defaults or docs
- Optional: stop-sequence unit tests
- Empirical smoke on CLI multi-turn with canonical model if VRAM free
