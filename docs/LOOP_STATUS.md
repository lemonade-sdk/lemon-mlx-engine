# Loop status — fix/eager-no-mtp-correctness

**Scope:** No MTP · eager only (no pure-graph as bug path)  
**Branch:** `fix/eager-no-mtp-correctness`  
**Scheduler task:** `019f7c546fc8` (5m)

## Supervisors (quintuple)

| Role | Focus this loop |
|------|-----------------|
| Decode/ROCm | Eager default; no MTP; no pure-graph enable |
| API/Client | Thinking budget floor + defaults 4096 |
| QA/Regression | `test_chat_session` hold; rebuild server |
| Ownership | Engine-only changes; no mlx PR |
| Product/Ops | Discord collapse = thinking budget / CLI multi-turn |

## Shipped (commits)

| Item | Status |
|------|--------|
| ChatSession multi-turn | **Fixed** — fresh KV + full re-prefill each turn |
| Request `stop` strings | **Honored** (block + stream) |
| Multi-id EOS | **Merge** (no singleton replace) |
| Pure-graph | **Opt-in** `MLX_DECODE_GRAPH_PURE=1` (default eager) |
| OpenAI `stop` unit tests | **`stop_sequences.h` + `test_stop_sequences`** |
| OWUI ops | **`docs/OWUI_OPS_CHECKLIST.md`** |
| Thinking budget | **Floor to 4096** when thinking=on and client max_tokens lower; CLI/server/OpenAI defaults **4096**; `thinking_budget.h` + unit tests |
| Unit tests | `test_chat_session` multi-turn + re-hydrate; `test_thinking_budget` |

## Deferred

- MTP / Stream(cpu,0)
- Pure-graph quality work (operator not using graph)
- Full OWUI Memory/tools product work (client-heavy)

## Open / next loop

- Empirical CLI multi-turn / thinking smoke on canonical 35B if VRAM free
- Optional: analysis experiment packs commit
- PR open for `fix/eager-no-mtp-correctness` when ready

## Operator posture

```bash
# no --use-mtp; pure-graph unset (eager)
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080
# thinking on by default → budget floor ≥4096 unless client already higher
```
