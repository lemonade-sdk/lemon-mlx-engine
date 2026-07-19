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

## Loop4 notes

- Unit tests re-run: chat_session / thinking_budget / stop_sequences **green**
- Empirical 35B server load: **segfault** during model load (see `docs/experiments/verify-loop4-smoke/`) — not decode path

## Loop (resume) — empirical + load mitigations

**Pack:** `docs/experiments/verify-eager-no-mtp-2026-07-19/`

| Gate | Status |
|------|--------|
| Units (chat_session / thinking_budget / stop) | **PASS** |
| E3 stop mid (`" 5"`) | **PASS** |
| L0 Maxwell short | **PASS** |
| C3 HTTP multi-turn Ada | **PASS** |
| Server reload after MTP-head skip + quant fuse off | **PASS** (health + hi smoke) |
| C1 CLI multi-turn Ada | **PASS** (`logs/C1-chat-loop.log`) |

**Code (landed):** skip MTP head default; quant fuse opt-in; chat/server warmup `synchronize` + `MLX_SKIP_WARMUP`.

## Loop6 — full empirical gate (same model)

**Pack:** `docs/experiments/verify-loop6-smoke/`

| Gate | Status |
|------|--------|
| Units (chat / thinking / stop) | **PASS** |
| S1 `2+2` → `4` | **PASS** |
| E3-mid stop `" 5"` | **PASS** |
| C3 HTTP Ada multi-turn | **PASS** |
| L0 Maxwell coherent | **PASS** |
| T1 thinking floor `64 → 4096` log + answer | **PASS** |
| C1 CLI Ada multi-turn | **PASS** |
| Dual-process load flakiness documented | concurrent chat+server can SIGSEGV; single process OK |

**PR:** https://github.com/lemonade-sdk/lemon-mlx-engine/pull/63 — description must carry full product/ops context (not just code summary).

## Open / next

- Optional: commit large gibberish-isolation experiment packs (analysis-only)
- MTP / pure-graph still deferred
- Monitor GDN first-launch flakiness; never load two 35B processes on 890M

## Operator posture

```bash
# no --use-mtp; pure-graph unset (eager); one process only
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080
# thinking on by default → budget floor ≥4096 unless client already higher
# --no-think for short Q&A smoke
```

## Load path (eager) — landed

| Fix | Status |
|-----|--------|
| Skip MTP head unless `MLX_LOAD_MTP_HEAD=1` | **done** |
| Quant fuse opt-in `MLX_ENABLE_QUANT_FUSE=1` (default off) | **done** |
| Warmup `mx::synchronize` + `MLX_SKIP_WARMUP=1` | **done** |
| 35B load + full loop6 matrix | **PASS** (see verify-loop5-load + verify-loop6-smoke) |

## mlx need confirmation (2026-07-19)

**Decision: NO NripeshN/mlx PR/issue for now.**

Exclusive cold matrix (`docs/experiments/mlx-need-confirm-2026-07-19/`): chat load 3/3 PASS; server decode PASS.
Historical chat SIGSEGV remains intermittent/confounded; product path unblocked by engine mitigations.
Escalate to mlx only if exclusive fail rate returns (see MLX_PR_NEED_CONFIRMATION.md).

