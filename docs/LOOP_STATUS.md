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

## Loop7 — OWUI H0 / I6 (this loop)

**Pack:** `docs/experiments/verify-loop7-owui/`  
**Clear Thought + supervisors:** OWUI next; no re-smoke loop6 matrix; no MTP/pure.

| Item | Status |
|------|--------|
| `role:tool` 400 message → Memory/tools guidance | **done** (server.cpp) |
| `test_server_api` tools cases + body assert | **PASS** |
| OWUI_OPS_CHECKLIST executable H0 (O0/O1/O3/O5) | **updated** |
| Multi-turn tools / Memory product | **deferred** (client + schema) |
| Units chat/thinking/stop | **PASS** (re-run loop7) |

## Loop8 — GDN materialize verify + live H0

**Pack:** `docs/experiments/verify-loop8-load/`  
**Code:** `a4d1d99` load-time `materialize_decode_constants` (qwen35_moe)

| Gate | Status |
|------|--------|
| 35B single-process load + health | **PASS** |
| S1 `2+2` → `4` | **PASS** |
| O0a `/v1/models` canonical id | **PASS** |
| O5 live `role:tool` 400 + Memory text | **PASS** |
| QA claim scope | load-time T=1 hygiene only (not mlx kernel root) |

## Loop9 — mlx dual-load ownership (comprehended)

**Local mlx:** `/home/antmi/mlx` @ `0dadb703` / `investigate/rocm-dual-load-segv`  
**Upstream draft:** https://github.com/NripeshN/mlx/pull/13  
**Engine pack:** `docs/experiments/verify-loop9-mlx-ownership/` + updated `mlx-need-confirm-…/MLX_PR_NEED_CONFIRMATION.md`

| Finding | Verdict |
|---------|---------|
| Exclusive 35B load/decode | **PASS** (product unblocked) |
| Dual-process second load | **FAIL SEGV 139** (M1) — ops + mlx/HIP robustness |
| Need mlx **code** PR for exclusive product? | **NO** |
| Need dual-load docs upstream? | **YES** — draft #13 |
| Engine soft warn (tight GPU mem before load) | **done** this loop |

## Open / next

- Optional: OWUI UI L7 (curl H0 green)
- Optional: pure-mlx microbench for dual-load OOM vs SEGV (mlx tree, not engine product)
- Optional: large isolation packs commit
- MTP / pure-graph still deferred
- **Never** two full 35B MLX processes on 890M

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

## GDN materialize (this loop)

- Load-time `materialize_decode_constants()` for Qwen35 MoE GDN (engine hygiene).
- Removes mid-forward bf16→f32 eval on first T=1 token when load succeeded.
- Exclusive chat load PASS after change; still **no** NripeshN/mlx PR (confirm pack stands).

