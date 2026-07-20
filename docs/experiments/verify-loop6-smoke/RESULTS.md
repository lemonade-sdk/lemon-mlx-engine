# Loop6 smoke — full eager / no-MTP gate on canonical 35B

**Date:** 2026-07-19  
**Branch tip:** `60860c4` (and local follow-up artifacts)  
**Model (ONLY):** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Hardware:** gfx1150 (AMD Radeon 890M), ROCm  
**Posture:** no `--use-mtp`; pure-graph off (default / `MLX_DECODE_GRAPH_PURE_OFF=1`); MTP head skipped; quant fuse off

## Units (local)

| Suite | Result |
|-------|--------|
| `test_chat_session` | **PASS** |
| `test_thinking_budget` | **PASS** |
| `test_stop_sequences` | **PASS** |

## Load / ops notes

| Observation | Detail |
|-------------|--------|
| Concurrent `./build/chat` holding 35B | **Causes** second process load SIGSEGV in GDN `copy_contiguous` → `hipLaunchKernel` after MTP-skip log. **Fix ops:** kill leftover chat/server before reload. |
| Clean single process load | **PASS** — health OK, active ~18.2 GB |
| Log marker | `[MTP] skipping optional MTP head build` |

GDB of dirty concurrent load: `docs/experiments/verify-loop6-smoke/gdb-load.txt`.

## HTTP (server `--no-think` on `:8080`)

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **S1** | `2+2` → `4` | **PASS** | `raw/S1-2plus2.json` |
| **E3-mid** | `stop: [" 5"]` count 1–20 | **PASS** content `"1 2 3 4"` | `raw/E3-stop-mid.json` |
| **C3** | HTTP multi-turn Ada → name? | **PASS** `"Ada"` | `raw/C3-http-multiturn.json` |
| **L0** | Maxwell short (coherent) | **PASS** | `raw/L0-maxwell.json` |

## Thinking budget floor (per-request `enable_thinking: true`)

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **T1** | `enable_thinking: true`, client `max_tokens: 64` | **PASS** server log `thinking_budget_floor: max_tokens 64 → 4096`; answer contains `8` | `raw/T1-thinking-floor.json`, `server.log` |
| **T2** | `enable_thinking: true`, client `max_tokens: 5000` | **PASS** (no second floor raise; only one floor line in log) | `server.log` |

## CLI ChatSession (C1)

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **C1** | multi-turn: name Ada → “What is my name?” | **PASS** turn1 `OK`, turn2 `Ada` | `logs/C1-chat-loop.log`, `raw/C1-verdict.txt` |

C1 must **not** share GPU/RAM with a live server process on this APU.

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Eager HTTP correctness (stop, multi-turn history, short Q&A) | **Yes** |
| Thinking floor raises low budgets when thinking on | **Yes** (log + answer) |
| CLI ChatSession multi-turn on 35B | **Yes** (C1) |
| MTP decode fixed | **No** — deferred |
| Pure-graph quality | **No** — not enabled |
| Zero load flakiness forever | **No** — concurrent dual-load and dirty GPU can still SIGSEGV first GDN launch |

## Operator

```bash
# single process only on 890M
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --no-think

# thinking floor is automatic when thinking is on and max_tokens < 4096
# optional: MLX_SKIP_WARMUP=1 if diagnosing first-forward crash
```
