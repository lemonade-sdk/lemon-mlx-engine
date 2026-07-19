# Empirical verify — eager / no-MTP

**Date:** 2026-07-19  
**Branch tip:** see `meta.jsonl` / latest commit  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Posture:** no `--use-mtp`; pure-graph off; `--no-think` for smoke

## Unit tests (local)

| Suite | Result |
|-------|--------|
| `test_chat_session` | **PASS** |
| `test_thinking_budget` | **PASS** |
| `test_stop_sequences` | **PASS** |

## HTTP (server)

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **E3-mid** | `stop: [" 5"]` | **PASS** | `raw/E3-stop-mid.json` → `"1 2 3 4"` |
| **L0-short** | Maxwell concise | **PASS** | `raw/L0-short-maxwell.json` |
| **C3** | HTTP multi-turn Ada | **PASS** | `raw/C3-http-multiturn.json` → `"Ada"` |
| Load smoke (MTP head skip) | health + hi | **PASS** | `raw/load-*.json` |

## CLI ChatSession (C1)

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **C1** | CLI multi-turn Ada → name? | **PASS** | `logs/C1-chat-loop.log`: turn1 `OK`, turn2 `Ada` (fresh KV full re-prefill path) |

Earlier intermittent **SIGSEGV** on chat first-forward (GDN/`hipLaunchKernel`) observed under dirty GPU / race; mitigated with post-load `mx::synchronize()` + optional `MLX_SKIP_WARMUP=1`. Re-run on clean GPU succeeded through multi-turn.

## Load mitigations (code)

- Skip optional MTP head unless `MLX_LOAD_MTP_HEAD=1`
- Quant fuse opt-in `MLX_ENABLE_QUANT_FUSE=1` + concat shape guards
- Warmup: synchronize before/after; `MLX_SKIP_WARMUP=1` escape hatch

## Operator

```bash
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit --host 127.0.0.1 --port 8080 --no-think
# CLI:
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit --no-think --max-tokens 64
```
