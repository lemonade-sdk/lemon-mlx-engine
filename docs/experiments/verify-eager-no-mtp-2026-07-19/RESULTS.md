# Empirical verify — eager / no-MTP

**Date:** 2026-07-19  
**Branch tip at evidence time:** `bda8f21` / later (see `meta.jsonl`)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Posture:** no `--use-mtp`; pure-graph off (`MLX_DECODE_GRAPH_PURE_OFF=1` or default OFF); `--no-think` for smoke

## Unit tests (local)

| Suite | Result |
|-------|--------|
| `test_chat_session` | **PASS** |
| `test_thinking_budget` | **PASS** |
| `test_stop_sequences` | **PASS** |

Log: `logs/ctest-unit.log`

## HTTP (server) — canonical 35B

| ID | Test | Result | Evidence |
|----|------|--------|----------|
| **E3** | `stop: ["###END###"]` count 1–20 | **PASS** | `raw/E3-stop.json` — `finish_reason=stop`, 54 toks, stop string stripped/not present |
| **E3-mid** | `stop: [" 5"]` count 1–20 | **PASS** | `raw/E3-stop-mid.json` — content `"1 2 3 4"`, 9 toks (stop honored mid-stream) |
| **L0-short** | Maxwell equations concise | **PASS** | `raw/L0-short-maxwell.json` — coherent Gauss/Faraday/Ampère |
| **C3** | HTTP multi-turn Ada → name? | **PASS** | `raw/C3-http-multiturn.json` — content `"Ada"` |

## CLI ChatSession (C1)

| ID | Test | Result | Notes |
|----|------|--------|-------|
| **C1** | CLI multi-turn Ada | **BLOCKED** | `./build/chat` **SIGSEGV (exit 139)** after MTP-head skip during first forward (GDN / `copy_contiguous` / `hipLaunchKernel` in `libamdhip64`). GDB: `Qwen35MoEGatedDeltaNet::operator()` from chat `main` warmup/first call. **HTTP server path load+decode PASS** on same binary/model. Unit multi-turn + C3 still PASS. |

## Load stability (rebuild after quant-fuse / MTP-head skip)

| Check | Result | Evidence |
|-------|--------|----------|
| Server load + `/health` | **PASS** | `raw/load-health.json`, log shows `[MTP] skipping optional MTP head build` |
| Short chat completion | **PASS** | `raw/load-smoke-hi.json` — `"Hi there!"` |
| Memory after load | ~18.2 GB active (no MTP head) | server log |

In-tree load mitigations:

- Skip optional MTP head unless `MLX_LOAD_MTP_HEAD=1`
- Quant projection fuse opt-in only (`MLX_ENABLE_QUANT_FUSE=1`) + shape guards

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Stop strings work on HTTP | **Yes** (E3-mid strong) |
| ChatSession multi-turn unit + HTTP history | **Yes** |
| CLI C1 instrumented on 35B | **Not yet** |
| MTP fixed | **No** — deferred |
| Pure-graph quality | **No** — not used |

## Operator day-to-day

```bash
# default: eager, skip MTP head
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --no-think
# thinking on: max_tokens floors to 4096
```
