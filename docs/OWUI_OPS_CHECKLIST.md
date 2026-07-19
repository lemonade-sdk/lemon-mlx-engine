# OpenWebUI ops checklist (eager, no MTP)

**Scope:** Operator path — **no** `--use-mtp`, **eager** decode only.  
**Canonical model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Engine branch:** `fix/eager-no-mtp-correctness` (ChatSession / stop / thinking floor / load)  
**Loop7 pack:** `docs/experiments/verify-loop7-owui/`

## APU / process rule (890M class)

| Rule | Why |
|------|-----|
| **One** 35B process only (`server` **or** `chat`, not both) | Concurrent full loads can SIGSEGV on first GDN forward after MTP-head skip |
| Kill leftover `./build/chat` before starting server | Loop6 ops finding |

## H0 — Backend gate (do this first)

| ID | Check | Action / PASS |
|----|-------|----------------|
| **O0a** | Engine models | `curl -sS http://127.0.0.1:8080/v1/models` lists `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| **O0b** | Not GGUF | Do **not** use lemonade/llama GGUF `:8001` as engine evidence |
| **O0c** | OWUI base URL | OWUI → `http://127.0.0.1:8080/v1` (same host/port as O0a) |
| **O1** | Curl twin | Short `2+2` curl on that URL is coherent (see loop6 S1) |
| **O3** | Memory A/B | If UI only fails, turn **Memory/RAG off** and retry new chat |
| **O5** | `role:tool` honesty | Curl below must return **HTTP 400** (not 200 soup) |

If OWUI hits GGUF, pure-graph/MTP/ChatSession notes do **not** apply.

## MTP head weights on hybrid checkpoints

Models named `*-MTP*` may still contain `mtp.*` tensors. On this branch the
engine **skips building the MTP head by default** so eager load stays stable.

| Goal | Env |
|------|-----|
| Eager / no MTP (default) | leave unset — log: `skipping optional MTP head build` |
| Enable MTP head construction | `export MLX_LOAD_MTP_HEAD=1` before starting the server |

Do **not** pass `--use-mtp` until MTP decode is fixed (still deferred).

## Recommended OWUI settings for stable chat

| Setting | Recommendation |
|---------|----------------|
| Memory / RAG tools | **Off** until multi-turn `role:tool` is product-supported |
| Native function tools | Off on stock main; tools branch may emit `tool_calls` but tool **follow-up** roles may still **400** |
| Max tokens | Prefer **≥ 4096** when reasoning/thinking is on (server floors low budgets when thinking=on) |
| New chat | Use for repros so history injection is controlled |

## Expected protocol behaviors

| Client action | Engine behavior |
|---------------|-----------------|
| Normal chat, no tools | Fresh KV per request; full messages re-sent each turn |
| `stop: ["…"]` | Honored (suffix strip + stop generation) |
| `role: "tool"` multi-turn | **400** (v1 freeze) — not gibberish 200; error body tells you to disable Memory/tools |
| Tools first-turn inject | May emit `tool_calls` (tools branch); **follow-up** `role:tool` still 400 |
| Thinking on + low max_tokens | Server may raise budget to **4096** and log `thinking_budget_floor` |
| tools_auto | Forces thinking **off** for that request only (not sticky) |

## O5 — `role:tool` must 400 (Memory / tools follow-up)

```bash
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit",
    "messages": [
      {"role": "user", "content": "hi"},
      {"role": "tool", "content": "result"}
    ],
    "max_tokens": 32
  }'
# PASS: HTTP 400 and body mentions role "tool" not supported / Memory
```

Unit equivalent (no 35B load): `test_server_api` case  
`role tool message returns 400 in v1` — **PASS** on this branch.

## Quick curl twin (parity with OWUI)

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1   # optional; pure default is already off on this branch
# no --use-mtp
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit",
    "temperature": 0,
    "max_tokens": 512,
    "messages": [{"role":"user","content":"What is 2+2? One word."}]
  }'
```

If **curl is clean** and OWUI is not → client history / Memory / proxy SSE, not decode.

## Related

- `docs/FIX_AND_VERIFY_GUIDE.md` — full P# verify matrix  
- `docs/LOOP_STATUS.md` — branch progress  
