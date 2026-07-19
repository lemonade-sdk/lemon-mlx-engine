# Loop7 — OWUI H0 / I6 gate (eager, no MTP)

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**Supervisors:** Clear Thought collaborative + explore (API/Ownership) + quality-reviewer  
**Scope:** OWUI Memory/tools (I6) — **not** MTP, **not** pure-graph  
**Canonical model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`

## Quintuple decision

| Role | Verdict |
|------|---------|
| Decode/ROCm | Do not re-open load/fuse; single process only |
| API/Client | Engine stable chat OK; Memory/tools off for product |
| QA | Do **not** re-run loop6 S1/E3/C3/C1 matrix as progress |
| Ownership | No mlx PR; no multi-turn tools product this loop |
| Product/Ops | Land executable H0 pack + clearer `role:tool` 400 |

## Code this loop

| Change | Why |
|--------|-----|
| Clearer `role:tool` **400** message (server.cpp) | OWUI Memory injects tool turns; operator gets actionable text, not silent soup |
| `test_server_api` asserts 400 body mentions tool + Memory | Falsifiable |

**Not done (deferred):** accept multi-turn `role:tool` + request `tool_calls`/`tool_call_id` schema.

## Unit / API (no 35B reload)

| Check | Result |
|-------|--------|
| `test_server_api` `[server-api][tools]` | **PASS** (includes role:tool → 400) |
| `test_chat_session` | **PASS** |
| `test_thinking_budget` | **PASS** |
| `test_stop_sequences` | **PASS** |

## H0 falsifiable matrix (operator)

| ID | Action | PASS criteria | Loop7 status |
|----|--------|---------------|--------------|
| **O0a** | `GET …:8080/v1/models` on engine | lists canonical MLX id | Requires live server; see OWUI_OPS_CHECKLIST |
| **O0b** | Do not use GGUF `:8001` as engine evidence | documented | **PASS** (docs) |
| **O5** | curl/`test` with `role:tool` in messages | **HTTP 400** + not-supported + Memory hint | **PASS** (unit + message) |
| **O1** | Memory/tools **off** + curl twin | same quality as loop6 S1/C3 | Engine proven loop6; OWUI UI optional |
| **O3** | Memory on vs off A/B | attribute failures to Memory | **docs** (client) |

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Engine rejects Memory-style `role:tool` with clear 400 | **Yes** |
| Full OWUI Memory product works end-to-end | **No** — disable Memory |
| Multi-turn native tools follow-up | **No** — deferred |
| Loop6 decode gates still green | **Yes** (not re-run; no regression intent) |

## Related

- `docs/OWUI_OPS_CHECKLIST.md` (executable H0)
- `docs/ISSUE_OPENWEBUI_GIBBERISH_KV_ANALYSIS.md` (analysis)
- PR https://github.com/lemonade-sdk/lemon-mlx-engine/pull/63
