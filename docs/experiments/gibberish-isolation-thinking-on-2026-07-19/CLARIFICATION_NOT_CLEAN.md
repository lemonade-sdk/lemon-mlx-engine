# Clarification: we are **not** claiming “no issues” on `feat/openai-tools-server`

**Date:** 2026-07-19

## What was *wrong* to read from the earlier no-think ladder

| Misreading | Actual meaning |
|------------|----------------|
| “L0–L2 coherent ⇒ branch is fine” | **False.** That only means: under *those* constraints (no tools, thinking **off**, single-turn HTTP, ≤800–1600 tokens), we did **not** reproduce mid-stream token soup. |
| “Same as main ⇒ no work left” | **False.** Shared decode code can still be **buggy**; tools branch can still have **protocol** issues; Discord path used **thinking** and possibly **MTP**. |
| “L4 500 is minor” | **False.** MTP is **broken** on this stack (`Stream(cpu, 0)`). High priority if any path uses `--use-mtp`. |

## Confirmed / open issue inventory (both feat and main unless noted)

| ID | Issue | Severity | Where | Status |
|----|--------|----------|--------|--------|
| **I1** | MTP generation fails: `There is no Stream(cpu, 0) in current thread` | **P0** | engine MTP + mlx stream context | **Reproduced** on feat (no-think L4). Same `generate.cpp` on main → expect same |
| **I2** | Thinking burns budget → empty/partial final content, `finish_reason=length` | **P0 product UX** | engine defaults + client max_tokens | Seen no-think L3; **re-running with thinking ON** |
| **I3** | CLI ChatSession multi-turn **double-prefill** on non-empty KV | **P0 CLI** | `chat_session.cpp` | Code-confirmed; not OWUI HTTP |
| **I4** | Pure-graph default ON + process-global `graph_external_pos` | **P1 risk** | `generate.cpp` / `graph_decode.cpp` | Not cleared as safe for all lengths; no-think short gens only |
| **I5** | Request `stop` ignored; multi-id EOS collapse | **P1** | server / `llm_factory` | Code-confirmed |
| **I6** | OpenWebUI tools/Memory protocol | **P1 client+server** | **feat differs from main** (tools parse/emit) | Not exercised by no-tools ladder |
| **I7** | mlx `rocm-support` tip drift | **P2 repro** | FetchContent branch pin | Same mlx in current feat/main proof builds |

## What we are doing now

1. **Re-run isolation with THINKING ON** (Discord-like) on **feat** binary.  
2. **Build `main` worktree** with **same mlx tree** → run **identical thinking-on ladder**.  
3. **Compare** feat vs main RESULTS for real product delta vs shared bugs.  
4. Update analysis MD with honest conclusions (not “clean”).

## Canonical model (unchanged)

`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` only.
