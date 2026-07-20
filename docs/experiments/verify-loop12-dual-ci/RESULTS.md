# Loop12 — dual CI thinking + no-thinking (Clear Thought)

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**Operator ask:** keep thinking **and** no-thinking in CI for analysis; do not permanently drop thinking.

## Clear Thought summary

| Tool | Outcome |
|------|---------|
| Collaborative (quintuple) | Dual lanes; never max_tokens=5 with thinking on; product floor stays; MTP separate |
| Decision framework | Prefer dual; thinking budget must be real (4096) |
| Scientific method | H: failure was undersized max_tokens × floor, not thinking mode itself |
| Structured argumentation | Reject “permanently drop thinking from CI” |
| Metacognitive | Product thinking was never removed—only CI matrix |

## What we changed

`build-mlx-engine.yml` `test-simple-math` matrix:

| mode | platforms | Server | Request | Extra |
|------|-----------|--------|---------|-------|
| `no-think` | rocm, cpu, macos | `--no-think` | `enable_thinking: false`, max_tokens 128 | fast |
| `thinking` | rocm, cpu, macos | thinking on | `enable_thinking: true`, max_tokens **4096**, curl 600s | floor probe 64→4096 (short curl timeout, log-only) |

Warm-up always `enable_thinking: false` (load only—not the experiment).

## What we did **not** change

- Product `thinking_budget.h` floor (4096)
- MTP enable / `--use-mtp` CI (still deferred)

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| CI again exercises both modes | **Yes** (config) |
| Dual lanes green on GHA this commit | **No** — wait for run |
| MTP product ready | **No** |
