# Loop11 — CI smoke thinking floor interaction

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**No MTP / pure-graph.**

## Failure (prior CI)

`test-simple-math-ubuntu-cpu` / `macos-arm64`:

- Server log: `thinking_budget_floor: max_tokens 5 → 4096`
- Smoke expected short `2+2` → `4`
- Runner OOM/kill after floor raised budget (thinking default ON)

## Root cause

| Factor | Detail |
|--------|--------|
| Matrix | `no_think: false` on all active simple-math rows |
| Product floor | When thinking on, raise `max_tokens` &lt; 4096 to 4096 |
| Warm-up | `max_tokens: 5` without `enable_thinking: false` |
| Inference | `max_tokens: 128` same |

**Product floor is correct** (Discord CoT UX). Smoke was misconfigured for Qwen thinking defaults.

## Fix (this commit)

| File | Change |
|------|--------|
| `build-mlx-engine.yml` matrix | `no_think: true` for ubuntu-rocm/cpu/macos |
| smoke shell | default `NO_THINK_FLAG=--no-think` (opt-out only if matrix false) |
| warm-up + inference JSON | `"enable_thinking": false` |
| `test-mlx-engine.yml` + composite action | same `enable_thinking: false` belt |

**Not changed:** `thinking_budget.h` / server floor policy.

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| CI smoke no longer forces 4096-token thinking gen | **Yes** (config) |
| Product thinking floor still floors low client budgets when thinking on | **Yes** |
| Full CI green verified this loop | **No** — wait for GHA after push |
