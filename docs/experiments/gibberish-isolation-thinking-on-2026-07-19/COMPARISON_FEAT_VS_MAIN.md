# COMPARISON: feat/openai-tools-server vs main (THINKING ON)

**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` only
**Thinking:** ON (`enable_thinking: true`, no `--no-think`)
**Binaries:**
- feat: `build/server` md5 `82d052b311c4934eb9af07a38b9e48a2`
- main-equiv: `build/server-main-equiv` md5 `9f86e19c3e994c56521b2f3d5d8b0ccc`

## Method for main binary

See `main-build-method/METHOD.md`. Main server sources from `origin/main` @ `915da7f`, same mlx tree and ROCm build as feat.

## Critical clarification

**We are NOT claiming feat is clean.** Prior no-think coherent L0–L2 only falsified *one* hypothesis class under *narrow* conditions.

## Byte-level response identity

| file | content_identical | feat finish | main finish | content_chars | notes |
|------|-------------------|-------------|-------------|---------------|-------|
| `T0-long.json` | **True** | `length` | `length` | 4003 | only `id`/`created` differ |
| `T0-short.json` | **True** | `stop` | `stop` | 3543 | only `id`/`created` differ |
| `T2-long.json` | **True** | `stop` | `stop` | 1253 | only `id`/`created` differ |
| `T2-short.json` | **True** | `stop` | `stop` | 3403 | only `id`/`created` differ |
| `T4-long-mtp.json` | **True** (error) | `ERROR` | `ERROR` | 0 | same Stream(cpu,0) |
| `T4-short-mtp.json` | **True** (error) | `ERROR` | `ERROR` | 0 | same Stream(cpu,0) |

**All model `content` strings identical feat vs main:** `True`  
(Raw JSON not byte-identical only because `id` / `created` timestamps differ.)

## Issue inventory after thinking-on dual run

| ID | Issue | feat | main |
|----|--------|------|------|
| I1 | MTP `Stream(cpu, 0)` HTTP 500 | **YES** T4 | **YES** T4 |
| I2 | Thinking budget / length cut | T0-long `length` | **same** |
| I3 | ChatSession double-prefill | code both | code both |
| I4 | Pure-graph default ON | T2 ran coherent here | **same** |
| I5 | tools API surface | **feat only** | absent |
| I6 | mid-stream token soup (this prompt/budget) | **not seen** T0/T2 | **not seen** |

## Honest conclusions

1. **feat is not “issue-free.”** MTP is hard-broken; thinking can hit length; CLI multi-turn bug remains; tools path is new surface area.
2. **Decode/generation quality for this ladder is NOT a feat regression vs main.** Responses matched byte-for-byte with same mlx + same decode stack.
3. Discord “gibberish with thinking” is still best explained by: (a) thinking+max_tokens length UX, (b) MTP if enabled, (c) OWUI tools/history, (d) CLI multi-turn — **not** by tools-branch-only decode drift.
4. Next work should fix **shared** bugs on main-line decode/MTP/session, not blame tools PR for pure-graph soup we could not reproduce under thinking-on single-turn HTTP.

## Manual verify paths

| Pack | Path |
|------|------|
| feat RESULTS | `feat/RESULTS.md` |
| main RESULTS | `main/RESULTS.md` |
| feat raw | `feat/raw/` |
| main raw | `main/raw/` |
| this comparison | `COMPARISON_FEAT_VS_MAIN.md` |
| clarification | `CLARIFICATION_NOT_CLEAN.md` |
