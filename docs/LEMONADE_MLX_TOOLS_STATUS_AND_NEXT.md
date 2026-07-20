# Lemonade MLX + Tools: Status & Next Work Plan

**Status date:** 2026-07-19 (live-checked against GitHub)  
**Owner (this workstream):** antmikinka  
**Backend owner (lemonade lemon-mlx integration):** fl0rianr (+ co-author bong-water-water-bong)  
**Constraint:** No force-push for landing tools work. Do not race a second full backend PR.

This document is the single coherent reference for **where we are**, **what each issue/PR/branch means**, and **exactly how we proceed next**.

---

## 1. One-sentence status

**Land lemon-mlx backend via #2013 first; ship OpenAI tools as a follow-up PR after #2013 merges, with engine tools from #62 (or a tools-enabled release pin). Full-stack draft #2751 is sandbox only.**

---

## 2. Architecture (two layers)

Tools require **both** repos. Do not confuse them.

```
┌──────────────────────────────────────────────────────────────────┐
│  lemon-mlx-engine  (inference binary)                             │
│  Job: load model, run chat, PARSE markup → emit OpenAI tool_calls │
│  Branch/PR: feat/openai-tools-server → PR #62                     │
│  Stock pin b1049 / main today: NO tools commits                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │  lemonade spawns server via
                             │  lemon-mlx.rocm_bin / builtin pin
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  lemonade  (API / process manager / UI)                           │
│  Job: forward tools, preserve tool_calls on SSE, caps/tests      │
│  Backend vehicle: PR #2013 (fl0rianr/add_mlx_lemon_backend)       │
│  Tools follow-up: AFTER #2013 → main (new PR, tools-only)        │
└──────────────────────────────────────────────────────────────────┘
```

| Layer | Without it… |
|-------|-------------|
| Engine tools (#62) | Model may print tool XML/text; no structured `tool_calls` |
| Lemonade stream hygiene | Engine tools can be dropped/clobbered on stream path |
| Lemonade caps True | `server_llm` 012/013 skip for lemon-mlx |
| Backend #2013 | No lemon-mlx recipe in product at all |

---

## 3. Issues

| Issue | Repo | Title | Role | State |
|-------|------|-------|------|--------|
| [#1642](https://github.com/lemonade-sdk/lemonade/issues/1642) | lemonade | MLX Engine ROCm backend feature | **Product home for lemon-mlx** | OPEN |
| [#60](https://github.com/lemonade-sdk/lemon-mlx-engine/issues/60) | engine | gfx1150 missing from ROCm release fatbin | Hardware/release (orthogonal to tools) | OPEN |

**Rule:** `#1642` is closed by the **backend** PR (#2013), not by a tools-only follow-up.

---

## 4. Pull requests (live roles)

### 4.1 lemonade

| PR | Branch | Role | State (as of status date) | Merge? |
|----|--------|------|---------------------------|--------|
| [#2013](https://github.com/lemonade-sdk/lemonade/pull/2013) | `fl0rianr/add_mlx_lemon_backend` | **Backend vehicle** (chat, reasoning stream, ROCm, registry, UI, CI) | OPEN, APPROVED, MERGEABLE | **Yes — primary** |
| [#2751](https://github.com/lemonade-sdk/lemonade/pull/2751) | `fix/mlx-stream-tool-hygiene` | **Sandbox / validation stack** (full reimpl + tools for learning) | OPEN **draft** | **No** as second full backend |

### 4.2 lemon-mlx-engine

| PR | Branch | Role | State | Merge? |
|----|--------|------|-------|--------|
| [#62](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/62) | `feat/openai-tools-server` | **Tools runtime** (parse/emit, Qwen XML, tools_auto thinking) | OPEN **draft** | **Yes — engine track** |
| [#61](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/61) | `fix/rocm-hip-arch-gfx1150` | gfx1150+gfx1151 HIP fatbin for ROCm CI/releases | OPEN **draft** | Independent of tools |

---

## 5. Branches (what exists / what does not)

| Branch | Repo | Status | Notes |
|--------|------|--------|-------|
| `fl0rianr/add_mlx_lemon_backend` | lemonade | **Active** | Tip (status date): `f300520b` — **does not** contain our tools commits |
| `fix/mlx-stream-tool-hygiene` | lemonade | **Active (draft PR)** | Sandbox tip `38424347` — has tools history for reference |
| `tools/stack-on-pr2013` | lemonade | **Deleted** | Temporary mirror; removed by design |
| `feat/openai-tools-server` | engine | **Active (draft PR)** | Tip `dbc5740` — 3 tools commits on top of main/b1049 |
| `fix/rocm-hip-arch-gfx1150` | engine | **Active (draft PR)** | gfx1150 fatbin only |
| `main` | both | trunk | lemonade main has **no** lemon-mlx until #2013 merges |

---

## 6. Does #2013 “already have tools”?

### Short answer: **No** (not end-to-end)

| Check on #2013 tip | Result |
|--------------------|--------|
| Engine pin | ~`b1049-stable` (stock main — **no** tools commits from #62) |
| `mlx_server.cpp` tool_calls preserve/hygiene | **Absent** |
| `capabilities.py` lemon-mlx `tool_calls` | **False** (deliberate: commit *set tool_calls… to False*) |
| server_llm 012/013 for lemon-mlx | **Skipped** |
| Registry `tool-calling` labels | Present on some MLX models (label-only; tests off) |

fl0rianr disabled tools caps **not only for CI** — he wants the large backend PR clean. That was correct given stock engine pins cannot honestly claim tools.

---

## 7. What engine `feat/openai-tools-server` (#62) contains

Commits (on top of `main` / b1049-stable):

| SHA (short) | Purpose |
|-------------|---------|
| `66c0353` | OpenAI `tools` / `tool_choice` parse & emit (blocking + stream Tier-1) |
| `52b061b` | Qwen multiline XML `<function=…>` parse |
| `dbc5740` | `tools_auto` thinking + `ThinkingContextGuard` (no sticky thinking-off) |

**Not in stock b1046–b1049.** Using tools against lemonade today requires:

```json
"lemon-mlx": {
  "backend": "rocm",
  "rocm_bin": "/path/to/build-of-feat-openai-tools-server/server",
  "args": ""
}
```

Product `args` must stay `""` (no global process `--no-think` default). Test harness may use `--no-think` for lemon-mlx tools tests only.

---

## 8. History of our lemonade tools work (what happened)

1. Built and validated tools end-to-end locally (engine + lemonade).
2. Opened draft PRs: lemonade **#2751**, engine **#62**, engine **#61**.
3. Recognized **#2013** already owns the backend + intends to close **#1642**.
4. Coordinated with fl0rianr:
   - Prefer **#2013** as backend vehicle (not a second full backend).
   - He agreed **merge #2013 first**, tools as **follow-up**.
   - He also OK’d **minimal tools on #2013** for review (“easy to review”).
5. We FF’d four tools-only commits onto `fl0rianr/add_mlx_lemon_backend` (no force-push).
6. His branch was later **force-updated** (merge main / tip move) → **those tools commits are no longer on #2013 tip**.
7. Deleted temporary branch **`tools/stack-on-pr2013`**.
8. **Current team decision:** wait for **#2013 → main**, then open a **clean tools-only PR**.

Sandbox #2751 remains for reference; **do not merge it as the backend.**

---

## 9. Social / process agreements (rules)

| Rule | Detail |
|------|--------|
| Backend ownership | fl0rianr / #2013 |
| Tools follow-up ownership | antmikinka (after #2013 merges) |
| No force-push | Do not rewrite his or published history for landing |
| No dual full backend | Do not land #2751 as competing “add lemon-mlx” |
| #1642 close | **#2013 only** — tools PR uses `Related to #1642`, never `Closes #1642` |
| Product defaults | `lemon-mlx.args` stays `""` |
| No lemonade L4a | Do not auto-inject `/no_think` merely because `tools` is present (corrupts small Qwen tool XML); engine `tools_auto` owns that |

---

## 10. What the tools follow-up PR will contain (after #2013)

Re-apply **only** these classes of change (from #2751 / prior stack, rebased onto post-merge `main`):

### 10.1 Must include

| Area | Intent |
|------|--------|
| `mlx_server.cpp` stream path | Preserve/dedupe engine `tool_calls`; finish_reason tracking; blocking→stream emit tools; no free-text invent |
| Optional | INFO log of engine argv on spawn (audit custom args) |
| `capabilities.py` | `tool_calls` / `tool_calls_streaming` **True** for lemon-mlx + short policy comments |
| `server_models.json` | Honest labels: **only** product gate model(s) with `tool-calling` (e.g. Qwen3.5-4B-MLX); strip dishonest labels on plumbing models |
| `server_base.py` | Nested config merge; optional test-only `lemon-mlx.args` / `--no-think` for lemon-mlx tests; hard-fail set if needed |
| `server_llm.py` | lemon-mlx tools budget (e.g. ≥128 tokens); name asserts on 012/013 |
| `test_models.py` | `SAMPLE_TOOL` `description` (small-model reliability) |
| Integration tests | Assert tools True + stream hygiene markers; lemon-mlx-scoped caps checks |

### 10.2 Engine dependency for honest CI

| Path | Honest? |
|------|---------|
| Caps True + stock b1049 only | **No** — binary cannot emit tools |
| Caps True + `rocm_bin` tools build / post-#62 pin | **Yes** |
| Caps False until pin ships | **Yes** (safer for green CI if pin lags) |

**Follow-up PR must pick an explicit CI policy** (document in PR body):

- **A)** Flip caps True only when tools-capable pin is ready, **or**
- **B)** Flip True with documented override / expected CI failure until pin, **or**
- **C)** Keep False in CI and enable tools behind operator pin only (less preferred if we want 012/013 on).

Recommended default: **land stream hygiene first**; flip caps True in same PR **only if** engine #62 is merged/pinned or CI uses a known tools binary; otherwise split “hygiene” and “caps True” commits/PRs.

### 10.3 Optional / careful

| Item | Note |
|------|------|
| `enable_thinking=true` forward (L-fwd) | Global strip change can break llamacpp tests; scope to lemon-mlx or co-update tests |
| Whole-file replace of `mlx_server.cpp` | **Forbidden** — surgical port onto post-#2013 tree only |

---

## 11. Next work plan (ordered)

### Phase 0 — Now (waiting)

| # | Action | Owner |
|---|--------|--------|
| 0.1 | **Do not** re-push tools onto #2013 unless fl0rianr asks again | antmikinka |
| 0.2 | Watch **#2013** for merge to `main` | antmikinka |
| 0.3 | Optionally advance **engine #62** review / CI / docs | antmikinka |
| 0.4 | Optionally keep **#61** (gfx1150) moving independently | antmikinka |
| 0.5 | Leave **#2751** as draft sandbox or close when ready (no merge as backend) | antmikinka |

### Phase 1 — After #2013 merges

```text
git fetch origin main
git checkout -b feat/lemon-mlx-tools origin/main

# Port tools-only diffs from sandbox history / prior stack knowledge
# (do NOT re-introduce full backend scaffold)

# Local validation
# - build lemond
# - pytest test/test_lemon_mlx_integration.py
# - 012/013 with tools-enabled engine binary if caps True

git push -u origin feat/lemon-mlx-tools   # normal push, no force
gh pr create --base main --head feat/lemon-mlx-tools
```

PR body must include:

- Summary of tools-only scope  
- `Related to #1642` (not Closes)  
- Depends on engine #62 / tools pin / `rocm_bin`  
- Product `args` empty  
- Test plan + CI honesty  

### Phase 2 — Engine pin / release (can parallelize)

| # | Action |
|---|--------|
| 2.1 | Merge engine #62 (or equivalent) |
| 2.2 | Cut tools-bearing release tag (e.g. next `bNNNN-stable`) |
| 2.3 | Bump lemonade `backend_versions.json` lemon-mlx pin |
| 2.4 | Re-run lemon-mlx tools CI on stock pin |

### Phase 3 — Optional hardware

| # | Action |
|---|--------|
| 3.1 | Land engine #61 for gfx1150 ROCm release fatbin (#60) |

---

## 12. Validation gates (before tools follow-up is “ready”)

| Gate | Requirement |
|------|-------------|
| G1 | Build `lemond` green on tools branch |
| G2 | `pytest test/test_lemon_mlx_integration.py` green |
| G3 | Product `defaults.json` → `lemon-mlx.args == ""` |
| G4 | If caps True: local 012/013 green against **tools-enabled** engine |
| G5 | PR base = `main` (post-#2013); no second full backend |
| G6 | Body: Related #1642 / engine #62; never Closes #1642 alone for tools |
| G7 | No force-push of published history |

---

## 13. Reference links

| Resource | URL |
|----------|-----|
| Feature issue | https://github.com/lemonade-sdk/lemonade/issues/1642 |
| Backend PR | https://github.com/lemonade-sdk/lemonade/pull/2013 |
| Sandbox PR | https://github.com/lemonade-sdk/lemonade/pull/2751 |
| Engine tools PR | https://github.com/lemonade-sdk/lemon-mlx-engine/pull/62 |
| Engine tools commits | https://github.com/lemonade-sdk/lemon-mlx-engine/commits/feat/openai-tools-server/ |
| gfx1150 PR | https://github.com/lemonade-sdk/lemon-mlx-engine/pull/61 |
| gfx1150 issue | https://github.com/lemonade-sdk/lemon-mlx-engine/issues/60 |

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **Backend vehicle** | PR that introduces lemon-mlx into lemonade (#2013) |
| **Tools delta / follow-up** | Small PR after backend: stream hygiene + caps/tests |
| **Tools runtime** | Engine code that emits `tool_calls` (#62) |
| **Sandbox** | #2751 full stack for validation; not the merge path |
| **Fast-forward** | Additive push of new commits only (no history rewrite) |
| **Force-push** | History rewrite — **avoid** for landing this workstream |
| **Tier-1 stream tools** | Complete tool_call deltas after generation (not true arg streaming) |

---

## 15. Bottom line

| Question | Answer |
|----------|--------|
| Where is the backend landing? | **#2013 → main** |
| Where is tools landing (lemonade)? | **New PR after #2013 merges** |
| Where is tools landing (engine)? | **#62** (then release pin) |
| What is #2751? | **Sandbox only** |
| What was `tools/stack-on-pr2013`? | **Deleted temporary mirror** |
| Are tools on #2013 tip now? | **No** |
| What do we do today? | **Wait for #2013; optionally push #62** |
| What do we do after #2013? | **Tools-only PR from main** |

---

*End of status document. Update this file when #2013 merges or when the tools follow-up PR is opened.*
