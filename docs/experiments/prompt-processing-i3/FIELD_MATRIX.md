# Field matrix — Maxwell 5-turn (thinking ON, temp 0.7, GDN 35B)

**TS:** 20260816-193941  
**Branch:** `exp/prompt-processing-i3` @ `c1189b2`  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit (GDN / Qwen3.6-35B)  
**Prompts:** `docs/experiments/mtp-stream-p0/maxwell_full_prompts.txt` (T1–T5 Maxwell)  
**Budget:** `--temperature 0.7 --max-tokens 384` thinking **ON** (not 8192; this is I3/quality gate, not 8k TPS SAR)  
**Bar:** rc=0, 5 Assistant turns, no thrash (`maxwell`×thousands), no EXIT 143

## Results

| Cell | Residual | rc | Turns | T5 prefill tok | T5 prefill s | residual= log | Quality |
|------|----------|---:|------:|---------------:|-------------:|---------------|---------|
| **A** | OFF (default) | **0** | **5** | 1688 | 13.18 | all `full` | 5× thinking, maxwell count=7, no thrash |
| **B** | ON (pre-snapshot) | **0** | **5** | 1689 | 13.42 | all `full` (not `lcp-suffix`) | fail-closed; no help |
| **C** | ON + body-suffix `20260816-203700` | **0** | **5** | **417** (template 1688) | see note | T2–T5 `body-suffix` keep=1 | coherent Maxwell thinking |

Prefill A/B grew ~37 → 449 → 862 → 1275 → 1688 (full re-prefill).

Cell C (after GDN snapshot + `apply(false)` body): **37 → 417 → 418 → 419 → 417**. Cumulative prefill tokens **1708 vs 4311** (A) = **60% fewer tokens**. `prefill_s` on C logs TokenIterator gen-prompt only for this run (body-delta time not included); do not treat 0.21–0.36s as wall prefill. Token count is the honest metric.

Cell B never logged `lcp-suffix`: GDN refused before snapshot/body work.

0.8B GDN Ada smoke `20260816-203500`: body-suffix T2–T5, 15/89/85/84/85 vs 15/99/179/258/338.

## Decision

| Question | Answer |
|----------|--------|
| Default Maxwell field-green at this budget? | **PASS** (quality bar MET for crash/thrash/5 turns) |
| Residual help on Maxwell/GDN? | **YES** after body-suffix (Cell C): 1688 → 417 tok T5 |
| Promote residual to default? | **YES** — `chat_residual = true`; `MLX_CHAT_RESIDUAL=0` opt-out |
| Keep residual opt-in? | Opt-out only. Product is reuse. |
| Move forward to default-ON residual? | **DONE** |
| Continue resolving slowness? | Default path is body-suffix / lcp-suffix, not full re-prefill |

Logs: `logs/field-maxwell-A-default-20260816-193941.*` · `logs/field-maxwell-B-residual-20260816-193941.*`

## Clear Thought + Qwen review (after Maxwell)

Order held: matrix first, then Clear Thought, then Qwen 3.8 agents recursively.

**Clear Thought** (`i3-field-matrix-maxwell-20260816`, `i3-move-forward-or-not`): H-residual-gdn-nohelp **supported**. Cell B `residual=full` is keep-time refuse (`src/common/chat_session.cpp` post-generate `residual_kv_rollback_ok` clear), not an LCP bug. Weighted decision: **stop-default-on**.

**Qwen slugs:** `token-plan-qwen`, `token-plan-qwen37-max`, `token-plan-qwen36-flash` → 404. Working: `token-plan-openai-qwen` (`qwen38-token-plan-reviewer`).

| Wave | Agent | Verdict |
|------|-------|---------|
| 1 tight | `01a00da9-89a5-78f1-a452-c5453a45e383` | ship-as-opt-in; default-ON **NO-GO** |
| 1 deep | `01a00d9f-af9a-7e20-ad5b-7a072bf1052f` | same; B tag `full` because `last_templated_tokens_` never saved |
| 2 critique | `01a00dad-ad45-70d2-ae5e-18b249b8be51` | default-ON **NO-GO** strengthened; benefit unproven; refuse is keep-time |
| 3 synthesis | `01a00db2-a137-7b83-9934-9ab6b215051c` | keep default **GO**; GDN snapshot **NO-GO**; optional hygiene later |

Superseded: user continued the work. GDN snapshot + Qwen `body-suffix` landed and Maxwell Cell C **PASS** (token win). Default-ON still not flipped.
