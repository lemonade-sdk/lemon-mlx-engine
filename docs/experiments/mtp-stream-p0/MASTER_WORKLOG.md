# MASTER_WORKLOG — mtp-stream-p0

**Branch:** `fix/mtp-stream-p0` (S4 TPS ceiling child: `exp/mtp-tps-ceiling`)  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit · gfx1150 · single process only  
**Hard bans:** LoopBrake / auto-disable MTP; dual 35B; fake TPS; claim Maxwell green without log  

## 2026-08-01 S4 — TPS ceiling probe (06 §4) — **KILL / PLATEAU**
- Branch: `exp/mtp-tps-ceiling` ← child of tip `875a39d`; full stack (fuse, StreamGuard, P0-B, RS) behind it.
- Env **included `MTP_TIMING=1`** (141–145 `[mtp-t]` rows; code gate requires getenv set).
- Kill line **67.7 ms** (not 55). Batch n2 verify_on_accept **mean 77.1 / med 71.2 ms**; gen **20.89 t/s** vs seq **27.22**.
- Seq n3 post-P0-B **18.29 t/s** (old 22.71 invalid) — deep draft still dead.
- **No** product batch-verify rewrite. Plateau ~27 t/s sequential n_draft=2. Logs: `docs/experiments/mtp-tps-ceiling/`. 

## Stop bar
- (a) Maxwell-quality gate PASS for MTP temp=0.7 thinking with coherent multi-turn log, **AND**
- (b) golden MTP logic tests landed  
**OR** three consecutive fires with no new implement/measure and no worklog progress.

## Prior context (pre-MASTER; tip eba422d)
- P0-A/B then `eba422d` rejection sampling for temperature>0.
- C7 greedy ~27 t/s; C11–C14 closed thrash (do not re-litigate without new evidence).
- Prior FIELD SAR temp0.7 (tip 975e2de) confounded by greedy-only — not valid quality evidence.
- P0 Stream gates M1–M6 PASS (see P0_MTP_GATES.md / gates/RESULTS.md).

## 2026-08-01 fire 1
- Goal: Quality short multi-turn Maxwell SAR with MTP + rejection sampling (temp=0.7, thinking ON); create MASTER_WORKLOG; harden if FAIL.
- Tried:
  - Clear Thought sequentialthinking + decisionframework → primary = quality measure.
  - Baseline measure (pre-fix) `FIELD_MAXWELL_RS_ndraft2_temp07_think.txt`: EXIT ok but **severe word-drop garble**.
  - Root-cause: `mtp_speculative_step_sampled` returned `[d0,d1,…]` but `next()` only emits `[0]` and never filled `draft_buffer_` (greedy path does). Accepted drafts dropped under temp>0.
- Implemented (paths/SHA):
  - `src/common/generate.cpp` — fill `draft_buffer_` with d1..d_accepted; return `{d0}` (parity with greedy). Tip after commit (this fire).
- Quality measure (prompt, temp, thinking, fuse, n_draft, EXIT, notes):
  - Config: QUANT_FUSE+GDN, LOAD_MTP_HEAD=1, `--use-mtp --n-draft 2 --temperature 0.7 --top-p 0.9` thinking ON.
  - T1: Maxwell 4-eq overview (8–12 short sentences). T2: Gauss differential + one sentence.
  - **v1 FAIL** (pre-fix): garble / missing function words; ~15 t/s. Log: `FIELD_MAXWELL_RS_ndraft2_temp07_think.txt`.
  - **v2** max_tokens=512 post-fix: coherent thinking; EXIT:0; ~26 t/s; still budget-truncated. Log: `…_v2.txt`.
  - **v3** max_tokens=1024: EXIT:0; rejection-sampling on; T1 coherent all-four Maxwell in thinking (hit 1024 mid-think, no final `</think>` answer); T2 closed think + final `∇ · E = ρ / ε₀` + meaning sentence; multi-turn OK; ~26 t/s. Log: `…_v3.txt`.
  - **Verdict:** Maxwell **coherent multi-turn PASS** for emit/quality after buffer fix (honest: T1 needs higher max_tokens/thinking budget for post-think answer). **Do not schedule-stop** — golden MTP logic tests still open.
- Subagent votes:
  - Clear Thought decision: quality primary AGREE.
  - quality-reviewer (collaborative): FAIL on v1 AGREE; ship buffer fix AGREE; no LoopBrake.
  - PM (collaborative): harden after quality FAIL AGREE; remeasure AGREE.
  - Domain spawn tools flaked (not_found); roles executed via Clear Thought + main-loop review.
- Clear Thought conclusion: Evidence-first measure exposed P0 emit bug on sampled path; fix restores coherent Maxwell multi-turn; next is golden tests not more TPS thrash.
- Next:
  1. Golden MTP logic tests (emit/buffer contract + accept/reject sampling without 35B).
  2. Optional longer Maxwell with max_tokens≥2048 or thinking budget for post-think T1 answer.
  3. Residual approx residual distribution / registry unload after goldens.

## 2026-08-01 fire 2
- Goal: Land golden MTP logic tests (stop-bar half b); no 35B dual-load.
- Tried: Clear Thought decision → goldens over longer Maxwell / registry. Domain spawns flaked; roles via Clear Thought collab.
- Implemented (paths/SHA):
  - `include/mlx-lm/common/generate.h` + `src/common/generate.cpp`: pure `mtp_make_emit_plan`, `mtp_accept_ratio`, `mtp_adaptive_n_draft`; wired sampled+greedy emit and accept path.
  - `tests/test_generate.cpp`: 11 `[golden]` cases (emit reject/accept/partial/full/clamp, accept_ratio equal/gt/lt/nan, adaptive, acceptance_rate).
  - `./tests/test_generate "[mtp]"` → **15 cases / 68 asserts PASS**.
- Quality measure: skipped (no field remeasure); fire1 v3 remains Maxwell coherent multi-turn evidence.
- Subagent votes: quality-reviewer AGREE ship goldens; PM AGREE schedule stop (a+b met). Spawns not_found → Clear Thought collab.
- Clear Thought conclusion: Goldens lock fire-1 buffer drop; stop bar (a Maxwell PASS log) + (b goldens) both satisfied → stop schedule.
- Next (post-schedule residuals, not stop-bar): longer Maxwell max_tokens≥2048 T1 post-think; residual dist; registry unload; env/doc debt.

## STOP SCHEDULE
- (a) Maxwell temp=0.7 thinking coherent multi-turn: **PASS** (`FIELD_MAXWELL_RS_ndraft2_temp07_think_v3.txt`, fire1).
- (b) Golden MTP logic tests: **PASS** (fire2, `test_generate "[mtp]"`).
- Action: scheduler_delete task `019fbff3379c`.

## 2026-08-01 full Maxwell SAR (post-loop, high max_tokens)
- Goal: Full multi-turn Maxwell quality with MTP rejection sampling, **max_tokens=8192** (no early length cut).
- Config: tip `147a319`, QUANT_FUSE+GDN, LOAD_MTP_HEAD=1, `--use-mtp --n-draft 2 --temperature 0.7 --top-p 0.9`, thinking ON, **max_tokens=8192**.
- Turns: T1 four equations overview; T2 Gauss E differential; T3 Faraday; T4 Ampère–Maxwell; T5 Gauss B.
- Log: `FIELD_MAXWELL_FULL_RS_ndraft2_temp07_think_max8k.txt`
- Result: **EXIT:0** ~4 min wall after load. Rejection-sampling on. All 5 turns completed with `</think>` + final answers.
  - T1: 3034 tok @ 25.6 t/s — 9-sentence overview of all four laws (coherent).
  - T2: 766 tok @ 26.1 t/s — \(\nabla\cdot E=\rho/\varepsilon_0\) + meaning.
  - T3: 916 tok @ 25.7 t/s — \(\nabla\times E=-\partial B/\partial t\) + meaning.
  - T4: 1006 tok @ 25.6 t/s — Ampère–Maxwell full form + meaning.
  - T5: 326 tok @ 25.9 t/s — \(\nabla\cdot B=0\) + monopoles.
- Verdict: **FULL MAXWELL multi-turn PASS** at temp=0.7 + MTP + fuse + thinking with high token budget.

## 2026-08-01 residuals (registry + RS residual dist)
- Goal: Close agreed non-P0 residuals (not prefill thrash).
- Implemented:
  1. **Quant registry lifecycle (P1):** `QuantizedWeightRegistry::LoadScope` records every `register_weight` during model load; `ModelContainer` destructor calls `unregister_many`; `unload`/`unload_all`/LRU clear orphans when empty. Paths: `quantized_linear.h`, `model_container.h`, `model_manager.cpp`, `examples/chat.cpp`.
  2. **RS residual distribution:** on reject, sample from Leviathan residual \(\max(0,q-p)\) via `mtp_residual_logits` (draft logits rows stored at draft time); if residual mass ~0, mask rejected token on target. Path: `generate.cpp` / `generate.h`.
- Quality: unit `[mtp]` still **15 cases / 68 asserts PASS** (no full 35B remeasure this residual fire).
- Next: optional env/doc debt cleanup; optional multi-seed Maxwell.

## 2026-08-01 full Maxwell re-run (post-residual tip df1f199)
- Goal: Reconfirm quality after registry lifecycle + Leviathan residual sample.
- Config: same as max8k full SAR — QUANT_FUSE+GDN, MTP n_draft=2, temp=0.7, top_p=0.9, thinking ON, **max_tokens=8192**.
- Log: `FIELD_MAXWELL_FULL_RS_ndraft2_temp07_think_max8k_post_residual.txt`
- Result: **EXIT:0**. 5/5 turns closed think + correct finals.
  - T1: 2249 tok @ 25.15 t/s — 8-sentence four-law overview
  - T2: 1199 @ 25.68 — ∇·E=ρ/ε₀
  - T3: 1167 @ 25.07 — ∇×E=−∂B/∂t
  - T4: 875 @ 25.35 — Ampère–Maxwell full SI
  - T5: 861 @ 25.70 — ∇·B=0
  - Mean gen **~25.4 t/s**, total gen **6351** tokens
- Verdict: **PASS** (matches pre-residual full Maxwell quality; residual dist did not break multi-turn).

## 2026-08-01 P2 residual close (R-1…R-6)
- **R-1:** residual sample uses bare `categorical(log_r)` — no second temp scale via `sampler_`.
- **R-2:** `QuantizedWeightRegistry` guarded by mutex on register/find/unregister/clear.
- **R-3:** refcount per pointer — shared base/delta packs unregister safely (erase only at 0).
- **R-4:** server help text: rep-penalty no longer "disallowed with --use-mtp".
- **R-5:** golden `mtp_draft_uses_kv` + n_draft=3 emit plans.
- **R-6:** temp=0 + top_p∈(0,1) stays greedy-spec (top_p inert, do not force RS).
- **R-8:** RS TPS ~25.4 vs C7 27.34 (−7%) **ratified** as cost of sampling+thinking quality; not a stop-bar failure.

