# MASTER_WORKLOG — mtp-stream-p0

**Branch:** `fix/mtp-stream-p0`  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit · gfx1150 · single process only  
**Hard bans:** LoopBrake / auto-disable MTP; dual 35B; fake TPS; claim Maxwell green without log  

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

