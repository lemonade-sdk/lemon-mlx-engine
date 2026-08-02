# 03 — Quality Review (Iter 1): Adversarial Audit of 01 + 02

**Role:** quality-reviewer · **Method:** attempt to REFUTE each pillar against code/raw logs (not armchair review) · clear-thought structured argumentation `arg-1785627271911`.

## Verdict

| Deliverable | Verdict | Rationale |
|-------------|---------|-----------|
| 01-strategist-iter1 | **CONDITIONAL PASS** | Findings are code-grounded and measurement-verified; 3 calibration defects + 1 new finding to absorb (D1–D4). No structural rework. |
| 02-program-plan-iter1 | **CONDITIONAL PASS** | Sequencing decision is sound and criteria-explicit; 2 estimation/sequencing defects (D5–D6). |

## A. Attacks executed and outcomes

| # | Attack | Method | Outcome |
|---|--------|--------|---------|
| AT-1 | "Pre-norm hidden return corrupts logits" (would make ALL MTP output wrong, invalidating the whole audit framing) | Read `qwen35_moe.cpp:1073-1085 call_impl` | **REFUTED** — `apply_norm(hidden)` feeds lm_head; only `State` carries pre-norm. Logits correct. (This check elevated to verified-fact in iter-2 appendix.) |
| AT-2 | "C7 27.34 is cherry-picked / not in raw logs" | grep Generation lines across 9 logs | **REFUTED** — 27.3409 raw; full ladder reproduced (26.1317/15.8665/19.7171/20.6388/22.3887); C13/C14/C15 confirmed, C15 undocumented (finding J upheld). |
| AT-3 | "Finding A overstates: temp=0 users unaffected" | Read `AnySampler::from_params` (generate.cpp:189-195) | **UPHELD with calibration (D1)** — temp≤0 → ArgMaxSampler ⇒ MTP≡eager at temp=0. Severity confined to temp>0 (still P0: silent contract violation), and… see D4. |
| AT-4 | "Finding B semantics claim could be wrong — MTP layer might be recurrence-only" | Read `MTPDecoderLayer::operator()` / `MTPDecoderLayerMoE::operator()` | **UPHELD as INFERENCE** — layers are full self-attention with RoPE offset from cache and `cache->update`; attention-over-draft-history loss at final step is real mechanics. Numerical parity test (W1.2) remains mandatory to quantify. |
| AT-5 | "Registry hazard theoretical — maybe manager lock covers it" | grep ModelManager for registry/lock interplay | **UNRESOLVED** — model_manager.cpp has zero registry references; load lock vs container mutex topology not traced end-to-end. Plan's W2.1 + TSAN exit criteria adequate; keep P1. |
| AT-6 | "T1 headroom derivation is made up" | Check MICROBENCH for lm_head split | **UPHELD as weakness (D2)** — no lm_head-cost microbench exists; 8–15ms is asserted. Plan mitigates (microbench-first) but strategist table confidence must drop to Low-Med and the +22% labeled *illustrative*. |
| AT-7 | "PM's 5-day rejection-sampling estimate is credible" | Check draft chain for retained distributions | **FAILED estimate (D3)** — `mtp_run_draft_chain` (generate.cpp:793-797) computes logits then immediately argmax-drops them; proper rejection sampling needs draft-probability plumbing through the whole chain + verify-side ratio test ⇒ realistic 8–12 d, or stay with greedy-error policy (which the PM already recommends for v1 — recommendation stands, estimate must). |
| AT-8 | "Finding H 'pure tax' wording" | Compare `forward_prenorm`+`apply_norm` vs `operator()` | **CALIBRATION (D4a)** — computation is identical split; n_draft=1 tax is negligible (state plumbing only). Reframe: *behavioral degeneration* (MTP enabled, zero speculation) + default mismatch, not perf tax. |

## B. NEW findings discovered during review (must enter iter-2 registers)

| ID | Sev | Finding | Evidence |
|----|-----|---------|----------|
| **D4** (N-1) | **P0-amend** | MTP path also bypasses `processor_` — `RepetitionProcessor::did_sample` is only called from `convert_to_token` (`generate.cpp:385-388`), never on the MTP path ⇒ **repetition penalty silently inert under --use-mtp** (same defect class as A; the field SAR harness even passes `--repetition-penalty 1.0`, masking it). Fold into W1.1 scope: "sampling + processing contract". | generate.cpp:385-388 vs 751-1438 |
| N-2 | P2 | Cross-model MTP parity UNVERIFIED: pre-norm-return verified only for `qwen35_moe`; `qwen3_next.cpp` (+18 lines) MTP support not checked for the same norm-for-logits discipline. | qwen3_next.cpp diff not read |
| N-3 | P2 | `MTPHead::operator()` concat order `[e_norm, h_norm]` is justified by a comment citing `qwen3_5.py:357` — a file not in this repo (upstream mlx-lm). If upstream reorders, silent accept collapse. Add a load-time shape/sanity assertion or doc link. | mtp_head.cpp:171-172 |
| N-4 | P2 | Plan gap: W1.3 golden tests do not cover the **norm-glue decision** (W1.4) — add synthetic-mean unit tests to G2. | 02 §3 WS1 |

## C. Coherence & consistency checks

- **01↔02 traceability:** every strategist finding (A–K) maps to a plan item — verified. T5/T6/T7/H1/H3 correctly parked, not silently dropped. ✔
- **HARD BAN compliance:** plan §1 inherits the ban; no seatbelt/auto-disable anywhere in recommendations; W3.5 declares ceiling instead. ✔
- **Gate dependency correctness:** G3 needs W1.1 — and after D4, W1.1 scope grows (processor too); critical-path estimate Fire 1 3.5 d → **~4.5 d** with the greedy-error policy (rejection sampling deferred per PM recommendation). Update plan math (D5).
- **KPI sanity:** "accept 0.85→0.90" — prompt-dependent (0.62–0.88 observed range!). KPI must pin the probe prompt or it's unmeasurable (D6).
- **Evidence hygiene:** strategist marked UNVERIFIED items honestly (registry topology, Metal, dense_kept=7). Reviewer adds N-2/N-3 to that list. ✔ discipline.

## D. Mandatory iter-2 actions (closure conditions)

1. **D1:** 01 finding A — confine severity statement to temp>0; note temp=0 parity via ArgMaxSampler.
2. **D2:** 01 T1 row — confidence Low-Med; "+15–22%" labeled illustrative pending lm_head microbench (add to evidence-needed).
3. **D3:** 02 W1.1 — estimates: greedy-error 1–2 d (v1, recommended); rejection sampling 8–12 d (follow-up, needs draft-distribution plumbing — state why).
4. **D4:** 01 finding H + N-1 — reframe n_draft=1 as behavioral; add processor bypass to finding A scope and W1.1.
5. **D5:** 02 — recompute Fire 1/critical path with amended W1.1; add pre-decision item to Fire 0 (★Q1 must be answered before Fire 1 starts, not during).
6. **D6:** 02 KPI — pin accept-rate KPI to the documented Fourier-style probe prompt + n≥3.
7. **N-2/N-3/N-4:** append to registers (01 §2 + 02 W1.3/W1.4 scope).

## E. What the analysis got RIGHT (keep for retrospective)

- Raw-log verification culture (this branch's strongest asset — the docs even self-falsify C6 attribution; rare intellectual honesty).
- Physics-first ceiling framing prevented a 3-week chase of a fake 100 t/s.
- Kill-criteria on every TPS lever (prevents sunk-cost experiments like C11–C14 recurring).

**Reviewer disposition:** proceed to iter-2 calibration pass; after D1–D6 applied, this package is ready for stakeholder review. No further full-loop iteration required if edits are mechanical (reviewer will spot-check, not re-audit).
