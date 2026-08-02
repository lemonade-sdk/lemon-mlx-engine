# 02 — Program Plan (Iter 1): MTP Remediation & TPS Program

**Role:** software-program-manager · **Input:** `01-strategist-iter1.md` · **Decision basis:** clear-thought weighted-criteria evaluation (`mtp-program-sequence`) → **correctness-first sequencing**.

---

## 1. Charter

**Goal:** make MTP on `fix/mtp-stream-p0` *correct-by-construction, quality-proven, and at its honest software ceiling* on gfx1150/35B (~31–35 t/s), with the 100 t/s bar closed via the documented H2 target and an explicit hardware-ceiling declaration for 35B.

**In scope:** the speculative path (`generate.cpp` MTP), `mtp_head/mtp_moe`, quant-registry lifecycle, defaults/UX, docs/metrics integrity, test harness.
**Out of scope (tracked as decisions, not work):** H1 hardware procurement; H3 continuous-batching architecture (needs its own program — the per-model mutex in `model_container.h` makes it multi-week).
**Inherited HARD BAN (from MTP_OPTIMALITY_PLAN):** no LoopBrake/seatbelts, no auto-disable, no silent eager fallback, no invented numbers. Every "win" ships with a probe log under `docs/experiments/mtp-stream-p0/` and a row in CRITICAL_ANALYSIS.

## 2. Operating principles

1. **Evidence discipline upgrade:** "wins" require **n≥3 runs, same prompt, accept rate reported**; single-run rows are labeled *probe* not *result* (kills the C6-attribution and C8-inside-noise problems).
2. **One ladder doc:** CRITICAL_ANALYSIS.md is the single source of truth; every fire appends a row (closes finding J — C13/C14/C15 backfill is Fire 0 work).
3. **Correctness before speed:** no TPS change merges while a P0 correctness item is open on the same path.
4. **Delete, don't default-off:** measured regressions leave the tree (branch archives are acceptable).

## 3. Workstreams & backlog

### WS1 — Correctness & Quality (priority 1)

| ID | Item | Src | P | Dep | Est. | Exit criteria |
|----|------|-----|---|-----|------|---------------|
| W1.1 | **Sampling & processing contract for MTP** — hard-reject temp>0 / top_p<1 / rep-penalty≠1 with explicit error (v1), **and** cover `processor_` bypass (rep-penalty inert under MTP — iter-2 N-1); rejection sampling as funded follow-up only | A+N-1 | P0 | ★Q1 | **1–2 d (v1 error path)** / **8–12 d (rejection follow-up — draft chain argmax-drops logits at generate.cpp:793-797, so draft-distribution plumbing + verify-side ratio test + per-token processor callbacks are all new work; iter-2 D3 corrected estimate)** | Contract test: temp=0.7+`--use-mtp` errors cleanly; temp=0 output identical to eager (ArgMaxSampler parity); rejection-sampling follow-up has its own gate |
| W1.2 | **Fix γ>1 draft cache** — pass MTP KV on final draft step; parity test draft logits | B | P0 | — | 1.5 d | Unit: logits(n_draft=3, fixed cache) ≠ logits(current) and match HF-reference expectation; n_draft=3 re-measured |
| W1.3 | **Golden-vector state-machine tests** — stub call_fn/head, scripted accept/reject/EOS-mid/rollback scenarios asserting tokens + cache positions; **+ norm-glue synthetic-mean tests (iter-2 N-4)** + qwen3_next logits-parity probe (iter-2 N-2/L) | D+N-4 | P0 | W1.2 | 2.5 d | ≥10 scenarios green in CI on CPU; batch + sequential + pipeline covered; glue tested at means {−0.3, 0.15, 0.49} |
| W1.4 | **Norm-glue hardening** — per-tensor mean test (not global), config override `MLX_MTP_NORM_SHIFT=auto|on|off`, log means of all norms pre-decision | A(5.3) | P1 | — | 1 d | guru87 + 35B + mlx-community-4B all load with correct accept; synthetic mean=0.15 shifted head doesn't double-shift |
| W1.5 | **Quality SAR matrix** — 5-turn Maxwell, temp 0 + temp 0.7 (post-W1.1), n≥3 seeds, n_draft=2, fuse on/off; PASS = coherent Python, no thrash, EXIT 0 | E | P0 | W1.1 | 2 d | All cells PASS or defect filed; RESULTS stamped |

### WS2 — Robustness & Lifecycle (priority 2)

| ID | Item | Src | P | Dep | Est. | Exit criteria |
|----|------|-----|---|-----|------|---------------|
| W2.1 | **Registry lifecycle** — unregister model-owned pointers on unload/evict; registry mutex or proven single-writer; leak test (load→unload→load cycle, registry size stable) | C | P1 | — | 2 d | Leak test green; ASAN/TSAN clean on cycle; multi-model server smoke |
| W2.2 | **Apple parity decision** — either run a Metal MTP measurement day (eager + C7 configs) or doc-gate MTP as ROCm-experimental; fix StreamGuard side-stream on Apple if measured worthwhile | F | P1 | — | 1–3 d | Documented decision + evidence or doc gate; README accurate |
| W2.3 | **pending_v1 invariant doc + guard** — assert cache-continuous turns are impossible while v1 pending; comment the re-prefill safety contract | A(5.5) | P2 | — | 0.5 d | Invariant comment + debug assert |

### WS3 — TPS Performance (priority 3, gated on WS1)

| ID | Item | Src | P | Dep | Est. | Exit criteria |
|----|------|-----|---|-----|------|---------------|
| W3.1 | **Accept lift** (γ>1 fix fallout + fc-BF16 A/B + draft audit) | T2 | P1 | W1.2 | 1.5 d | accept 0.85→≥0.90 @ γ=1 (n≥3) or kill; ladder row |
| W3.2 | **Draft lm_head cut** — microbench first; vocab-slice/two-stage top-k; device argmax kept | T1 | P1 | W3.1 | 5 d | joint −≥6 ms and gen ≥30 t/s (n≥3) or kill at microbench <4 ms |
| W3.3 | **Dense audit** — identify `dense_kept=7`, force gather_qmm where aligned | T4 | P2 | — | 1 d | Per-linear dump; quantize or justify each; Δ measured |
| W3.4 | **C15 triage** — re-derive device-accept result on C7 base; keep or delete | T3 | P2 | — | 0.5 d | Ladder row either way |
| W3.5 | **Plateau declaration** — when W3.1–3.4 stop yielding ≥2%, stamp software ceiling ~31–35 and close single-seq 35B bar as FAIL-with-ceiling (HARD-BAN-compliant) | plan | — | W3.1-4 | 0.5 d | Signed stop-checklist in MTP_OPTIMALITY_PLAN §0.6 |

### WS4 — Hygiene (parallel, low risk)

| ID | Item | Src | P | Dep | Est. | Exit criteria |
|----|------|-----|---|-----|------|---------------|
| W4.1 | **Defaults:** chat + server n_draft → 2; drop "scaffolding" wording | H | P1 | — | 0.5 d | Both binaries default 2; help text accurate |
| W4.2 | **Delete 6 regression paths + prefetch loop** (or gate behind `MLX_MTP_EXPERIMENTS=1`) | G | P2 | W1.3 (tests exist first) | 1.5 d | −~250 LOC; env census ≤ 8; CI green |
| W4.3 | **Metrics/doc backfill** — C13/C14/C15 rows, sequential-β section, proposed/accepted counter fix, KV-offset log on MTP path | I,J | P2 | — | 1 d | Docs consistent with raw logs; counters correct |
| W4.4 | **Fuse memory:** unregister+release originals after MTP QKV fuse | K | P2 | W2.1 | 0.5 d | Memory delta verified when fuse opt-in |

## 4. Gates & sequencing

```
Fire 0 (1 d):   W4.3 doc backfill + plan review
                + ★ PRE-DECISIONS: Q1 sampling policy, Q4 multi-model scope,
                  Q6 Apple claims, Q7 stop-bar wording  (iter-2 D5: before Fire 1, not during)
                + 100-bar ceiling memo escalated to stakeholders (R6)        ─┐
Fire 1 (4.5 d): W1.1(v1 error path) + W1.2 + W1.4    → G1 correctness       ─┤  (iter-2 D5 recompute)
Fire 2 (2.5 d): W1.3 golden tests (incl. N-2/N-4)    → G2 test gate         ─┤
Fire 3 (2 d):   W1.5 SAR matrix                      → G3 quality gate      ─┤  (needs W1.1)
Fire 4 (2 d):   W4.1 + W2.1 + W2.3                   → G4 robustness        ─┤  (parallel with Fire 3)
Fire 5 (7 d):   W3.1 → W3.2 (seq; microbench-gated) + W3.3/W3.4 (parallel)  ─┤
Fire 6 (1 d):   W3.5 + W4.2 + W4.4                   → G5 ship/declare      ─┘
Rejection-sampling follow-up (8–12 d): only if ★Q1 follow-up funded — separate mini-program, NOT on critical path.
```

| Gate | Entry | Exit |
|------|-------|------|
| **G1 correctness** | W1.1/W1.2/W1.4 done | sampling contract enforced; γ>1 parity test green; glue hardened |
| **G2 tests** | G1 | golden vectors in CI; batch+seq+pipeline covered |
| **G3 quality** | G1 | SAR matrix all cells PASS @ temp 0 and (working) 0.7 |
| **G4 robustness** | W2.1 | registry leak test + TSAN clean; Apple decision recorded |
| **G5 ship/declare** | G1–G4 + WS3 | n≥3 ladder stamp; plateau declared; 100-bar closed via H2 + ceiling memo |

**Critical path:** ★Q1 decision → W1.1 → W1.5 → G3 → (WS3) → G5. Total ≈ **18.5 fire-days ≈ 3.5–4 focused weeks** single-engineer (iter-2 D5 recompute: Fire 0 +1 d pre-decisions, Fire 1 +1 d amended W1.1, Fire 2 +0.5 d added tests); ~2.5 weeks with two engineers (WS2/WS4 parallel).

## 5. Risk register

| # | Risk | L×I | Mitigation |
|---|------|-----|------------|
| R1 | Rejection sampling costs TPS (extra evals) | M×M | Budget ≤5% TPS; else ship greedy-only-with-error (policy fallback decided up front) |
| R2 | γ>1 fix *raises* n_draft=3 perf and re-opens deep-draft scope creep | M×L | Timebox: one re-measure; if <27.34 @ n=3, freeze γ=1 as product config (W3.5) |
| R3 | SAR matrix fails at temp 0.7 even with correct sampling | M×H | Then MTP ships temp=0-only with contract error; do NOT seatbelt (HARD BAN) |
| R4 | Registry mutex adds per-forward latency | L×M | Reader-writer lock or prove single-writer via manager lock; bench before/after |
| R5 | Single-run "wins" recur under schedule pressure | M×M | Gate rule: no ladder claim without n≥3 — enforced in CR template |
| R6 | 100-bar stakeholder rejects ceiling declaration | M×M | Early escalation (Fire 0 memo) with H1/H2/H3 options menu; H2 already MET as fallback |
| R7 | Deletion of regression paths removes future research footholds | L×L | Archive to `experiments/` git tag before delete |

## 6. Decision log (strategist's open questions — PM recommendations; ★ = needs stakeholder ratification)

| # | Question | PM recommendation |
|---|----------|-------------------|
| 1 | Sampling policy | ★ **Hard-error at temp>0 for v1** (1 d) + rejection sampling as follow-up if product needs sampled MTP. Rationale: correctness now, cost later. |
| 2 | Server default n_draft 3→2 | **Yes, ship now** (W4.1) — measured, no downside. |
| 3 | Regression-path deletion | **Archive tag + delete** (W4.2) after G2. |
| 4 | Multi-model server support | ★ If supported → W2.1 is P0-upgrade and must precede any "server GA" claim; registry hazard is real. |
| 5 | Quality bar definition | **Adopt W1.5 matrix as the bar** (5-turn Maxwell × {0,0.7} × 3 seeds × fuse{on,off}; PASS = coherent + EXIT 0). |
| 6 | Apple claims | ★ Default: **doc-gate** (MTP = ROCm-experimental) unless a 1-day Metal probe shows parity; avoids unfounded README claims. |
| 7 | Stop-bar reconciliation | **Accept documented-target wording**: H2 0.8B MET (5 runs); 35B FAIL-with-ceiling memo at G5. Escalate at Fire 0, not G5 (R6). |
| 8 | Deep draft after B fix | **One measured re-try** (R2); expect freeze at γ=1. |

## 7. KPIs & reporting

- **Gen t/s (35B, n≥3 mean±std, pinned probe)**: baseline 27.34 → target ≥31 stretch 35. **Probe discipline (iter-2 D6):** the documented Fourier-style 256-tok prompt, full quant fuse, temp=0 — same binary flags as `C7_TPS_probe_ndraft2.txt`; any other prompt = a different KPI, labeled as such.
- **Accept rate** γ=1 on that pinned prompt: 0.85 → ≥0.90 (n≥3 runs).
- **P0 open count**: 2 → 0 by G1.
- **MTP env-var census**: 19 → ≤8.
- **Test scenarios**: 3 → ≥11.
- **Doc↔raw-log consistency**: 100% (audit script: grep Generation lines vs CRITICAL_ANALYSIS table — add to CI as W4.3 stretch).
- Weekly: ladder-diff report (rows added, gates passed, kills declared).

## 8. Definition of Done (program close)

1. G1–G5 green; probe logs + n≥3 rows in CRITICAL_ANALYSIS.
2. MTP_OPTIMALITY_PLAN §0.6 stop-checklist stamped: H2 MET, 35B ceiling declared, no invented numbers.
3. README/help text match behavior (defaults, platform support, sampling contract).
4. No default-off measured-regression code paths in `src/`.
5. This program's retrospective appended (what the C-ladder discipline got right — keep it).
