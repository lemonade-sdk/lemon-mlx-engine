# 04 — Final Synthesis: MTP Branch Critical Audit (CONVERGED)

**Process:** recursive role chain `planning-analysis-strategist → software-program-manager → quality-reviewer`, 2 iterations, clear-thought MCP reasoning throughout (sequentialthinking ×2, metacognitivemonitoring, decisionframework, structuredargumentation).
**Note:** subagent infrastructure was unavailable this session (provider rejected all Agent launches: `enable_thinking` restriction, 5/5 attempts across agent types and model overrides). Roles were executed role-isolated in the main loop with identical deliverables; every adversarial check was run against code/raw logs, not asserted.

**Artifacts:** `01-strategist-iter1.md` (calibrated) · `02-program-plan-iter1.md` (calibrated) · `03-quality-review-iter1.md` · this file. All under `docs/analysis/mtp-review/`.

---

## Fire 1 status (P0 land)

**Shipped on `fix/mtp-stream-p0`:** P0-A hard-error contract (`mtp_greedy_only_violation` + `TokenIterator` throw when MTP actually runs; chat coerces unset temp/top_p on `--use-mtp`; server CLI defaults greedy + HTTP 400 on non-greedy MTP after request merge); P0-B final draft step now receives MTP KV whenever `use_mtp_kv` (n_draft≥3); chat+server **n_draft default 2**. Greedy-only MTP v1 documented in CLI help. Full rejection sampling remains a funded follow-up (not Fire 1).

## Bottom line

**Issues with MTP: YES — 2 P0, 4 P1, 9 P2 (incl. 2 added in iter-2).** The branch is otherwise the most measurement-disciplined work in this repo (60+ probe logs, self-falsifying docs, HARD BAN respected).

| # | Finding | One-liner |
|---|---------|-----------|
| **A** | **P0** | `--use-mtp` silently forces greedy and ignores `temperature`/`top_p`/**`repetition_penalty`** at temp>0 (sampler+processor bypassed; `generate.cpp:385-388` vs 751-1438). Temp=0 is clean (ArgMaxSampler parity). All temp-0.7 quality evidence is confounded. |
| **B** | **P0** | γ>1 draft chain drops MTP self-attention history on the final draft step (`generate.cpp:785-798` nullptr cache) — suppresses deep-draft accept; **server default n_draft=3 runs this on step 1**. C9's "n_draft=3 regresses" may be this bug, not physics. |
| C | P1 | `QuantizedWeightRegistry` global, never cleared on unload — leak + stale-pointer hazard; thread-safety UNVERIFIED. |
| D | P1 | 690-line speculative state machine, 3 shape tests, 0 logic tests. |
| E | P1 | Quality bar unmet: sole MTP Maxwell SAR ended EXIT:143, with inert sampling. |
| F | P1 | Zero Metal measurement; StreamGuard Apple no-op silently kills side-stream overlap. |
| G–M | P2 | 19 env vars (6 gate dead regressions), degenerate defaults, metric anomalies, undocumented C15 regression (25.33), doc drift, fuse memory doubling, qwen3_next parity UNVERIFIED, upstream concat-order coupling. |

**Reviewer-verified non-issues (do not chase):** pre-norm hidden does NOT corrupt logits (`call_impl` norms for logits — AT-1); cross-request graph-decode-mode leak does NOT exist (prefill resets); KV position math on the sequential accept/reject path is CORRECT (traced).

## TPS verdict (honest)

- **Verified today:** eager 26.13 · MTP C7 **27.34** (both raw-log confirmed) · 0.8B+MTP **100.045** (5 runs).
- **Realistic software destination on gfx1150/35B:** **~31–35 t/s** via fix-B → accept lift → draft lm_head cut (microbench-gated; +22% claim is illustrative until measured) → dense audit. Then declare plateau.
- **100 t/s on 35B single-seq: impossible** (β≈1.5–1.7, bound ≤2× ⇒ ≤~52). Close the bar as: **H2 MET (0.8B, documented target) + 35B FAIL-with-ceiling memo.** H1 (dGPU) and H3 (continuous batching — architecturally blocked by the per-model mutex; separate program) are the only ≥100 paths.

## Program (correctness-first; clear-thought decision `mtp-program-sequence`)

Fire 0 pre-decisions (sampling policy ★, multi-model scope ★, Apple claims ★, stop-bar wording ★, ceiling memo) → Fire 1 G1 correctness (A-error-path 1–2 d, B fix, glue hardening) → Fire 2 G2 golden tests (≥10 scenarios + glue + qwen3_next probe) → Fire 3 G3 SAR matrix (temp 0 + working 0.7, n≥3) → Fire 4 G4 robustness (registry lifecycle, TSAN) → Fire 5 TPS levers with kill-criteria → Fire 6 G5 ship/declare. **~18.5 fire-days (~3.5–4 weeks solo).** Rejection sampling (8–12 d) only if product funds sampled MTP — separate mini-program.

## Residual honest limits (tracked as work, not analysis gaps)

1. Registry lock topology — verified under W2.1 + TSAN.
2. Finding B quantification — W1.2 parity test.
3. Metal behavior — W2.2 measurement day or doc-gate.
4. lm_head cost share — Fire 5 microbench gates T1.

## Convergence record

Iter-1 reviewer issued 7 demands (D1–D6 + N-series) → all applied in iter-2 edits to 01/02 (traceability: 03 §D ↔ edit log) → iter-2 spot-check found no new contradictions → **loop closed**. Package is stakeholder-ready.
