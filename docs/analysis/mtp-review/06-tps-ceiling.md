# 06 — TPS Ceiling Analysis: Why MTP ≈ Eager on gfx1150, and the One Lever Left

**Scope:** MTP throughput only (single-stream 35B, gfx1150/890M). Companion to `05-p0-review.md`.
**Question:** MTP gives 27.34 t/s vs eager 26.13 (+4.6%) — is further code at fault, and is there any path to a real MTP payoff on this machine?
**Method:** raw-log decomposition (`[mtp-t]` rows in C6/C3 probes), arithmetic ceiling proof, C-ladder archaeology; Clear Thought MCP (sequentialthinking ×2, decisionframework `mtp-tps-next-action`).
**Verdict:** **No remaining TPS bug. The +4.6% is the arithmetic ceiling under sequential verify — proven below — and the sequential design is a *measured* decision, not an oversight. One untried measurement (batch verify on the post-fuse stack) can either reopen the math to ~+40% or close the question forever.**

---

## 1. Evidence base (all figures raw-log-sourced)

| Quantity | Value | Source |
|----------|-------|--------|
| Eager gen (35B, pinned probe) | 26.1317 t/s ⇒ **T₁ ≈ 38.3 ms/token** | CRITICAL_ANALYSIS ladder / synthesis 04 |
| MTP C7 (n_draft=2, greedy) | **27.3409 t/s** ⇒ effective wall ≈ 67.7 ms/step at E[tokens]=1.85 | C7_TPS_probe_ndraft2.txt |
| Verify cost per trunk call (n=2) | **35.4–35.9 ms** (= T₁; per-token linear) | C6_TPS_probe_ndraft2.txt `[mtp-t]` rows |
| Pure draft step (warm) | ~10–12.5 ms/step | C3 probe rows: n=3 draft=24.5ms /2 steps; n=4 30.4ms /3 steps |
| Batch T=2 verify (C1 era) | **≈ 86 ms = 2.26× T₁** — zero amortization | `generate.cpp:1285-1289` (C2 decision comment, field datum) |
| n_draft=3 (pre-P0-B) | 22.71 t/s — measured under the final-step KV-starvation bug | D3 ladder rows (datum invalidated by P0-B) |
| Sampled RS path (temp 0.7) | ~25.4 t/s (−7% vs greedy; serial T=1 verify, per-position logprob evals) | MASTER_WORKLOG post-residual rows |
| C6 timing sample | `step=2 accepted=1 draft=52,547us verify=35,398us total=87,960us` (draft timer includes the C4 join wait; pure draft ≪ 52ms) | C6_TPS_probe_ndraft2.txt |

## 2. The arithmetic ceiling (why acceptance rate can't save us)

Verification is a **sequential T=1 loop** (`generate.cpp:1316` dispatch; `1489-1515` loop body). Each accepted draft requires its own full trunk pass:

```
E[tokens/step] = 1 + p                              (p ≈ 0.85 accept rate)
E[wall/step]   = T₁ + p·T₁ + max(0, D − overlap)    (D = draft, hidden by C4)
               ≈ (1+p)·T₁     when D ≤ T₁
speedup        = (1+p)·T₁ / ((1+p)·T₁ + unhidden)  ≤ 1 + overlap margin
```

**(1+p) cancels.** Under sequential verify, MTP's theoretical speedup is **1.000** — every token MTP emits costs exactly one eager-priced trunk pass. The observed +4.6% is entirely the overlap machinery (C4 side-stream draft under verify; C8 async residual eval under host emit). Two corollaries, both confirmed by the branch's own history:

1. **Acceptance-rate lifts are inert**: 0.85 → 0.95 changes nothing in the ratio. The Fourier-probe accept KPI (plan 02 §7) measures health, not upside.
2. **Draft-side optimization is inert**: C11 (top_k), C13 (QKV fuse), C14 (shared-expert skip), C15 (device accept) all regressed or flatlined — they shaved a term that is not on the critical path. The C-ladder didn't fail; it correctly mapped the wall.

## 3. Why sequential verify is the *default*, not a bug

`generate.cpp:1285-1289`:
> "Default (C2): sequential T=1 verify — uses ROCm graph-decode mode and fused GDN T=1 path; early-exit on mismatch (no capture_spec tax, no multi-token gated_delta_update_seq). Field: multi-token verify was the residual after C1 quant draft (~86ms verify vs ~38ms eager T=1). Opt into old batch verify: `MLX_MTP_BATCH_VERIFY=1`."

Batch verify was **built (C1), field-measured (86 ms for 2 tokens = 2.26× T₁), and deliberately replaced (C2)**. Physics of the 2.26×:

- **8 CUs, launch/dispatch-bound**: decode is dominated by per-token fixed costs (MoE expert routing kernel launches, small per-expert matmuls), so batching 2 tokens replays the fixed costs twice instead of amortizing a bandwidth read.
- **Hybrid GDN recurrence** (per the C2 rationale): gated-delta layers update recurrent state sequentially per token — multi-token verify pays `gated_delta_update_seq`, which is linear in token count by construction. No batch design escapes this on this architecture.
- Sequential also buys **early exit on mismatch** (reject ⇒ stop verifying; batch pays for all K tokens even on reject at position 1).

The old batch path remains compiled behind `MLX_MTP_BATCH_VERIFY=1` — which is what makes §4 possible.

## 4. The one untried lever: re-probe batch verify on the current stack

The 86 ms datum is **C1-era**. Since then the stack gained quant-fuse (+GDN-fuse opt-in), graph-decode mode maturity, and the P0-B draft-cache fix. **Status: MEASURED 2026-08-01 on `exp/mtp-tps-ceiling` (child of `fix/mtp-stream-p0`).**

### 4.0 Field result — **KILL** (plateau stamped)

Full write-up: [`docs/experiments/mtp-tps-ceiling/RESULTS.md`](../../experiments/mtp-tps-ceiling/RESULTS.md).  
**MTP_TIMING was on** (`MTP_TIMING=1` in env; 141–145 `[mtp-t]` rows per n_draft=2 log — code only prints those when `getenv("MTP_TIMING")` is set).

| Config | gen t/s | Batch verify on accept (mean / median) | Warm mean total ms | Decision |
|--------|---------|----------------------------------------|--------------------|----------|
| **Seq n_draft=2** (baseline) | **27.216** | n/a (C4 residual verify field ~4 ms; wall = **total 66.5 ms**) | 66.5 | = C7 plateau |
| **Batch n_draft=2** `MLX_MTP_BATCH_VERIFY=1` | **20.890** | **77.1 / 71.2 ms** | 84.0 | **KILL (>67.7)** |
| Seq n_draft=3 post-P0-B | 18.290 | — | 121.9 | deep draft still loses |
| Batch n_draft=3 | 10.152 | 175.7 / 101.4 ms | 188.5 | kill |

**Timer note:** sequential C4/C7 path attributes first trunk T=1 into the `draft=` field (joint draft‖verify); use **gen t/s** and **`total=`** for sequential wall. Batch path attributes multi-token forward to **`verify=`** — that is the correct §4 comparator to 67.7 ms.

**Product:** do **not** open batch-verify rewrite WS on gfx1150 35B. Keep sequential default; leave `MLX_MTP_BATCH_VERIFY=1` opt-in only.

---

### 4.1 Pre-committed kill table (unchanged; kill line = **67.7 ms**, not 55)

```bash
MLX_MTP_BATCH_VERIFY=1 MLX_ENABLE_QUANT_FUSE=1 MLX_ENABLE_QUANT_FUSE_GDN=1 \
  MTP_TIMING=1 MTP_DEBUG=1 MLX_LOAD_MTP_HEAD=1 \
  ./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --use-mtp --n-draft 2 --temperature 0 --top-p 1 --max-tokens 256 --no-think --ignore-eos
```

**Break-even algebra** (E[tokens]=1.85; current wall 67.7 ms/step = 27.34 t/s):

| Batch T=2 verify | Resulting t/s | Δ vs C7 | Decision | S4 outcome |
|---|---|---|---|---|
| ≤ 50 ms | ≥ 37.0 (up to ~41 at 45ms) | **+35–50%** | **Reopen: build batch verify for real** | — |
| 50–60 ms | 30.8–37.0 | +13–35% | Worth building | — |
| 60–67 ms | 27.6–30.8 | +1–13% | Marginal | — |
| **> 67.7 ms** | < 27.34 | **≤ 0** | **KILL — declare plateau** | **HIT (77.1 mean / 71.2 med; gen 20.89)** |

**n_draft=3 under P0-B:** old 22.71 was invalid (final-draft KV starve). **New valid seq row = 18.29 t/s** — still regresses vs n_draft=2. Deep draft retired on this machine.

## 5. What changes the verdict if §4 kills

| Path | What it buys | Cost |
|---|---|---|
| **H1 dGPU measurement day** | MTP payoff scales with launch-overhead share; 890M is the worst case (8 CUs). On a launch-bound dGPU, batch verify plausibly goes sublinear ⇒ +25–50% regime. This is the real decision for MTP's product worth. | 1 d + hardware |
| **Attack T₁ directly** (dense_kept=7 audit W3.3, KV quant, GDN fuse) | +10–15% — but lifts **eager equally**; MTP's relative point unchanged. Do it anyway; don't credit MTP. | 2–3 d |
| **Reposition MTP** | H2 small-model MTP already touched ~100 t/s (0.8B, n≥5 runs); H3 batching is throughput not single-stream. If §4 and H1 both fail on 35B, MTP's home is the small-model/dGPU product surface, not 890M-35B. | decision, not work |
| **Declare plateau (HARD-BAN-compliant)** | Stamp ~27–31 t/s software ceiling in MTP_OPTIMALITY_PLAN §0.6 with §2 algebra + §4 probe row; keep `--use-mtp` opt-in; no seatbelts, no auto-disable. | 0.5 d |

## 6. Residual code issues (not TPS — pointer)

None of the open items from `05-p0-review.md` (R-1 residual temp-scaling, R-2/R-3 registry, R-4…R-10) move throughput. The RS sampled path's −7% (25.4 vs 27.34) is serial-T=1-verify + per-position eval inherent; batch verify (§4) would also fix its structure later (Leviathan processes positions in order from one batched forward). No additional TPS defects found in this pass.

## 7. Recommendation & decision log

1. ~~**Run the §4 probe.**~~ **DONE** (`exp/mtp-tps-ceiling`, RESULTS.md). T₂ mean **77.1 ms > 67.7** → **KILL**.
2. ~~If T₂ ≤ 60 ms → open WS5.~~ **Not opened.** Plateau memo = this section + RESULTS.
3. Escalate **H1 dGPU day** as the stakeholder decision on MTP's product scope (only plausible reopen for batch amortization).
4. Do not fund further draft-side micro-optimization (C11–C15 class) — §2 identity confirmed by S4 (accept rate healthy, wall still sequential).

**Decision record (`mtp-tps-next-action`, post-S4):** probe closed; **declare plateau ~27 t/s** single-stream 35B @ 890M; sequential default; batch opt-in only; H1/H2 strategic; T₁ work orthogonal (don't credit MTP).

## 8. Confidence & limits

Overall 0.9 on the ceiling arithmetic (identity). **Post-S4: 0.95 that batch verify does not win on this stack** (measured, not prior). Prior 0.85 on fail was directionally correct. Residual limits: single Fourier-style prompt family; one thermal order (seq before batch); batch timer includes mamba capture/rollback — a rewrite might shave some ms but **gen wall already −23% vs sequential**, so product rewrite is not justified. No invented numbers: every S4 value traces to named logs under `docs/experiments/mtp-tps-ceiling/`.
