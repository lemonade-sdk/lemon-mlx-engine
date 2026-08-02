# Product claim bounds (experiment → #77 hygiene)

**Audience:** reviewers of [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) `fix/mtp-product` and anyone writing release notes.  
**Rule:** every throughput or quality claim needs a **log path**. HARD BAN: LoopBrake, dual-load, fake/accept-only TPS, inventing numbers.

**Product branch tip (lean):** `777f398` · **Experiment archive:** `exp/mtp-t1-attack` / siblings + `fix/mtp-stream-p0` probe trees.

---

## Ship-safe claims (supported)

| Claim | Evidence (path) | Caveat |
|-------|-----------------|--------|
| ROCm StreamGuard / server CPU encoder TLS rebind fixes MTP server 500s | P0 gates `mtp-stream-p0/gates/RESULTS.md` | Operational correctness, not t/s |
| Sequential T=1 verify is the default product path | S4 KILL batch; `exp/mtp-tps-ceiling` RESULTS | Keep `MLX_MTP_BATCH_VERIFY` opt-in only |
| Rejection sampling + residual emit for temp>0 | Maxwell post-residual **EXIT:0** | Quality = exit/coherence, not gen t/s |
| SAFE quant fuse is a small **eager T₁** knob (~+2% class on one day) | `mtp-t1-attack/RESULTS.md` T1 matrix | Within-session; not “MTP speedup” |
| dense_kept is norms-only (no free T₁ left) | `mtp-t1-attack/RESULTS.md` §2 | Closed |
| 0.8B MTP can measure ~100 gen t/s on gfx1150 (H2 path) | `H2_TPS_probe_0p8B_MTP_ndraft2_normshift_PASS100.txt` **100.045**; formalize `mtp-h2-small-model/RESULTS.md` | **Not** 35B; n=5 nodebug mean ≈**99.7** — do not claim strict mean≥100 |
| Software plateau ~27–29 t/s single-seq 35B @ 890M | C7/S4/T1/T1L band | Do not market 100 on 35B iGPU |

## Do **not** claim (killed / void / unsupported)

| Anti-claim | Why | Evidence |
|------------|-----|----------|
| Batch verify product win on this stack | S4 **KILL** (−23% gen t/s) | `exp/mtp-tps-ceiling` S4_* |
| n_draft=3 deeper draft win | S4 seq n3 **18.29** < n2 | same |
| C11–C15 draft micro-opts | consecutive non-wins | `mtp-stream-p0` C11–C15 logs |
| KV quant decode win (short or ~2k long) | T1 flat; T1L r2 +1.0–1.4% ≪5% | `T1L_eager_*.txt` |
| 35B gen ≥100 t/s on gfx1150 | ceiling ~27–29 | C7 / T1 / S4 |
| Multi-turn long-ctx r1 numbers | **VOID** getline multi-turn | `mtp-t1-attack/void_multiturn_r1/` |
| Fuse GDN in_proj always safe @ temp0.7 | historical thrash | keep double-gate |

## Quality language

- Prefer: “Maxwell-style multi-turn temp0.7 run completed **EXIT:0** after residual fixes” with path  
  `docs/experiments/mtp-stream-p0/FIELD_MAXWELL_FULL_RS_ndraft2_temp07_think_max8k_post_residual.txt`.  
- Avoid: “quality green” without naming that log (or CI goldens on #77).

## Residual **outside** this field machine

| Item | Status |
|------|--------|
| **H1 dGPU day** | Protocol only: `mtp-h1-dgpu/PROTOCOL.md` — needs discrete GPU |
| Multi-seed Maxwell | Optional P3 |
| Prefill arena / HIP graph | Separate exp trees; not MTP decode claim |

## Recommended #77 one-liner

> Lean MTP product: StreamGuard + RS/temp>0 emit, P0 gates, registry lifecycle, selective fuse; sequential verify default. Throughput: ~eager-parity 35B on 890M (not 100 t/s); batch verify remains opt-in experiment. Evidence lives on exp archives, not this PR bulk.

---

*Written for research-loop product hygiene; not a substitute for maintainer review.*
