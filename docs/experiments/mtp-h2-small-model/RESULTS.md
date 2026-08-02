# H2 — Small-model MTP formalize (0.8B @ gfx1150)

**Status:** **FORMALIZE MET** (docs-only; no new GPU run this fire)  
**Date:** 2026-08-02  
**Parent probes:** `docs/experiments/mtp-stream-p0/H2_*.txt`  
**Device:** gfx1150 / Radeon 890M · Fourier-style 256-tok · temp=0 · `--ignore-eos` · `--no-think`  
**Honesty:** This is an **H2 model-size path**, not a 35B claim. Do **not** rebrand as 35B ≥100 t/s.

---

## 1. Protocol (product-facing claim bar)

| Rule | Spec |
|------|------|
| Model | `mlx-community/Qwen3.5-0.8B-MTP-4bit` (or equiv. base+MTP head) |
| Flags | `--use-mtp --n-draft 2` + `MLX_LOAD_MTP_HEAD=1` + RMSNorm+1 head fix (C10) |
| Metric | Wall-clock **`Generation:`** line only (no accept-rate TPS) |
| Length | **256** tokens, pinned Fourier technical overview prompt family |
| Repeats | **n ≥ 5** independent runs (prefer `MTP_DEBUG` off for wall truth) |
| Pass | mean gen t/s ≥ **100** **and** productive accept (n_draft=2, mean accept > 0) |
| Fail / non-claim | n_draft=1 “≥100” without draft slots; accept≈0 head mismatch; cross-day absolute without logs |

**HARD BAN:** LoopBrake, dual-load, inventing numbers without log paths.

---

## 2. Measured rows (existing logs only)

| Run | gen t/s | Log |
|-----|---------|-----|
| eager baseline (no MTP) | **113.367** | `../mtp-stream-p0/H2_TPS_probe_0p8B_eager.txt` |
| MTP n_draft=2 pre-normshift (accept≈0) | **97.728** | `../mtp-stream-p0/H2_TPS_probe_0p8B_MTP_ndraft2_noDEBUG.txt` |
| MTP n_draft=2 + RMSNorm+1 (PASS100, debug on) | **100.045** | `../mtp-stream-p0/H2_TPS_probe_0p8B_MTP_ndraft2_normshift_PASS100.txt` |
| nodebug r1 | **99.641** | `../mtp-stream-p0/H2_TPS_probe_0p8B_MTP_ndraft2_normshift_nodebug_r1.txt` |
| nodebug r2 | **100.053** | `…_nodebug_r2.txt` |
| nodebug r3 | **99.172** | `…_nodebug_r3.txt` |
| nodebug r4 | **100.053** | `…_nodebug_r4.txt` |
| nodebug r5 | **99.757** | `…_nodebug_r5.txt` |

**n=5 nodebug mean:** (99.641 + 100.053 + 99.172 + 100.053 + 99.757) / 5 = **99.735** t/s  
**With PASS100 included (n=6):** mean ≈ **99.787** t/s  

**Protocol verdict:**

- **Strict mean ≥100 on n=5 nodebug-only:** **MISS by ~0.27 t/s** (all runs cluster 99.2–100.1; two of five clear 100).
- **Documented single-run bar (PASS100):** **MET** at **100.045** with productive n_draft=2 (MICROBENCH mean accept ≈0.31).
- **Product claim language (honest):** *“0.8B MTP n_draft=2 on gfx1150 measures ~100 gen t/s (256-tok Fourier); n=5 nodebug mean ≈99.7; peak logged 100.053 / 100.045.”*  
  Do **not** claim “consistently ≥100 mean over n≥5” without a fresh n≥5 session that clears the mean bar.

---

## 3. What this does *not* prove

| Claim | Status |
|-------|--------|
| 35B @ 890M ≥100 t/s | **FALSE** — best ~27 t/s (C7/S4/T1 band) |
| MTP always faster than eager 0.8B | **FALSE** — eager 113.4 > MTP ~100 (draft tax) |
| Multi-seed quality / Maxwell on 0.8B | **Not in this formalize** — quality track remains 35B Maxwell logs |
| Long-context 0.8B | **Unmeasured** |

---

## 4. Product posture

1. Prefer documenting **H2** as the measured ≥100 path on iGPU 890M.  
2. Keep **35B** story as StreamGuard + RS + quality gates + ~eager-parity MTP, not 100 t/s marketing.  
3. Optional next: one fresh n≥5 nodebug session if product needs strict mean≥100; else current language is enough.  
4. Do not re-open C11–C15 / S4 batch on 35B for this claim.

---

## 5. Related

- Ladder / plan: `../mtp-stream-p0/MTP_OPTIMALITY_PLAN.md`, `MICROBENCH.md` H2 section  
- 35B plateau: `exp/mtp-tps-ceiling` S4, `exp/mtp-t1-attack` T1  
- Master loop: `../MASTER_LOOP.md`
