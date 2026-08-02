# C1 config sweep results — Pareto + recommended defaults

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-01 (local) / sweep CSV `sweep_out/SWEEP_20260801_200647.csv`  
**Harness:** `run_c1_config_sweep.sh`  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` · gfx1150 · SAFE quant fuse  
**Protocol:** temp=0, no-think, Fourier prompt; CHECK 64 tok → PERF 128 tok; any measured +Δ is **NOTABLE**.

**Ctrl:** **29.4136 t/s** (same-session baseline)

---

## 1. Recommended defaults (research opt-in)

Product default remains **full head** (`MLX_LM_HEAD_TWOSTAGE` unset).  
If experimenting, use one of these profiles:

| Profile | Env | Δ% | Mismatch (64-step) | Use when |
|---------|-----|----|--------------------|----------|
| **Speed** | `TWOSTAGE=1 R=128 POWER=1 K=16384` | **+4.09%** | 3.12% | Maximize gen t/s; accept rare argmax flips |
| **Balanced** ⭐ | `TWOSTAGE=1 R=96 POWER=0 K=16384` | **+3.10%** | **1.56%** | Best speed/quality tradeoff in this grid |
| **Match-first** | `TWOSTAGE=1 R=64 POWER=1 K=16384` | **+1.47%** | **0%** | Prefer exact greedy match on check |
| **Avoid** | R≥160–192 as default | often **negative** Δ% | mixed | Stage-1 denser than full 4-bit head |

### Copy-paste (balanced)

```bash
export MLX_LM_HEAD_TWOSTAGE=1
export MLX_LM_HEAD_STAGE1_R=96
export MLX_LM_HEAD_STAGE1_POWER=0
export MLX_LM_HEAD_STAGE1_K=16384
# optional: MLX_LM_HEAD_TWOSTAGE_CHECK=1  # doubles cost; for QA only
```

### Copy-paste (speed)

```bash
export MLX_LM_HEAD_TWOSTAGE=1
export MLX_LM_HEAD_STAGE1_R=128
export MLX_LM_HEAD_STAGE1_POWER=1
export MLX_LM_HEAD_STAGE1_K=16384
```

**Not for product yet without:** multi-seed, multi-prompt, longer max_tokens, and no default-on under temp>0 / MTP RS.

---

## 2. Pareto-style shortlists

### A. Best speed (quality_ok=1, any mismatch)

| Rank | Cell | r | p | K | mism% | tps | Δ% |
|------|------|---|---|---|-------|-----|-----|
| 1 | `r128_p1_K16384` | 128 | 1 | 16384 | 3.12 | **30.62** | **+4.09** |
| 2 | `r64_p2_K8192` | 64 | 2 | 8192 | 3.12 | 30.40 | +3.35 |
| 3 | `r128_p1_K4096` | 128 | 1 | 4096 | 6.25 | 30.35 | +3.19 |
| 4 | `r96_p0_K16384` | 96 | 0 | 16384 | **1.56** | 30.33 | **+3.10** |
| 5 | `r64_p0_K8192` | 64 | 0 | 8192 | **29.7** | 30.31 | +3.05 ⚠️ high mism |
| 6 | `r128_p2_K8192` | 128 | 2 | 8192 | 3.12 | 30.28 | +2.94 |
| 7 | `r96_p2_K16384` | 96 | 2 | 16384 | 3.12 | 30.27 | +2.92 |
| 8 | `r96_p2_K8192` | 96 | 2 | 8192 | 3.12 | 30.21 | +2.69 |

### B. Quality bar: mism% ≤ 3 and Δ% > 0

| Cell | Δ% | mism% | Role |
|------|-----|-------|------|
| **`r96_p0_K16384`** | **+3.10** | **1.56** | **Primary recommend** |
| `r64_p1_K16384` | +1.47 | **0** | Safest match |
| `r160_p0_K16384` | +1.13 | **0** | Clean match, smaller win |
| `r192_p0_K16384` | +0.78 | 1.56 | Marginal |
| `r160_p2_K16384` | +0.48 | **0** | Marginal |
| `r128_p0_K16384` | +0.05 | 1.56 | Noise-floor |

### C. Zero mismatch on 64-step CHECK and faster than ctrl

| Cell | Δ% | tps |
|------|-----|-----|
| `r64_p1_K16384` | +1.47 | 29.85 |
| `r160_p0_K16384` | +1.13 | 29.75 |
| `r160_p2_K16384` | +0.48 | 29.55 |

---

## 3. What the grid taught us

1. **Region that works:** r ∈ **{64, 96, 128}**, K large (**8192–16384**), power **0–2**.  
2. **Region that fails speed:** r ∈ **{160, 192}** often **negative** mean Δ (stage-1 cost > full qmm). Worst cell: `r192_p2_K16384` **−14.6%**.  
3. **High TPS + high mismatch is a trap:** e.g. `r64_p0_K8192` +3% with **~30%** mismatch — do not recommend.  
4. **Large K is the reliability knob;** moderate r is the speed knob.  
5. **+2–4% is the honest win band** on this machine for this method — notable, not free-head +13%.  
6. **CHECK runs are not product TPS** (they double head work). Rank only **perf** columns for speed.

### Mean Δ% by rank (all 9 cells per r)

| r | mean Δ% |
|---|---------|
| 64 | +1.59 |
| **96** | **+2.00** |
| 128 | +0.54 |
| 160 | −0.73 |
| 192 | −3.17 |

---

## 4. Full grid (sorted by Δ% desc)

| Cell | r | p | K | mism% | tps | Δ% | q_ok |
|------|---|---|---|-------|-----|-----|------|
| `r128_p1_K16384` | 128 | 1 | 16384 | 3.125 | 30.6162 | +4.089 | 1 |
| `r64_p2_K8192` | 64 | 2 | 8192 | 3.125 | 30.3981 | +3.347 | 1 |
| `r128_p1_K4096` | 128 | 1 | 4096 | 6.25 | 30.3509 | +3.187 | 1 |
| `r96_p0_K16384` | 96 | 0 | 16384 | 1.5625 | 30.3261 | +3.102 | 1 |
| `r64_p0_K8192` | 64 | 0 | 8192 | 29.6875 | 30.3117 | +3.053 | 1 |
| `r128_p2_K8192` | 128 | 2 | 8192 | 3.125 | 30.279 | +2.942 | 1 |
| `r96_p2_K16384` | 96 | 2 | 16384 | 3.125 | 30.2715 | +2.917 | 1 |
| `r96_p2_K8192` | 96 | 2 | 8192 | 3.125 | 30.205 | +2.691 | 1 |
| `r96_p2_K4096` | 96 | 2 | 4096 | 3.125 | 30.1604 | +2.539 | 1 |
| `r96_p1_K16384` | 96 | 1 | 16384 | 3.125 | 30.0706 | +2.234 | 1 |
| `r64_p2_K4096` | 64 | 2 | 4096 | 7.8125 | 30.0655 | +2.216 | 1 |
| `r64_p0_K4096` | 64 | 0 | 4096 | 31.25 | 30.0068 | +2.017 | 1 |
| `r128_p1_K8192` | 128 | 1 | 8192 | 3.125 | 29.9753 | +1.910 | 1 |
| `r96_p1_K4096` | 96 | 1 | 4096 | 9.375 | 29.948 | +1.817 | 1 |
| `r64_p1_K8192` | 64 | 1 | 8192 | 6.25 | 29.9468 | +1.813 | 1 |
| `r64_p0_K16384` | 64 | 0 | 16384 | 14.0625 | 29.9212 | +1.726 | 1 |
| `r64_p1_K4096` | 64 | 1 | 4096 | 15.625 | 29.8782 | +1.580 | 1 |
| `r64_p1_K16384` | 64 | 1 | 16384 | 0 | 29.8469 | +1.473 | 1 |
| `r192_p1_K4096` | 192 | 1 | 4096 | 3.125 | 29.8003 | +1.315 | 1 |
| `r160_p2_K8192` | 160 | 2 | 8192 | 3.125 | 29.7591 | +1.175 | 1 |
| `r96_p0_K8192` | 96 | 0 | 8192 | 18.75 | 29.7502 | +1.144 | 1 |
| `r160_p0_K16384` | 160 | 0 | 16384 | 0 | 29.7452 | +1.127 | 1 |
| `r96_p1_K8192` | 96 | 1 | 8192 | 3.125 | 29.6812 | +0.910 | 1 |
| `r192_p0_K16384` | 192 | 0 | 16384 | 1.5625 | 29.6432 | +0.781 | 1 |
| `r96_p0_K4096` | 96 | 0 | 4096 | 17.1875 | 29.5945 | +0.615 | 1 |
| `r160_p2_K16384` | 160 | 2 | 16384 | 0 | 29.5548 | +0.480 | 1 |
| `r128_p0_K16384` | 128 | 0 | 16384 | 1.5625 | 29.427 | +0.046 | 1 |
| `r160_p0_K8192` | 160 | 0 | 8192 | 1.5625 | 29.1885 | −0.765 | 1 |
| `r128_p2_K16384` | 128 | 2 | 16384 | 3.125 | 29.0974 | −1.075 | 1 |
| `r160_p0_K4096` | 160 | 0 | 4096 | 6.25 | 29.0958 | −1.080 | 1 |
| `r160_p1_K8192` | 160 | 1 | 8192 | 0 | 29.0861 | −1.113 | 1 |
| `r160_p2_K4096` | 160 | 2 | 4096 | 3.125 | 29.0757 | −1.149 | 1 |
| `r192_p2_K8192` | 192 | 2 | 8192 | 3.125 | 29.0754 | −1.150 | 1 |
| `r192_p1_K16384` | 192 | 1 | 16384 | 0 | 28.9203 | −1.677 | 1 |
| `r192_p1_K8192` | 192 | 1 | 8192 | 3.125 | 28.9013 | −1.742 | 1 |
| `r128_p2_K4096` | 128 | 2 | 4096 | 3.125 | 28.8786 | −1.819 | 1 |
| `r128_p0_K8192` | 128 | 0 | 8192 | 9.375 | 28.7992 | −2.089 | 1 |
| `r128_p0_K4096` | 128 | 0 | 4096 | 4.6875 | 28.7226 | −2.349 | 1 |
| `r160_p1_K4096` | 160 | 1 | 4096 | 3.125 | 28.6895 | −2.462 | 1 |
| `r160_p1_K16384` | 160 | 1 | 16384 | 0 | 28.6068 | −2.743 | 1 |
| `r64_p2_K16384` | 64 | 2 | 16384 | 0 | 28.5438 | −2.957 | 1 |
| `r192_p2_K4096` | 192 | 2 | 4096 | 3.125 | 28.3624 | −3.574 | 1 |
| `r192_p0_K4096` | 192 | 0 | 4096 | 1.5625 | 28.3073 | −3.761 | 1 |
| `r192_p0_K8192` | 192 | 0 | 8192 | 6.25 | 28.1968 | −4.137 | 1 |
| `r192_p2_K16384` | 192 | 2 | 16384 | 3.125 | 25.1222 | −14.590 | 1 |

Raw logs: `sweep_out/<cell>_check.txt`, `*_perf.txt`.

---

## 5. Next validation (not more blind thrash)

1. Re-run **balanced + speed** profiles, n≥3 seeds, max_tokens≥256.  
2. Second prompt family (code / Q&A).  
3. Confirm **temp>0 / think / MTP RS stay full-head** (or fail closed).  
4. Optionally set code defaults to **balanced** when flag is on (R=96, P=0, K=16384).  
5. Do **not** re-grid r≥192 as a product path.

---

## 6. Bottom line

| Claim | Status |
|-------|--------|
| Config research found real wins | **Yes** (+2–4% band) |
| Best practical pick | **`r=96, power=0, K=16384`** |
| Best speed pick | **`r=128, power=1, K=16384`** |
| Product default-on | **Not yet** |
| Method still worth iterating | **Yes**, on winner configs only |
