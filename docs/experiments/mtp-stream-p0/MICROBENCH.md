# MTP microbench — VERIFY_COST / draft / accept

**Date:** 2026-08-01  
**Branch:** `fix/mtp-stream-p0`  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` @ gfx1150  
**Method:** Post-hoc parse of single-process field probes (`MTP_DEBUG` + `MTP_TIMING` / `[mtp-t]` lines). No full Maxwell SAR this fire.  
**Sources:**

| File | Role |
|------|------|
| [`TPS_probe_no_mtp.txt`](./TPS_probe_no_mtp.txt) | Full fuse eager baseline (MTP head still loaded/dequantized; path not used) |
| [`TPS_probe_ndraft2.txt`](./TPS_probe_ndraft2.txt) | MTP `--n-draft 2` (1 draft slot + d0) |
| [`TPS_probe_ndraft4.txt`](./TPS_probe_ndraft4.txt) | MTP `--n-draft 4` |
| [`TPS_probe_ndraft6.txt`](./TPS_probe_ndraft6.txt) | MTP `--n-draft 6` |

**Timing brackets** (`TokenIterator::mtp_speculative_step` in `src/common/generate.cpp`):

- **draft** = MTP serial draft chain through first `mx::eval` of draft tokens  
- **verify** = trunk multi-token `call_fn` + argmax + `eval(y_)` (before commit/rollback)  
- **commit** = cache trim / rollback / clear `capture_spec`  
- **total** = wall for whole speculative step  

**Warm stats** drop step=1 (cold / first-step outliers). Primary tables use warm rows.

---

## Eager baseline

| Metric | Value |
|--------|-------|
| Generation t/s (256 tok) | **26.1317** |
| **Eager T=1 wall** \(T₁ = 1000 / tps\) | **38.27 ms** |
| Prompt prefill | 36 tok, 40.97 t/s |

Note: load log still shows `[MTP] Dequantized 20 weights` even on no-MTP path when `MLX_LOAD_MTP_HEAD=1` (memory tax, not decode path).

---

## VERIFY_COST (required row)

Verify sequence length **K = n_draft** (tokens `d0..d_{K-1}` fed to trunk in one forward).  
β_K = mean_verify / (K · T₁). Per-token verify cost vs eager: verify_ms / K / T₁.

| K (n_draft) | steps (warm) | mean verify (ms) | median verify (ms) | p90 verify (ms) | K·T₁ (ms) | **β_K** | verify/token (ms) | vs T₁ |
|-------------|----------------|------------------|--------------------|-----------------|-----------|---------|-------------------|-------|
| **2** | 150 | **124.1** | 80.9 | 223.2 | 76.5 | **1.62** | 62.0 | **1.62×** |
| **4** | 112 | **259.0** | 285.9 | 301.6 | 153.1 | **1.69** | 64.7 | **1.69×** |
| **6** | 119 | **337.8** | 345.6 | 367.9 | 229.6 | **1.47** | 56.3 | **1.47×** |

### VERIFY_COST findings

1. **Multi-token verify is ~1.5–1.7× costlier per token than eager T=1**, not free reuse. Consistent with MoE expert traffic scaling with sequence length + GDN `capture_spec` intermediates.  
2. **Near-linear in K**: ~56–65 ms/token across K∈{2,4,6}; no superlinear explosion, but also **no CUDA-style free K-verify**.  
3. Verify is **bimodal at K=2** (median ~81 ms, p90 ~223 ms) — not explained by accept (accept=0 and accept=1 have similar mean verify). Suspect MoE routing / device variance / capture path.  
4. At K≥4 verify dominates step time (see draft/verify fractions below).

**Implication for speedup model** (mlx-lm PR#990 bandwidth form `speedup ≈ (1+p)/(β+δ)`):

- Even if draft were free (δ=0) and p=1, β≈1.6 ⇒ max ~1.25× — and only if every draft accepted.  
- With measured δ and p, prediction is deep regression (next section).

---

## Draft cost

| K | warm mean draft (ms) | median | p90 | δ = draft/T₁ | draft tokens (K−1) | notes |
|---|----------------------|--------|-----|--------------|--------------------|-------|
| 2 | **156.8** | 159.9 | 166.7 | **4.10** | 1 | Dominates K=2 (~56% of total) |
| 4 | 77.4 | 36.1 | 179.3 | 2.02 | 3 | Heavy bimodal; median cheap after warm |
| 6 | 60.2 | 35.7 | 187.6 | 1.57 | 5 | Median low; tail expensive |

Per **true** draft token (excluding d0), K=2 serial path: **~157 ms for one MTP MoE step + full `lm_head`** ≈ **4.1× one eager trunk token**. That alone guarantees MTP loses on this stack unless draft collapses dramatically.

K=4/6 medians ~36 ms suggest some steps amortize or hit a faster path (cached hidden / fewer host barriers) while p90 stays ~180 ms — still not competitive with free draft assumption.

---

## Accept rate

Definition: `accepted / (n_draft − 1)` per step (d0 not counted). Literature A/V style.

| K | warm steps | mean accepted | **accept rate p** | mean tokens/step (=1+accepted) | accept histogram (warm) |
|---|------------|---------------|-------------------|--------------------------------|-------------------------|
| 2 | 150 | 0.693 | **0.693** | 1.693 | 0:46, 1:104 |
| 4 | 112 | 1.277 | **0.426** | 2.277 | 0:33, 1:36, 2:22, 3:21 |
| 6 | 119 | 1.126 | **0.225** | 2.126 | 0:46, 1:32, 2:25, 3:12, 4:4 |

Accept **falls with depth** (per-slot). Extra draft slots rarely fully convert; K=6 often wastes verify on rejected suffixes.

---

## End-to-end gen t/s (256-token probe)

| Config | Gen t/s | vs eager | Timing-implied t/s* | mean total/step (ms) | draft frac | verify frac |
|--------|---------|----------|---------------------|----------------------|------------|-------------|
| eager (no MTP path) | **26.13** | 1.00× | — | ~38.3 (T₁) | — | — |
| n_draft=2 | **6.05** | 0.23× | 6.03 | 281 | 56% | 44% |
| n_draft=4 | **6.76** | 0.26× | 6.76 | 337 | 23% | **77%** |
| n_draft=6 | **5.38** | 0.21× | 5.34 | 398 | 15% | **85%** |

\*From Σ(1+accepted) / Σ(total_us) on warm steps — matches reported Generation line within noise.

### Bandwidth-model check (K=2 warm)

- p ≈ 0.69, β ≈ 1.62, δ ≈ 4.10  
- (1+p)/(β+δ) ≈ 1.69 / 5.72 ≈ **0.30×**  
- Measured ≈ **0.23×** (extra commit, host eval, non-ideal accounting) — **same order; hypothesis supported**.

---

## What must move for MTP to beat eager

Break-even rough for K=2 (1 draft slot): need  
`(1+p) · T₁  ≳  draft + verify`  
⇒ with p=0.7: left side ≈ 65 ms; right side today ≈ 157+124 = **281 ms**.  

To match eager (~1.0×):

| Lever | Required ballpark |
|-------|-------------------|
| Draft | ≤ ~20–30 ms (δ ≲ 0.7), not 157 ms |
| Verify K=2 | ≤ ~1.0–1.1× · 2 · T₁ ≈ 80 ms (today 124 ms mean; median 81 already near) |
| Accept | Keep p ≳ 0.6–0.8 |
| Policy | Until then: **auto-fallback to eager** (plan P0-1) |

Median verify at K=2 is already ~81 ms (~β≈1.06 vs 2·T₁); **mean is pulled by ~223 ms p90 tail**. Killing the verify tail + collapsing draft are both P0.

---

## Ranked next actions (from this microbench)

1. **P0-1 Auto-disable** when rolling gen t/s ≪ eager (product safety; 4× regression is unambiguous).  
2. **P0-4 Draft path** — MoE MTP + full `lm_head` at ~157 ms/token is the largest single number at K=2.  
3. **P0-3 Verify tail / capture_spec** — reduce p90 verify; median already near linear K·T₁.  
4. **Do not raise n_draft** — K=6 is slowest; accept rate collapses; verify share → 85%.  
5. Adaptive `current_draft_count` (P1) is **pointless until draft+verify each ≪ T₁**.

---

## Online context (microbench-relevant)

- mlx-lm native MTP: [PR #990](https://github.com/ml-explore/mlx-lm/pull/990) — form `speedup=(1+p)/(β+δ)`; MoE 35B-A3B often **~1.03–1.11×** on Apple when overhead is small; dense ~1.5×.  
- OptiQ: [MTP docs](https://mlx-optiq.com/docs/mtp) — γ=1 optimal when K-verify scales near-linear; skip MTP when base already fast.  
- Our gfx1150 MoE bar sits **far below** those win regimes because **δ≈4** and **β≈1.5–1.7**, not δ≪1 and β≈1.

---

## Repro snippet (parse only)

```bash
# From repo root — recompute means from existing logs
python3 - <<'PY'
# (same parser as used for this document; see git history of this fire)
PY
```

Or re-run short probe (not required for this file’s numbers):

```bash
env MLX_ENABLE_QUANT_FUSE=1 MLX_ENABLE_QUANT_FUSE_GDN=1 MLX_LOAD_MTP_HEAD=1 MTP_DEBUG=1 MTP_TIMING=1 \
  ./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --use-mtp --n-draft 2 --temperature 0 --max-tokens 256 --no-think
```
