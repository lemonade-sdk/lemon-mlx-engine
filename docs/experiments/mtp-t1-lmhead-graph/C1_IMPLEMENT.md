# C1 implement spike — two-stage lm_head (temp=0)

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  
**Flag:** `MLX_LM_HEAD_TWOSTAGE=1`  
**Optional:** `MLX_LM_HEAD_STAGE1_K` (default 4096), `MLX_LM_HEAD_STAGE1_R` (default 64)

## What we built

Decode **T=1 only** path in `Qwen35MoEModel::call_impl`:

1. **Stage-1 (one-time build):** random projection `R [H,r]` + `B [r,V]` from **streamed dequant** of real 4-bit `lm_head` rows (`deq @ R`).  
2. **Per token:** `s1 = (h @ R) @ B` → top-K indices → **exact** `quantized_matmul` on K packed rows → scatter into full vocab (`-1e4` fill).

Stage-2 alone was already shown cheap (`B_stage2_K_sweep.txt`: K=4096 take+qmm ~0.3 ms vs full ~4.0 ms).

## Field e2e (SAFE fuse, temp=0, no-think, 128 max, Fourier)

| Cell | gen t/s | Quality | Log |
|------|---------|---------|-----|
| Ctrl full head | **29.378** | Coherent Fourier overview | `C1_E0_ctrl.txt` |
| TWOSTAGE K=4096 r=64 | **29.490** (+0.4%) | **FAIL** — garbled / repetitive | `C1_E0_twostage.txt` |
| TWOSTAGE K=1024 r=64 | **29.345** | **FAIL** — “The” loop, early stop 93 tok | `C1_E0_twostage_K1024.txt` |

## Verdict

| Gate | Result |
|------|--------|
| Perf notable win | **No** — flat within noise (stage-1 dense `[1,r]×[r,V]` + topk + stage-2 ≈ full 4-bit qmm wall) |
| Quality (argmax / coherent text) | **FAIL** — random / streamed low-rank stage-1 does **not** preserve greedy argmax |
| Ship default | **NO** — flag stays research opt-in only |
| Free-head +13% claim | **Still not achieved** |

### Why stage-1 failed quality

Random `R` + linear sketch of dequant rows is a weak approximation of the true head. Top-K misses the true argmax often enough to derail greedy decode into loops/garble.

### Why perf didn’t win

Even with cheap stage-2, stage-1 still touches **all V columns** of `B` (BF16 `r×V` ≈ 32 MB at r=64) every token + argsort over V. On this APU that lands ~same wall as the optimized 4-bit full qmm (~4 ms class).

## What would still resolve it (next research)

1. **Better stage-1** offline SVD / trained low-rank of lm_head (not random R).  
2. **Kernel C2** — faster full 4-bit QMV (quality-neutral).  
3. **Do not** enable current TWOSTAGE for product temp=0.7 / MTP RS.

## How to reproduce

```bash
# ctrl
MLX_ENABLE_QUANT_FUSE=1 ./build/chat MODEL --temperature 0 --no-think --max-tokens 128 ...

# experiment
MLX_LM_HEAD_TWOSTAGE=1 MLX_LM_HEAD_STAGE1_K=4096 MLX_LM_HEAD_STAGE1_R=64 \
  MLX_ENABLE_QUANT_FUSE=1 ./build/chat MODEL --temperature 0 --no-think --max-tokens 128 ...
```
