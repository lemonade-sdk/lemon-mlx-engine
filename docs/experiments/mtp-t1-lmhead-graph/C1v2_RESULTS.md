# C1 v2 — range-finder stage-1 (continue resolve)

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  

## Change vs v1

| | v1 (random proj) | **v2 (range-finder)** |
|--|------------------|------------------------|
| Stage-1 | `s1=(h R)(W R)ᵀ` weak | `W≈Q Bh` via randomized range finder + 1 power iter; QR on **CPU** |
| Defaults | r=64, K=4096 | **r=128, K=8192, power=1** |
| Quality | Garble / loops | **Coherent Fourier** |
| Argmax vs full | broken | **~2.3% mismatch** over 128 steps (CHECK) |

## Flags

```bash
MLX_LM_HEAD_TWOSTAGE=1
MLX_LM_HEAD_STAGE1_R=128          # rank
MLX_LM_HEAD_STAGE1_K=8192         # shortlist size
MLX_LM_HEAD_STAGE1_POWER=1        # power iterations (0–3)
MLX_LM_HEAD_TWOSTAGE_CHECK=1      # optional argmax vs full (doubles cost)
```

## e2e (SAFE fuse, temp=0, no-think, 128 tok, Fourier)

| Cell | gen t/s | Δ vs ctrl | Quality | Log |
|------|---------|-----------|---------|-----|
| Ctrl full head | **29.215** | — | Coherent | `C1v2_E0_ctrl.txt` |
| v2 + CHECK | 26.370 | −9.7% | Coherent; mismatch rate **2.34%** | `C1v2_E0_twostage.txt` |
| **v2 no CHECK** | **29.880** | **+2.28% NOTABLE** | Coherent | `C1v2_E0_twostage_nocheck.txt` |

## Verdict

| Gate | Result |
|------|--------|
| Quality usable greedy text | **PASS** (session smoke) |
| Strict 0% argmax match | **FAIL** (~2.3% with CHECK) — improve with higher r/power or accept rare flips |
| Notable perf (no CHECK) | **YES +2.28%** |
| Ship product default | **Not yet** — opt-in experiment; need multi-seed + temp0.7 off + more r if we want 0% mismatch |
| Resolve residual head fully | **Partial** — real progress; not free-head +13% |

## Next to close further

1. Raise r / power; re-CHECK until mismatch ≪1%.  
2. Multi-seed Fourier + code prompts.  
3. Avoid enabling under temp>0 / MTP RS.  
4. Optional kernel C2 for quality-neutral full-head speedup.
