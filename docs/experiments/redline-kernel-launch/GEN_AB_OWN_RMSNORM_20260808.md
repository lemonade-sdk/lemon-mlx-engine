# Gen t/s A/B — OWN_RMSNORM only (M2 after P12)

**Date:** 2026-08-08 · **TS:** `20260808-123445`  
**Binary:** `build/chat` @ P12 (`15c98d5` / tip docs)  
**Model:** mlx-community/Qwen3.5-0.8B-4bit  
**Protocol:** 64 tokens, temp 0, fixed prompt `Write a short paragraph about mountains.`, `--raw`  
**Host note:** residual VRAM ~96% (no concurrent `chat`); early runs noisier than late pairs.

## Flag stacks

| Mode | Env |
|------|-----|
| **B0** baseline | all `MLX_REDLINE_*` unset |
| **B1** OWN_RMSNORM only | `DECODE=1` `LIB=…` `OWN_RMSNORM=1` `RMS_HSACO=rms_norm_kernels-gfx1150.co` — **no** HSACO / GLUE / SMALL_OP / SIDECAR |
| **B2** all-flags | DECODE+LIB+HSACO+GLUE_HSACO+OWN_GLUE+OWN_RMSNORM+RMS_HSACO+SMALL_OP+SIDECAR |

Forbidden: `MLX_DECODE_GRAPH_PURE` with DECODE.

## Results (0.8B)

| Case | gen t/s | Notes |
|------|--------:|-------|
| B0 | 106.36 | first |
| **B1** | **108.71** | `rms=PASS rms_armed=1` + live OWN_RMSNORM log |
| B2 | 40.67 | **outlier** (discard for speed claim) |
| B0b | 69.79 | **outlier** (discard) |
| B0r2 | **115.34** | stable wave |
| **B1r2** | **111.91** | stable wave |
| B0r3 | **115.19** | stable wave |
| **B1r3** | **109.48** | stable wave |
| B2r2 | **100.03** | stable; glue+rms armed; additive tax |

### Stable-pair Δ (primary)

| Pair | B0 | B1 | Δ B1 vs B0 |
|------|---:|---:|-----------:|
| r2 | 115.34 | 111.91 | **−3.0%** |
| r3 | 115.19 | 109.48 | **−5.0%** |

B1 mean (n=3, all B1 runs): **110.03** t/s  
B0 late mean (B0r2+B0r3): **115.26** t/s → B1 **~−4.5%** vs late baseline  
B2r2 vs late B0: **100.03** ≈ **−13%**

## Correctness / arming

- B1/B2: `rms_multi=PASS_n4 rms=PASS rms_armed=1` retained; product log  
  `OWN_RMSNORM packed launch handled by Redline retained PM4`  
- B2: also `glue=PASS retained=1`, sidecar/small_op PASS (additive; **not** gen proof)

## Logs

`docs/experiments/redline-kernel-launch/logs/m2-ownrms-{B0,B1,B2,B0b,B0r2,B1r2,B0r3,B1r3,B2r2}-20260808-123445.err`

## Interpretation

1. **B1 isolates ownership:** packed RMSNorm is owned and armed without glue/small_op.  
2. **No ≥2% gen win on B1** — stable pairs show **slight regression** (~3–5%), consistent with mid-eval stream-sync tax called out in P12.  
3. **B2 slower** as expected (additive SMALL_OP/SIDECAR + ownership stack). Never use B2 alone for product default-on.  
4. **Product default ON remains FORBIDDEN.**  
5. **Not a hard KILL yet:** path is correct ownership infrastructure; gen loss is tax-shaped. Prefer **P12b** (cut sync tax) or own heavier residual (CustomKernel/strided) before enabling by default.

## 35B

Not re-run this fire (VRAM residual + prior all-flags already showed B2 slower). Claim 35B only after dedicated B0/B1 when GPU free.
