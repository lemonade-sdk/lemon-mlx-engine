# Gen t/s A/B — B0 baseline vs B2 all Redline flags

**Date:** 2026-08-08 · **TS:** 20260808-123234  
**Binary:** rebuilt `chat` (P12-capable)  
**Protocol:** 64 tokens, temp 0, fixed prompt; same build.

## B2 flags (all research on)

```text
MLX_REDLINE_DECODE=1
MLX_REDLINE_LIB=.../libredline_dispatch.so
MLX_REDLINE_HSACO=acc_kernel-gfx1150.co
MLX_REDLINE_GLUE_HSACO=glue_kernels-gfx1150.co
MLX_REDLINE_OWN_GLUE=1
MLX_REDLINE_OWN_RMSNORM=1
MLX_REDLINE_RMS_HSACO=rms_norm_kernels-gfx1150.co
MLX_REDLINE_SMALL_OP=1
MLX_REDLINE_SIDECAR=1
```

(Not pure-graph — XOR with DECODE.)

## Results

| Case | gen t/s | Δ vs mean baseline |
|------|--------:|--------------------|
| **0.8B B0** | **112.4** | — |
| **0.8B B2 all-flags** | **101.2** | **~−10%** |
| 0.8B B0b | 115.6 | noise |
| **35B B0** | **26.7** | — |
| **35B B2 all-flags** | **24.8** | **~−7% vs B0 / ~−12% vs B0b** |
| 35B B0b | 28.3 | noise |

Logs: `logs/m2-allflags-*-20260808-123234.err`

## Redline health (B2)

Both models: READY with `glue=PASS retained=1 rms=PASS rms_armed=1 small_op fullgen PASS` + live `OWN_GLUE` / `OWN_RMSNORM` logs.

## Interpretation

- **All flags work** (ownership + research stacks arm correctly).  
- **Gen t/s drops** because B2 still runs full product `call_fn` **plus** SMALL_OP/SIDECAR additive work, and OWN_RMSNORM may add mid-eval sync tax.  
- This is **not** a product-speed recommendation. For isolation use **B1** (OWN_GLUE only or OWN_RMSNORM only).
