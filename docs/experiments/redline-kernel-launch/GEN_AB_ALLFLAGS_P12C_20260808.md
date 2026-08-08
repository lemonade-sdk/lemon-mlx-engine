# Gen t/s A/B — B0 baseline vs B2 all-flags (P12c binary)

**TS:** 20260808-130807  
**Binary:** `build/chat` @ P12c (`set_k-before-pre`, POST_SYNC default device)  
**Host:** gfx1150 · GTT 60 GiB · GPU ~3% · clear-ish mem  
**Protocol:** 64 tok · temp 0 · raw · fixed `hi` · 2s settle · interleaved

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
# PRE_SYNC / POST_SYNC unset → stream / device defaults
```

## 0.8B (×3 interleaved)

| Stack | r1 | r2 | r3 | **Mean** | vs B0 |
|-------|---:|---:|---:|---------:|------:|
| **B0** baseline | 115.2 | 116.1 | 116.7 | **116.0** | — |
| **B2** all flags | 105.6 | 104.7 | 105.0 | **105.1** | **−9.4%** |

## 35B LemonMLXE

| Stack | gen t/s | vs B0 |
|-------|--------:|------:|
| **B0** | **29.13** | — |
| **B2** all flags | **27.00** | **−7.3%** |
| B0b | 28.94 | noise OK |

## Health (B2)

All arms PASS: `micro` · `sidecar` · `small_op fullgen` · `glue retained` · `rms` · OWN_GLUE + OWN_RMSNORM (P12c) logs.  
Glue microbench retained ~**310×** oneshot (NOT gen t/s).

## Cross-check prior all-flags

| Suite | 0.8B B2 vs B0 | 35B B2 vs B0 |
|-------|--------------:|-------------:|
| CLEAN | −8.4% | −8.7% |
| RETRY2 | −9.4% | −7.5% |
| **P12c fresh** | **−9.4%** | **−7.3%** |

Same shape after P12b/P12c: **all-flags still clearly slower**. Additive SMALL_OP/SIDECAR + ownership taxes; full product `call_fn` still runs.

## Conclusion

- **No gen win** with everything on.  
- **Do not** use B2 as a product speed stack or default-ON argument.  
- B1-only (OWN_RMSNORM) remains ~−3%; B2 piles more loss on top.

Logs: `logs/allflags-p12c-*-20260808-130807.*` · meta `logs/allflags-p12c-meta-20260808-130807.txt`
