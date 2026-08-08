# Gen t/s A/B retry — B0 / B1 OWN_RMSNORM / B2 all-flags

**TS:** 20260808-123434 · rebuilt `chat` · 64 tokens · fixed prompt

## Stacks

| ID | Flags |
|----|--------|
| **B0** | all `MLX_REDLINE_*` unset |
| **B1** | `DECODE=1` + `OWN_RMSNORM=1` + `RMS_HSACO` + `LIB` only |
| **B2** | DECODE+LIB+HSACO+GLUE_HSACO+OWN_GLUE+OWN_RMSNORM+RMS_HSACO+SMALL_OP+SIDECAR |

## Results

| Case | gen t/s | Notes |
|------|--------:|-------|
| 0.8B **B0** | **114.2** | baseline |
| 0.8B **B1** OWN_RMSNORM | **112.4** | ≈ baseline (~−1.5%) |
| 0.8B **B2** all flags | **96.5** | ~**−15%** vs B0; small_op+tax |
| 0.8B B0b | 112.7 | noise |
| 35B **B0** | **22.6** | cold/noisy first run |
| 35B **B1** OWN_RMSNORM | **28.3** | ≈ warm baseline |
| 35B **B2** all flags | **26.7** | below B0b |
| 35B B0b | 28.9 | warm baseline |

Logs: `logs/retry-ab-*-20260808-123434.err`

## Takeaway

- **B1 (own RMSNorm only):** no clear gen win; roughly baseline on 0.8B; 35B first baseline was an outlier.  
- **B2 (everything on):** clearly **slower** — additive research work + full product forward still running.  
- Flags **arm correctly** (rms/glue/small PASS on B2).
