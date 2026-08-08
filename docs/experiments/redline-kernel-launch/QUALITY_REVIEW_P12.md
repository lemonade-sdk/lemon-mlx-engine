# Quality review — P12 OWN_RMSNORM (quintuple + supervisor)

**Date:** 2026-08-08  
**Verdict:** **PASS**

| # | Role | Result | Evidence |
|---|------|--------|----------|
| 1 | explore | **PASS** | P11 table: RMSNorm 37; AOT pointer-launch like qmm; OWN_GLUE pattern for product replace |
| 2 | plan | **PASS** | Own packed only; weak MLX hook; retained IB cache by (dtype,n_rows); mid-eval stream sync; default OFF |
| 3 | implement | **PASS** | `rms_norm_kernels.hip` CO; `try_arm_rmsnorm`; C ABI; MLX patch; workitem geometry fix |
| 4 | quality | **PASS** | arm `rms=PASS` smoke_f32; `rms_multi=PASS_n4`; product OWN_RMSNORM log; inv 37→6 RMSNorm; xor fail-closed; chat rebuild 0 |
| 5 | supervisor | **PASS** | No fake TPS; no microbench as gen t/s; no default ON; no force-push; slogan (replace launches) satisfied |

## Bans

| Ban | Status |
|-----|--------|
| No fake TPS | **OK** — no gen t/s claim |
| No microbench µs as gen t/s | **OK** |
| No product default ON | **OK** — env exact `1` |
| No force-push | **OK** |

## Residuals (not FAIL)

- Mid-eval `hipStreamSynchronize` tax — may slow gen; M2 will measure.  
- 6 strided RMSNorm still product HIP.  
- Gen A/B (M2) deferred to next fire after this ownership commit.

## Supervisor decision

**PASS** — ship P12 ownership docs + code; next fire **M2** B0/B1/B2 only (no further product logic until measured).
