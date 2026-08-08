# Gen t/s A/B — clean re-run (user: other processes may have contended)

**TS:** 20260808-123723  
**Conditions:** rebuild `chat`; 2s settle between runs; no concurrent chat; 64 tok; fixed prompt.

## Stacks

| ID | Meaning |
|----|---------|
| **B0** | all Redline env unset |
| **B1** | `OWN_RMSNORM` only (`DECODE+LIB+OWN_RMSNORM+RMS_HSACO`) |
| **B2** | all research flags (DECODE+LIB+HSACO+GLUE+OWN_GLUE+OWN_RMSNORM+SMALL_OP+SIDECAR) |

## 0.8B (2 runs each → mean)

| Stack | Run1 | Run2 | **Mean** | vs B0 |
|-------|-----:|-----:|---------:|------:|
| **B0 baseline** | 113.9 | 114.6 | **114.3** | — |
| **B1 OWN_RMSNORM** | 111.1 | 112.6 | **111.8** | **−2.1%** |
| **B2 all flags** | 104.0 | 105.4 | **104.7** | **−8.4%** |

## 35B LemonMLXE

| Stack | gen t/s | vs warm B0 (~28.9) |
|-------|--------:|-------------------:|
| **B0** | **28.89** | — |
| **B1 OWN_RMSNORM** | **28.32** | **−2.0%** |
| **B2 all flags** | **26.38** | **−8.7%** |
| B0b | 28.83 | noise check |

## Health
All Redline arms PASS on B1/B2 (`rms=PASS`, B2 also glue+small_op fullgen PASS).

## Conclusion
With quieter machine: results are **stable**. **B1** is slight regression (sync tax), **B2** is clearly slower. No ≥2% gen win.

Logs: `logs/clean-ab-*-20260808-123723.err`
