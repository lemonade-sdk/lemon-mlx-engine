# Gen t/s A/B — retry #2 (user: other processing may contend)

**TS:** 20260808-124030  
**Binary:** `build/chat` (unchanged since clean rebench)  
**Protocol:** 64 tok · temp 0 · raw · fixed prompt `hi` · 2s settle · no concurrent `chat`  
**Contention note:** lemonade `llama-server` (Vulkan 35B GGUF) held ~**7.7 / 8.0 GB** VRAM entire run; GPU use ~**2%** at start → **73%** post (likely lemonade traffic). Numbers still stable vs prior clean suite.

## Stacks

| ID | Meaning |
|----|---------|
| **B0** | all Redline env unset |
| **B1** | `OWN_RMSNORM` only (`DECODE+LIB+OWN_RMSNORM+RMS_HSACO`) |
| **B2** | all research flags (DECODE+LIB+HSACO+GLUE+OWN_GLUE+OWN_RMSNORM+SMALL_OP+SIDECAR) |

## 0.8B (2 runs each → mean)

| Stack | Run1 | Run2 | **Mean** | vs B0 |
|-------|-----:|-----:|---------:|------:|
| **B0 baseline** | 114.8 | 114.3 | **114.5** | — |
| **B1 OWN_RMSNORM** | 112.1 | 108.9 | **110.5** | **−3.5%** |
| **B2 all flags** | 103.8 | 103.7 | **103.7** | **−9.4%** |

## 35B LemonMLXE

| Stack | gen t/s | vs B0 |
|-------|--------:|------:|
| **B0** | **28.68** | — |
| **B1 OWN_RMSNORM** | **28.05** | **−2.2%** |
| **B2 all flags** | **26.53** | **−7.5%** |
| B0b | 28.74 | noise check (OK) |

## Health

| Stack | Redline |
|-------|---------|
| B1 | `rms=PASS` `rms_armed=1` · OWN_RMSNORM product log |
| B2 | READY full arm · glue/small_op/sidecar/rms PASS · fullgen PASS |

## Cross-check vs clean rebench (`123723`)

| | clean B0 | retry2 B0 | clean B1 | retry2 B1 | clean B2 | retry2 B2 |
|--|--------:|----------:|--------:|----------:|--------:|----------:|
| 0.8B mean | 114.3 | 114.5 | 111.8 (−2.1%) | 110.5 (−3.5%) | 104.7 (−8.4%) | 103.7 (−9.4%) |
| 35B | 28.89 | 28.68 | 28.32 (−2.0%) | 28.05 (−2.2%) | 26.38 (−8.7%) | 26.53 (−7.5%) |

**Same shape under residual lemonade VRAM:** B1 slight regression (sync tax), B2 clearly slower. **No ≥2% B1 win** → product default stays **OFF**.

Logs: `logs/retry2-ab-*-20260808-124030.{err,out}` · meta `logs/retry2-ab-meta-20260808-124030.txt`
