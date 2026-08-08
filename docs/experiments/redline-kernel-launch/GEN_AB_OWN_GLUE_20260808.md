# Gen t/s A/B — OWN_GLUE only (no small_op/sidecar)

**Date:** 2026-08-08  
**Purpose:** Isolate **product glue ownership** from additive research flags.

## Flags

| Mode | Env |
|------|-----|
| baseline | all unset |
| own_glue | `DECODE=1` `OWN_GLUE=1` `GLUE_HSACO` `LIB` — **no** HSACO/SMALL_OP/SIDECAR |

## Results (64 tokens)

| Case | gen t/s |
|------|--------:|
| 0.8B baseline | **116.06** |
| 0.8B OWN_GLUE | **116.74** |
| 0.8B baseline2 | **115.82** |
| 35B baseline | **29.05** |
| 35B OWN_GLUE | **29.21** |
| 35B baseline2 | **29.03** |

Logs: `logs/ownglue-ab-*-20260808-120958.err`

## Interpretation

- OWN_GLUE **≈ baseline** (within noise); **does not hurt** like all-flags (small_op tax).  
- Also **does not win ≥2%** — glue is a small slice of token time.  
- Confirms roadmap: keep OWN_GLUE as real ownership; next gen wins need **heavier** owned launches (P11–P12).
