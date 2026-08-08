# Other Redline kernels × Path B (20260808)

**Clear Thought:** sequential + decision framework.  
**TS:** 20260808-145207 · 0.8B · 64 tok · tip dual-IB + completion-signal WaitValue  
**.so:** `/tmp/redline-warpfront-target/release/libredline_dispatch.so`

## Kernel inventory

| Kernel / flag | Product path? | Launch path today | Path B phases? |
|---------------|---------------|-------------------|----------------|
| **OWN_RMSNORM** | Yes (packed RMS) | set_k + phase1 / phase2 / **async WaitValue** | **Yes** |
| **OWN_GLUE** set/inc/copy | Yes (pos glue) | set_k + plain `rl_pm4_replay` | **No** (no `hip_stream` in glue try-own) |
| **SMALL_OP** | No — additive on L=1 | plain replay | No |
| **SIDECAR** | No — additive | plain replay | No |
| **HSACO micro / acc** | No — init smoke | one-shot / retained micro | No |
| floor_kernel | floor bench only | out-of-process | No |

**Key code fact:** `MLX_REDLINE_PHASE2` / `PHASE2_ASYNC` only gate `redline_try_own_rmsnorm_packed`. Glue/sidecar/small_op never call submit_after / WaitValue.

## Composition remeasure (gen t/s)

| Arm | Flags | Mode log | gen t/s | vs B0 mean ~115.8 |
|-----|-------|----------|--------:|------------------:|
| **B0** | none | — | **114.6** | — |
| **G1glue** | OWN_GLUE | plain glue | **115.6** | ≈0% |
| **R1p1** | OWN_RMS phase1 | phase1-used | **111.4** | ~−4% |
| **R1async** | OWN_RMS Path B | phase2-async-used | **110.4** | ~−5% |
| **GRasync** | GLUE+RMS async | both OWN; RMS async | **100.5** | **~−13%** |
| **ALLasync** | all research + RMS async | all PASS + async | **92.3** | **~−20%** |
| **ALLp1** | all research + RMS phase1 | all PASS + phase1 | **104.7** | ~−10% |
| **B0b** | none | — | **117.0** | noise |

All arms **rc=0**; correctness logs PASS (glue, rms, small_op fullgen, sidecar).

## Interpretation

1. **OWN_GLUE alone** still ≈ baseline — real ownership, tiny time slice; Path B not needed for gen.  
2. **Path B on RMS alone** works but no ≥2% win (same story as before).  
3. **Stacking kernels makes gen worse**, not better: glue+rms async **−13%**; all-flags **−10–20%**. Additive SMALL_OP/SIDECAR + dual-queue interaction; all-flags is **not** a speed stack.  
4. **ALLp1 > ALLasync** here — Path B does not rescue the all-flags tax.  
5. **Do not** enable SMALL_OP/SIDECAR for product gen.  
6. **Next for glue Path B** (if pursued): plumb `hipStream_t` from MLX encoder into `redline_try_own_pos_*` like RMS hook; until then glue stays plain replay + POST device.

## Policy

| Stack | Ship? |
|-------|-------|
| OWN_GLUE only | Ownership OK; default OFF until policy says ON for correctness |
| OWN_RMS + Path B | Correct; **no gen ship** |
| All-flags / GR stack | Research only; slower |

Logs: `logs/multi-*-20260808-145207.err` · meta `logs/multi-kernel-meta-20260808-145207.txt`
