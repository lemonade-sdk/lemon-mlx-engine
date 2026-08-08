# P12 — Own packed product RMSNorm launches (multi-instance non-qmm)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (arm smoke + product route + launch-inv delta)  
**Depends on:** P2b session · P11 inventory · Redline PM4 retained IBs  
**Not shipped:** product default ON · qmm ownership · gen t/s win claim

---

## 1. What “own a multi-launch chain” means here

P11 measured **37 RMSNorm** product HIP dispatches per L=1 token on 0.8B (non-qmm residual).  
P12 **replaces the packed** subset of those launches with Redline retained PM4 when opt-in.

| Layer | Owner when OFF | Owner when `OWN_RMSNORM=1` |
|-------|----------------|---------------------------|
| Packed `RMSNorm::eval_gpu` | mlx `hipLaunchKernel` via `add_kernel_node` | **Redline** `rms_norm_{f32,f16,bf16}.kd` retained PM4 |
| Strided RMSNorm | product HIP | **still product** (fallback) |
| Full model `call_fn` / qmm | product HIP | **still product** |

This is **not** a sidecar. Product `RMSNorm::eval_gpu` calls a weak hook; when the engine provides the strong symbol and envs arm, **HIP is skipped** for packed launches.

**Multi-launch:** multi-instance product family (≤31 packed / token on 0.8B) + structural multi-dispatch arm smoke (N=4 in one IB).

---

## 2. Env (default OFF)

| Env | Meaning |
|-----|---------|
| `MLX_REDLINE_DECODE=1` | Master |
| `MLX_REDLINE_OWN_RMSNORM=1` | Arm + route packed RMSNorm through Redline |
| `MLX_REDLINE_RMS_HSACO` | Path to `rms_norm_kernels-gfx1150.co` (optional if default path exists) |
| `MLX_REDLINE_LIB` | `libredline_dispatch.so` |

Compile CO:

```bash
/opt/rocm/bin/hipcc --genco --offload-arch=gfx1150 \
  docs/experiments/redline-kernel-launch/harness/rms_norm_kernels.hip \
  -o docs/experiments/redline-kernel-launch/logs/rms_norm_kernels-gfx1150.co
```

**Forbidden:** `DECODE=1` + `MLX_DECODE_GRAPH_PURE=1` (XOR fail-closed).

---

## 3. Implementation

| Piece | Location |
|-------|----------|
| Kernels | `harness/rms_norm_kernels.hip` → `logs/rms_norm_kernels-gfx1150.co` |
| Arm + smoke f32 [1,2,3,4] | `try_arm_rmsnorm` in `redline_decode_session.cpp` |
| Product try-own | `redline_try_own_rmsnorm_packed` + C ABI `mlx_redline_try_own_rmsnorm` |
| MLX route | weak hook in `rms_norm.hip` packed path only |
| Patch | [`patches/p12-own-rmsnorm-mlx-rocm.patch`](patches/p12-own-rmsnorm-mlx-rocm.patch) |

**Geometry note:** `rl_pm4_dispatch` grid is **total workitems** (`HIP gridDim×blockDim`), not HIP gridDim. Packed IB uses `work_x = n_rows * 256`, `block_x = 256`.

**Ordering:** mid-eval ownership drains the product HIP stream (`hipStreamSynchronize`) before Redline `replay_and_wait`, then device-sync for HIP consumers. **Documented tax** until stream-integrated dispatch exists.

---

## 4. Smoke (gfx1150, Qwen3.5-0.8B-4bit)

| Case | Result |
|------|--------|
| unset | **0×** `[redline]` / OWN_RMSNORM |
| `DECODE=1 OWN_RMSNORM=1` | READY `rms=PASS rms_armed=1 rms_multi=PASS_n4`; log **`OWN_RMSNORM packed launch handled by Redline retained PM4`**; gen emits text |
| XOR pure | fail-closed banner; no OWN_RMSNORM |

Logs: `logs/p12-{off,on,xor,inv-on}-20260808-122950.*`

### Launch inventory delta (product HIP sites)

| Metric | Baseline (P11) | OWN_RMSNORM ON |
|--------|---------------:|---------------:|
| Total dispatches / L=1 | **395** | **364** |
| RMSNorm kernel count | **37** | **6** (strided residual) |
| Owned (implied) | 0 | **~31** |

Inventory only counts CommandEncoder HIP sites — Redline PM4 is not counted as product HIP (correct for ownership proof).

---

## 5. Honesty

- **Does** replace packed product RMSNorm HIP when opt-in.  
- **Does not** replace strided RMSNorm, qmm, or full forward.  
- **Does not** claim gen t/s ≥2% (stream sync may **hurt** gen until optimized).  
- Default remains eager product when envs unset.  
- M2 gen A/B (B0/B1/B2) is the next measure step.

---

## 6. Next

1. **M2** clean gen A/B: B0 baseline · B1 OWN_RMSNORM only · B2 all-flags.  
2. Optional: reduce mid-eval sync tax / stream-integrated Redline.  
3. Optional: strided path or other non-qmm clusters (CustomKernel / RoPE).
