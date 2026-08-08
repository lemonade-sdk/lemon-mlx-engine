# P11 — Product HIP launch inventory (per L=1 token)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (env-gated instrument + measured table on gfx1150 0.8B)  
**Not shipped:** gen t/s claim · product default ON · ownership of qmm/call_fn

---

## 1. Goal

Count **product path** host→device dispatch sites per L=1 decode token, by MLX primitive name and launch kind, so P12 can own a **multi-launch** chain that matters — not more research flags.

**Slogan:** *Replace product launches, don’t pile flags.*

---

## 2. Env (default OFF)

| Env | Meaning |
|-----|---------|
| `MLX_LAUNCH_INV=1` **or** `MLX_REDLINE_LAUNCH_INV=1` | Enable counters + L=1 window dump |
| `MLX_LAUNCH_INV_TOKENS=N` | How many decode tokens to tabulate (default **4**) |

Unset → zero inventory overhead (early-out before map update).

---

## 3. What is counted

| Kind | Site | Meaning |
|------|------|---------|
| `kernel` | `CommandEncoder::add_kernel_node_raw` eager `hipLaunchKernel` | Manual kernels (qmm, RMSNorm, RoPE, …) |
| `module` | `add_module_kernel_node` eager `hipModuleLaunchKernel` | JIT module (rare on this smoke) |
| `lib` | `CommandEncoder::launch_kernel` residual | CustomKernel / library-style lambdas (**one count per residual site**, may wrap &gt;1 HIP call inside) |
| `inline` | graph-split residual when HIP graphs ON | N/A on product default (graphs OFF) |

**Attributed by** `arr.primitive().name()` set in `eval.cpp` before `eval_gpu` (always, not only when graphs on).

**Not gen t/s.** `est_host_us` = total × **1.5 µs** (E1-ish floor) — **lower-bound host dispatch tax sketch only**.

### Known gaps

- Direct `hipLaunchKernelGGL` **outside** `CommandEncoder` (e.g. some glue helpers) may not appear; OWN_GLUE still owns those product glue paths when enabled.
- `lib` is dispatch-site count, not always 1:1 with HIP kernels inside the lambda.

---

## 4. Implementation

| Layer | Change |
|-------|--------|
| MLX ROCm | `set_current_prim` real; `record_hip_launch`; `decode_hip_launch_count` / `decode_launch_inv_reset` / `decode_launch_inv_dump` |
| Engine | `TokenIterator::next` L=1 window: reset → step + full `eval` → dump for first N tokens |
| Patch file | [`patches/p11-launch-inv-mlx-rocm.patch`](patches/p11-launch-inv-mlx-rocm.patch) (re-apply after FetchContent refresh) |

---

## 5. Smoke (gfx1150, Qwen3.5-0.8B-4bit)

| Case | Result |
|------|--------|
| unset | **0×** `[launch-inv]` |
| `MLX_LAUNCH_INV=1` `MLX_LAUNCH_INV_TOKENS=3` | Stable **395** dispatches / L=1 token (tokens 0..2) |

**Protocol:**

```bash
printf 'hi\nquit\n' | MLX_SKIP_WARMUP=1 MLX_LAUNCH_INV=1 MLX_LAUNCH_INV_TOKENS=3 \
  ./build/chat <0.8B-snapshot> --max-tokens 6 --temperature 0 --raw
```

Logs: `logs/p11-{off,on}-20260808-121700.*`

---

## 6. Table — L=1 product path (0.8B, token 0..2 identical)

### By kind

| Kind | Count / token |
|------|--------------:|
| `kernel` | **275** |
| `lib` | **120** |
| **Total** | **395** |
| est_host_us (floor×N) | ~**592** (NOT gen t/s) |

### By prim (top)

| Prim | Count | Kind(s) | est_us (floor) |
|------|------:|---------|---------------:|
| QuantizedMatmul | **187** | kernel | ~280 |
| CustomKernel | **90** | lib | ~135 |
| RMSNorm | **37** | kernel | ~56 |
| Add | **24** | kernel | ~36 |
| CompiledSigmoid…Multiply… (long fused name) | **24** | lib | ~36 |
| RoPE | **12** | kernel | ~18 |
| CompiledSigmoid…Multiply (shorter) | **6** | lib | ~9 |
| Reshape | **6** | kernel | ~9 |
| ScaledDotProductAttention | **6** | kernel | ~9 |
| ArgReduce | **2** | kernel | ~3 |
| Gather | **1** | kernel | ~2 |

Sum of listed prims = 395.

---

## 7. Implications for ownership (P12)

| Fact | Action |
|------|--------|
| OWN_GLUE owns **glue only** (few launches vs **395**) | Explains OWN_GLUE gen A/B ≈ baseline |
| **~47%** of counted dispatches are **QuantizedMatmul** | Compute-heavy + E3 not drop-in HSACO → P14 long gate |
| Non-qmm multi-launch residual: RMSNorm (37) + Add (24) + compiled elementwise (30) + RoPE (12) + reshape/SDPA… | **P12 candidates** — replace multi-launch product/JIT chains when `MLX_REDLINE_OWN_*=1` |
| CustomKernel (90 lib) | Likely JIT fused ops; inventory attributes site as `lib` |

**P12 preference (this revision):** own a **real multi-launch non-qmm chain** used every token (e.g. elementwise/RMSNorm/RoPE family or fused CustomKernel cluster), default OFF, correctness vs eager — **not** more additive sidecars.

---

## 8. Honesty

- Inventory is **host dispatch count**, not wall-clock gen t/s.  
- No product default ON.  
- No claim that reducing glue alone wins ≥2% gen.  
- 35B MoE will have **higher** per-token counts (more experts / gathers) — re-run inventory when targeting 35B ownership.
