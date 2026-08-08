# P9 — Own product decode glue launches (real slice)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (Redline PM4 replaces mlx HIP for glue ops when opt-in)  
**Depends on:** P2b session · glue CO · graph_decode VRAM ptrs  
**Not shipped:** product default ON · qmm ownership · gen t/s claim as win

---

## 1. What “own a real slice” means here

| Layer | Owner when OFF | Owner when `OWN_GLUE=1` |
|-------|----------------|-------------------------|
| `set_graph_decode_pos` | mlx `gpu_kv_pos_set` HIP | **Redline** `glue_pos_set.kd` PM4 |
| `advance_graph_decode_pos` | mlx `gpu_kv_pos_increment` | **Redline** `glue_pos_inc.kd` |
| `set_graph_decode_input_from` | mlx `gpu_scalar_copy_i32` | **Redline** `glue_scalar_copy_i32.kd` |
| Full model `call_fn` / qmm | product HIP | **still product** |

This is **not** a sidecar counter. Product code paths call `set_graph_decode_*`; with OWN_GLUE those **skip** mlx `hipLaunchKernelGGL` and launch via Redline instead.

---

## 2. Env (default OFF)

| Env | Meaning |
|-----|---------|
| `MLX_REDLINE_DECODE=1` | Master |
| `MLX_REDLINE_OWN_GLUE=1` | Arm + route glue through Redline |
| `MLX_REDLINE_GLUE_HSACO` | Path to `glue_kernels-gfx1150.co` (optional if default path exists) |
| `MLX_REDLINE_LIB` | `libredline_dispatch.so` |

Compile CO:

```bash
hipcc --genco --offload-arch=gfx1150 \
  docs/experiments/redline-kernel-launch/harness/glue_kernels.hip \
  -o docs/experiments/redline-kernel-launch/logs/glue_kernels-gfx1150.co
```

---

## 3. Smoke (gfx1150)

| Case | Result |
|------|--------|
| `DECODE=1 OWN_GLUE=1` | READY `glue_armed=1`; log **`OWN_GLUE pos_set handled by Redline PM4`**; gd_bind PASS |
| `+ HSACO + SMALL_OP=1` | micro PASS; small_op fullgen PASS; OWN_GLUE still handles pos_set |

Logs: `logs/p9-own-glue-20260808-115532.err`, `logs/p9-own-glue-smallop-20260808-p9.err`

---

## 4. Code

| Piece | Location |
|-------|----------|
| Glue kernels | `harness/glue_kernels.hip` |
| Arm + smoke set→7 | `try_arm_glue` in `redline_decode_session.cpp` |
| Try-own API | `redline_try_own_pos_set/inc/scalar_copy_i32` |
| Product route | `graph_decode.cpp` — try Redline first, else mlx HIP |

---

## 5. Honesty

- **Does** replace product **glue** launches when opt-in.  
- **Does not** replace qmm / full forward (E3).  
- Gen t/s: may be **same or slower** if one-shot; **P10** moves product glue to retained set_k+replay (host wall win; still not a gen t/s claim).  
- Default remains **eager product** when envs unset.
