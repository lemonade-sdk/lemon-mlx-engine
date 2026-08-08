# P9 — Own product decode glue (`MLX_REDLINE_OWN_GLUE`)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (arm correctness + live product routing)  
**Depends on:** P2b session · P6 gd_bind · glue CO  
**Not shipped:** gen t/s A/B win · product default ON · `call_fn` / qmm replace · HIP graphs

---

## 1. Goal

Take **real product-path ownership** of the three decode glue launches that lemon-mlx already owns as thin HIP kernels:

| Product API | HIP (default) | Redline OWN_GLUE |
|-------------|---------------|------------------|
| `set_graph_decode_pos` | `gpu_kv_pos_set` | `glue_pos_set.kd` |
| `advance_graph_decode_pos` | `gpu_kv_pos_increment` | `glue_pos_inc.kd` |
| `set_graph_decode_input_from` | `gpu_scalar_copy_i32` | `glue_scalar_copy_i32.kd` |

These are **not** matmul/qmm; they are the stable-buffer patch ops that pure-graph / decode loops need. Owning them is the first honest product-path slice (E4 partial-forward).

---

## 2. Env (defaults OFF)

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_REDLINE_DECODE` | unset | Master exact `"1"` |
| `MLX_REDLINE_OWN_GLUE` | unset | Exact `"1"` → arm glue + route product glue APIs |
| `MLX_REDLINE_GLUE_HSACO` | auto candidates | Path to `glue_kernels-gfx1150.co` |
| pure-graph XOR | — | fail-closed; no arm |

Source: [`harness/glue_kernels.hip`](harness/glue_kernels.hip)  
CO: [`logs/glue_kernels-gfx1150.co`](logs/glue_kernels-gfx1150.co)

```bash
hipcc --genco --offload-arch=gfx1150 \
  docs/experiments/redline-kernel-launch/harness/glue_kernels.hip \
  -o docs/experiments/redline-kernel-launch/logs/glue_kernels-gfx1150.co
```

---

## 3. Behavior

1. Session init (`DECODE=1` + `OWN_GLUE=1`): load glue CO; one-shot PM4 smokes:  
   - `pos_set(7)` → D2H 7  
   - `pos_inc(+3)` → D2H 10  
   - `scalar_copy(42)` into `graph_decode_input` → D2H 42  
   - Restore product buffers via H2D; banner `glue=PASS glue_armed=1 set=7 inc=10 copy=42`  
2. Live: `graph_decode.cpp` calls `redline_try_own_*` first; on success skip mlx HIP glue.  
3. `try_to_lock(g_mu)`: if session init holds the lock (or contention), fall back to HIP (no deadlock).  
4. **`context_.call_fn` still product** (qmm/attention unchanged).

---

## 4. Smoke (gfx1150, Qwen3.5-0.8B-4bit)

| Case | Result | Log |
|------|--------|-----|
| off | **0×** `[redline]` | `logs/p9-off-20260808-115626.err` |
| on-glue | **`glue=PASS glue_armed=1 set=7 inc=10 copy=42`** + live `OWN_GLUE pos_set` | `logs/p9-on-glue-20260808-115626.err` |
| xor | fail-closed | `logs/p9-xor-20260808-115626.err` |

Generation completes under OWN_GLUE (product forward). **Not** claimed as gen t/s A/B.

---

## 5. Code

| Piece | Location |
|-------|----------|
| Glue HIP source | `harness/glue_kernels.hip` |
| Arm + correctness | `try_arm_glue` in `redline_decode_session.cpp` |
| Live try_own | `redline_try_own_pos_{set,inc}` / `scalar_copy_i32` |
| Product route | `graph_decode.cpp` |

---

## 6. Honesty

| Claim | Status |
|-------|--------|
| Product glue ops owned by Redline when OWN_GLUE=1 | **YES** |
| Arm correctness set/inc/copy | **YES** |
| call_fn / qmm still product | **YES** |
| Gen t/s ≥2% win | **NO** (not measured this fire) |
| Default ON | **NO** |

---

## 7. Next

- Optional same-build gen t/s A/B: eager vs `OWN_GLUE=1` (product path **did** change for glue only).  
- Retained-IB optimize (avoid rebuild per glue launch).  
- Still need larger product ownership (or multi-dispatch chain) for a realistic gen win.
