# P10 — Retained OWN_GLUE (set_kernargs + replay)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (retained PM4 IBs for product glue; host-wall win vs one-shot)  
**Depends on:** P9 OWN_GLUE  
**Not shipped:** product default ON · gen t/s ≥2% win claim · qmm ownership

---

## 1. What changed

| Layer | P9 (one-shot) | P10 (retained) |
|-------|---------------|----------------|
| Arm | load CO; smoke via builder→finalize→free each smoke | load CO; **finalize 3 process-lifetime IBs** |
| Product `pos_set` / `pos_inc` / `scalar_copy` | `b_new`+`dispatch`+`finalize`+`replay`+`ib_free` every call | **`set_kernargs` + `replay`** on retained IB |
| call_fn / qmm | still product | still product |

This is still **product-path ownership** of the same glue launches as P9; P10 removes per-call IB rebuild cost.

---

## 2. Env (default OFF)

Same as P9:

| Env | Meaning |
|-----|---------|
| `MLX_REDLINE_DECODE=1` | Master |
| `MLX_REDLINE_OWN_GLUE=1` | Arm retained glue + route product glue |
| `MLX_REDLINE_GLUE_HSACO` | Path to `glue_kernels-gfx1150.co` |

---

## 3. Smoke (gfx1150, Qwen3.5-0.8B-4bit)

| Case | Result |
|------|--------|
| unset | 0× `[redline]` |
| `DECODE=1 OWN_GLUE=1` | `glue=PASS glue_armed=1 retained=1 set=7 inc=10 copy=42` + host wall |
| + `SMALL_OP=1` + HSACO | micro PASS; small_op fullgen PASS; OWN_GLUE retained log |
| XOR `PURE=1` | fail-closed (no session) |

**Host wall (NOT gen t/s)** — N=64 `glue_pos_set` only:

| Mode | µs/call (from READY line) |
|------|---------------------------|
| one-shot builder/finalize | ~1370–1400 |
| retained set_k+replay | ~4.5–4.6 |
| speedup | ~**300×** |

Logs: `logs/p10-{off,on-glue,xor,own-glue-smallop}-20260808-120517.*`

---

## 4. Implementation notes

- Three retained IBs: `g_glue_ib_set` / `g_glue_ib_inc` / `g_glue_ib_copy`.
- Kernarg patches must fit the **kernel segment** (`rl_pm4_ib_set_kernargs` → `RL_ERR_RECORD` if `offset+len` exceeds segment). Use u64@0 + i32@8 (or u64@8 for copy), not full 512B.
- Product route still `graph_decode.cpp` try-own first; try_to_lock avoids init deadlock.

---

## 5. Honesty

- **Does** make owned glue path much cheaper on host wall vs P9 one-shot.  
- **Does not** claim model gen t/s ≥2% win (glue is tiny vs qmm).  
- Default remains **OFF**.  
- Next for stop rule: gen A/B OWN_GLUE retained vs baseline, or own a heavier product launch.
