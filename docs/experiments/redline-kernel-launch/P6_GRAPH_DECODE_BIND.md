# P6 — `graph_decode_*` stable device-ptr bind (product buffer bake)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (in-process; correctness + pointer stability)  
**Depends on:** P5 in-proc micro · P3 design · P2b `gpu_new=ok`  
**Not shipped:** gen t/s · product default ON · `call_fn` replace · HIP graphs

---

## 1. Goal

Prove the E4/P3 hinge in the **engine binary**:

1. `graph_decode_pos` / `graph_decode_input` expose **stable GPU device addresses** after in-place mutation.  
2. A retained PM4 IB can **bake** `graph_decode_pos`’s device pointer as the `acc_k` accumulator.  
3. Per-token kernarg patch + replay still matches `sum(1..T)` with **no buffer realloc**.

---

## 2. API

| Symbol | File | Role |
|--------|------|------|
| `graph_decode_device_data_ptr(array&)` | `graph_decode.{h,cpp}` | VRAM ptr = `RocmBuffer::data + offset` (not host shadow) |
| `try_micro_op` (P5/P6) | `redline_decode_session.cpp` | Stability check → bake pos → patch/replay |
| `maybe_probe_redline_graph_decode_bind` | same | L=1 one-shot `gd_bind` log (no HSACO required) |

Env (unchanged master): `MLX_REDLINE_DECODE=1` + optional `MLX_REDLINE_HSACO` for full bake+correctness.

---

## 3. Smoke (gfx1150)

| Case | Result | Log |
|------|--------|-----|
| off | 0× `[redline]` | `logs/p6-off-20260808-113412.err` |
| on-skip | READY `micro=skip` | `logs/p6-on-skip-20260808-113412.err` |
| on-micro | **`gd_bind=PASS` … `gd_post=stable micro=PASS observed=2080 expected=2080`** | `logs/p6-on-micro-20260808-113412.err` |
| xor | fail-closed | `logs/p6-xor-20260808-113412.err` |

Banner (on-micro excerpt):

```text
[redline] session READY (... gd_bind=PASS pos=0x... input=0x... gd_post=stable
  micro=PASS observed=2080 expected=2080 tokens=64 host_total_us=324.82 (NOT gen t/s))
```

Expected: `sum(1..64)=2080` (single PM4 dispatch). Host µs labeled **not** gen t/s.

---

## 4. Honesty

| Claim | Status |
|-------|--------|
| Stable product buffer identity under in-place mutate | **YES** |
| Retained PM4 bake of product pos ptr + correctness | **YES** |
| Model gen t/s A/B | **NO** |
| Product default ON / forward replace | **NO** |

---

## 5. Next

- Optional L=1 sidecar replay without replacing `call_fn` (still not gen t/s).  
- Gen t/s A/B **only** after a real product-path op is owned by Redline.
