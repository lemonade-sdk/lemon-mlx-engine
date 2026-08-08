# P6 — Stable `graph_decode_*` device-pointer bind probe

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (pointer identity after in-place mutation)  
**Depends on:** P2b session · P3 design · P5 in-proc micro (optional)  
**Not shipped:** product forward replacement · gen t/s · default ON · HIP-graph re-enable

---

## 1. Goal

Prove the **E4 / P3 hinge**: product fixed-address buffers

- `graph_decode_input()` — `[1,1] int32`
- `graph_decode_pos()` — `[1] int32`

keep **stable** `buffer().raw_ptr()` values across in-place updates (`set_graph_decode_pos`, `set_graph_decode_input_from`). That is the precondition for future Redline kernarg patching of a product-owned micro-sequence without reallocating.

---

## 2. Behavior

| Env | Result |
|-----|--------|
| `MLX_REDLINE_DECODE` unset | 0× `[redline]` (incl. no gd_bind) |
| `=1` | Session READY (as P2b/P5) + one-shot **`gd_bind PASS|FAIL`** |
| XOR pure | fail-closed; **no** gd_bind |

Probe (once per process):

1. Ensure resident buffers via `graph_decode_device_data_ptr` (VRAM `RocmBuffer::data+offset`, not host shadow)  
2. Snapshot input & pos pointers  
3. In-place `set_graph_decode_pos(0)` + `set_graph_decode_input_from(token=1)`  
4. Snapshot again → require non-null and equal  

**API:** `maybe_probe_redline_graph_decode_bind()` + `graph_decode_device_data_ptr()`  
**Sites:** `TokenIterator::step` L=1 / `next()`, and `chat` after model load.

---

## 3. Smoke (gfx1150)

| Case | Evidence |
|------|----------|
| off | 0× `[redline]` — `logs/p6-off-20260808-113247.err` |
| on | READY `gpu_new=ok micro=skip` + **`gd_bind PASS … stable=1`** — `logs/p6-on-20260808-113247.err` (+ vram re-smoke `p6-on-vram-*`) |
| xor | XOR banner only — `logs/p6-xor-20260808-113247.err` |

Example:

```text
[redline] gd_bind PASS input=0x… pos=0x… stable=1 (P6; not gen t/s; forward still product)
```

---

## 4. Honesty

| Claim | Status |
|-------|--------|
| Stable buffer addresses under in-place update | **YES** (this host) |
| Model gen t/s | **NO** |
| Product default ON / call_fn replace | **NO** |
| Kernargs already patched from these ptrs into product kernels | **NO** (next) |

---

## 5. Next

Wire a real engine-owned small launch that **consumes** these stable addresses (or log launch-count A/B). Gen t/s A/B only after product path actually changes.
