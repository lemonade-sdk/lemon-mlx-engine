# P7 — L=1 retained-PM4 sidecar (product `call_fn` unchanged)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (arm + inline correctness; L=1 wire present)  
**Depends on:** P5 micro · P6 graph_decode bake  
**Not shipped:** gen t/s · product default ON · forward replace · HIP graphs

---

## 1. Goal

Arm a **process-lifetime retained PM4 IB** after P5/P6 micro PASS, and optionally tick **patch+replay on every L=1 step** as a **sidecar** next to product `context_.call_fn` — never replacing it.

---

## 2. Env (all opt-in)

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_REDLINE_DECODE` | unset | Master exact `"1"` |
| `MLX_REDLINE_HSACO` | unset | CO required for micro + sidecar |
| `MLX_REDLINE_SIDECAR` | unset | Exact `"1"` → arm retained IB after micro PASS |
| `MLX_REDLINE_SIDECAR_TOKENS` | `16` | Inline arm correctness T |
| pure-graph XOR | — | fail-closed; no arm |

---

## 3. Behavior

1. Micro (P5/P6) on product `graph_decode_pos` as acc → PASS.  
2. If `SIDECAR!=1` → free IB; `sidecar=skip`.  
3. If `SIDECAR=1`: hipMalloc side acc; rebind kernarg acc@0; prime val=0; inline T-token correctness; on PASS keep gpu/mod/ib → `sidecar_armed=1`.  
4. `maybe_redline_sidecar_l1()` on L=1 in `generate.cpp` `TokenIterator::step` — patch val=n, replay; **call_fn still product**.

---

## 4. Smoke (gfx1150)

| Case | Result | Log |
|------|--------|-----|
| off | 0× | `logs/p7-off-20260808-113934.err` |
| on-skip | READY micro=skip | `logs/p7-on-skip-20260808-113934.err` |
| on-micro | micro=PASS … sidecar=skip | `logs/p7-on-micro-20260808-113934.err` |
| on-sidecar | **sidecar=PASS side_obs=136 side_exp=136 sidecar_armed=1** | `logs/p7-on-sidecar-20260808-113934.err` |
| xor | fail-closed | `logs/p7-xor-20260808-113934.err` |

Expected arm smoke: `sum(1..16)=136`. Host µs labeled **NOT gen t/s**.

---

## 5. Honesty

| Claim | Status |
|-------|--------|
| Retained multi-token patch+replay after micro | **YES** |
| L=1 hook without replacing call_fn | **YES** |
| Model gen t/s A/B | **NO** |
| Product default ON | **NO** |

---

## 6. Next

- Optional: verify L=1 ticks against side_acc after full model run (when model load works).  
- Gen t/s A/B only after a **product-owned** op is replaced by Redline.
