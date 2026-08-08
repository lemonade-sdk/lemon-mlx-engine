# P4 — MoE multipath design (optional)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **DESIGN SKETCH** (optional after stop A)  
**Depends on:** P0–P3; E3/E4

---

## 1. Why multipath

MoE decode can change the **set of expert kernels** (and thus launch topology) across tokens. A **single** retained AQL IB / graph that assumes a fixed expert set will:

- mis-dispatch if experts change, or  
- force re-record every token (killing launch-floor wins).

**v0 Redline path deliberately avoids expert kernels** (E4 §6). P4 is how to extend later **without** product default ON.

---

## 2. Options

| Option | Idea | Pros | Cons |
|--------|------|------|------|
| **A. Re-record on topology change** | Detect expert mask change → rebuild batch | Correct | Host cost may erase win |
| **B. Multipath catalog** | Pre-record K common expert sets (top-k patterns) | Fast replay when hit | Memory; cold misses |
| **C. Product HIP for MoE only** | Redline only non-MoE small ops | Simple; matches E3 | Limited surface |
| **D. Hybrid** | B for hot sets + A fallback | Practical | Complex code |

**Recommendation:** **C for product-adjacent v0**; research **D** only after P3 has a real non-MoE micro-op measured win.

---

## 3. Detection hooks (future)

- Expert routing outputs (ids tensor) each layer — compare to previous token's set (sorted).  
- Hash of active expert ids → path key.  
- Cap catalog size (`MLX_REDLINE_MOE_PATHS`, default small).

---

## 4. Non-goals

- Claiming MoE gen t/s wins without measure  
- BoundaryIndependent on mixed expert deps  
- Shipping multipath in default binary

---

## 5. Relation to stop rules

Stop A already met without P4. P4 is optional continuous-loop work; empty fires after P4 sketch should not thrash product code.
