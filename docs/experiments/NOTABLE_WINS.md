# Policy: any measured performance improvement is **notable**

**Date:** 2026-08-02  
**Scope:** All MTP / T₁ / fuse / graph / lm_head field work on lemonade-sdk/lemon-mlx-engine  
**User directive:** *Any and all improvement in performance shall be notable.*

---

## Three bars (do not collapse them)

| Bar | Meaning | Action |
|-----|---------|--------|
| **NOTABLE** | Measured Δ gen t/s, T₁ ms, pp/s, or step wall **with a log** — including **&lt;5%** | **Always record** in RESULTS / MASTER / NOTABLE table. Treat as a real signal. |
| **FUND** | Worth multi-day implement / product PR | Still use cost–benefit (often ~≥5% e2e **or** strategic product path) — but **do not erase** smaller notables. |
| **NOISE** | Same-session variance, single short run, or timer semantic mismatch | Label as provisional; prefer n≥2 or same-session A/B. Still **mention**, don’t invent. |

**HARD BAN unchanged:** no invented TPS; no LoopBrake; no claiming unmeasured +15–25%.

---

## Re-read of prior results under NOTABLE policy

All numbers log-backed. Small positives are **notable**, not “round to zero.”

| Lever | Δ (measured) | Old language | **New** |
|-------|----------------|--------------|---------|
| Quant fuse SAFE vs nofuse (eager, short) | **+2.1%** (28.90→29.50) | “small” | **NOTABLE** |
| Quant fuse + GDN in_proj vs nofuse | **+3.1%** (28.90→29.79) | “small” | **NOTABLE** (quality-gated @0.7) |
| Long-ctx KV8 vs fuse (2039 prompt) | **+1.4%** (28.63→29.04) | “KILL &lt;5%” | **NOTABLE** micro-win; **low FUND** (cost/complexity) |
| Long-ctx KV4 | **+1.0%** | kill | **NOTABLE** micro; low FUND |
| MTP C7 vs eager historical | **+4.6%** (26.13→27.34) | plateau | **NOTABLE** MTP edge on that day |
| Batch verify S4 | **−23%** (20.89 vs 27.22) | KILL | **NOTABLE regress** — keep sequential |
| C11 top_k=2 | **−0.4 t/s** + accept cliff | REGRESS | **NOTABLE regress** — flag off |
| Isolated 4-bit lm_head | **~3.87 ms ≈ 11.5% T₁** | fund design | **NOTABLE tax** still on the table |
| Free-head ceiling (sketch) | **~+13%** if head→0 | not claimed | **NOTABLE upside bound** if recovered (must measure) |
| Eager temp×think matrix | flat ~29.6–29.9 | “no effect” | **NOTABLE:** product modes **don’t hurt** decode t/s |
| MTP RS vs greedy | ~26.1 vs 27.1 (**~−3.7%**) | expected tax | **NOTABLE cost** of product sampling |
| MTP RS+think | ~25.2 (**~−7%** vs greedy) | expected | **NOTABLE** product-mode cost |
| HIP graph **decode** vs ctrl | **28.73 vs 29.81 (−3.6%)** | lever4 hope | **NOTABLE REGRESS** — leave decode HIP **off** |
| Pure decode graph | 829 t/s + garble | — | **VOID** (not a win; quality fail) |
| Prefill HIP F1–F3 | ~+2–4% pp/s | miss 10% bar | **NOTABLE micro** if logged; opt-in only |

---

## How to write future RESULT rows

```markdown
| Cell | gen t/s | Δ vs baseline | Notable? | Fund? | Log |
|------|---------|---------------|----------|-------|-----|
| … | … | +1.4% | YES | optional / low | path |
```

Never: “&lt;5% so ignore.”  
Always: “&lt;5% still **notable**; fund decision separate.”

---

## Implementation implication

- **Ship cheap notables** when quality-safe (e.g. SAFE quant fuse already product-shaped).  
- **Park expensive micro-notables** (KV path complexity for +1%) unless free.  
- **Still pursue** residual lm_head / graph if they can recover multi-ms T₁ — even +2–4% e2e is **notable**.
