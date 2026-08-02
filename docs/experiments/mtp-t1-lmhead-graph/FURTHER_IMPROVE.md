# Further improvement review — what’s still open

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  
**Policy:** every measured +Δ is **NOTABLE** ([`../NOTABLE_WINS.md`](../NOTABLE_WINS.md))  
**Clear Thought:** decisionframework `further-improve-remaining` + sequential review of closed vs open levers.

We’re still willing to improve performance. This is the **honest map** of what can still move, what is done, and what to do next.

---

## 1. Closed (do not re-thrash)

| Area | Result | Why closed |
|------|--------|------------|
| MTP draft fuses C11–C15 | REGRESS / flat | Draft hidden under T₁; accept cliffs |
| S4 batch verify | **−23%** | Multi-token verify superlinear |
| Decode HIP graph | **−3.6%** | Not stable / not worth it |
| Pure decode graph | VOID | Garble + fake t/s |
| “Quantize lm_head to 4-bit” | VOID | **Already 4-bit** on LemonMLXE package |
| dense_kept | CLOSED | All RMSNorm |
| Short-ctx KV quant | ~flat | Low fund |
| n_draft=3 | REGRESS | Post-P0-B still loses |

---

## 2. Already notable wins (keep / ship when quality-safe)

| Win | Δ | Product |
|-----|---|---------|
| SAFE quant fuse | **~+2.1%** eager | Opt-in `MLX_ENABLE_QUANT_FUSE=1` — **keep** |
| + GDN in_proj fuse | **~+3.1%** @ temp0 | Opt-in; quality gate @ 0.7 |
| MTP C7 vs eager (historical) | **~+4.6%** | Keep sequential MTP opt-in |
| Long-ctx KV8 | **~+1.4%** | Notable micro; low fund complexity |
| Product temp/think | **flat** t/s | Notable: modes don’t hurt decode |

---

## 3. Still open for further improve (ranked)

### A. Residual lm_head cut — Design C (**best on-box multi-ms lever left**)

| Fact | Value |
|------|--------|
| Full 4-bit head qmm | **~3.87 ms** |
| Share of T₁ | **~11.5%** |
| Free-head ceiling (sketch) | **~+13%** gen t/s if head→0 |

**Today:** every token still does **full vocab** `quantized_matmul` then sampler. Even **temp=0 ArgMax** only *uses* argmax after full logits exist (`generate.cpp` + `call_impl`).

| Path | Scope | Risk | Effort |
|------|--------|------|--------|
| **C1 two-stage / shortlist @ temp=0** | Greedy + MTP greedy only | Argmax match must be 100% | 2–4 d |
| **C2 kernel faster full qmm** | All modes | Quality-neutral | mlx-rocm / 3–5 d |
| **C3 two-stage @ temp=0.7 / RS** | Product sample + think | Distribution / RS bias | 5+ d after C1 |

**Recommended next implement:** env-gated **C1 temp=0 only**, measure e2e vs `T_E0` / `B_t1_eager_ref`. Any +Δ is **notable**; multi-% is fundable.

### B. Prefill HIP only (not decode)

- Stance: decode HIP **OFF**; prefill **opt-in** ([`HIP_GRAPH_STANCE.md`](HIP_GRAPH_STANCE.md)).  
- Historical **~+2–4% pp/s** — notable micro, not decode t/s.  
- Fund only if prompt-heavy product cares.

### C. Product hygiene (not raw t/s, but ships wins)

- Document defaults: SAFE fuse, no decode HIP, claim bounds.  
- Cherry-pick notables into PR **#77** / docs.  
- Goldens stay green.

### D. Off-box (real step-changes)

| Path | Why |
|------|-----|
| **H1 dGPU** | Launch/bandwidth change; batch/graph economics may flip |
| **H2 small model** | Already ~100 t/s class on 0.8B |
| **H3 multi-stream** | Aggregate tokens/s ≠ single-stream gen line |

---

## 4. What will *not* get another 2-minute thrash

- Re-running L2 batch, L4 decode HIP, C11 top_k, pure graph “optimizations.”  
- Claiming +15–25% from BF16 head math on this package.  
- Fake TPS / LoopBrake.

---

## 5. Suggested work order (if we keep going)

1. **Now:** C1 design freeze + optional `bench` for row-gather stage2 feasibility (code spike).  
2. **Implement day:** `MLX_LM_HEAD_TWOSTAGE=1` temp=0 path → e2e A/B (notable any +Δ).  
3. **If C1 wins:** try MTP greedy; leave RS full-head.  
4. **Parallel:** product PR hygiene for fuse + HIP stance.  
5. **Strategic:** H1/H2 when hardware/product surface allows.

---

## 6. Bottom line

Yes — we can still improve further, but the **menu is narrow and honest**:

| Still worth it | Not worth it |
|----------------|--------------|
| Residual **lm_head** algorithm/kernel (C1/C2) | Decode HIP / pure / batch verify |
| Ship **SAFE fuse** + clear flags | Draft-side MoE micro-opts |
| Prefill-only HIP if pp/s matters | Short-ctx KV complexity for +1% |
| H1 / H2 for big jumps | 35B@890M “path to 100” fantasy |

**Largest remaining on-box tax:** ~**11.5% of T₁** in the already-4-bit lm_head. That’s the one place left where further engineering can still be **notable** without reopening killed levers.
