# Lever 3 — Design C: residual lm_head cut (post 4-bit)

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  
**Status:** **DESIGN ONLY** — no implementation this fire; **no measured win %**.  
**Prerequisite measures (A+B):** already 4-bit head; isolated qmm mean **3.86958 ms** ≈ **11.48%** of same-fire T₁ @ **29.68 t/s**  
Logs: `B_lm_head_qmm.txt`, `B_t1_eager_ref.txt`.

---

## 0. Hard bounds from B (do not exceed in claims)

| Quantity | Value | Source |
|----------|-------|--------|
| Full vocab qmm mean | **3.86958 ms** | B log |
| T₁ (eager SAFE fuse) | **1000/29.68 ≈ 33.69 ms** | B log |
| Head share of T₁ | **~11.5%** | arithmetic |
| **Free-head ceiling** (if head cost → 0) | T₁′=29.82 ms → **~33.5 t/s** ≈ **+13%** vs 29.68 | arithmetic sketch |
| Program “+15–25% from 4-bit head” | **VOID** | already 4-bit; exceeds free-head ceiling |

Any two-stage or kernel path **cannot honestly advertise >~13% gen t/s** on this stack without new evidence that trunk also shrinks. Realistic partial recovery of the 3.87 ms is **smaller**.

---

## 1. Current code path (where head cost lives)

### Logits production (every decode token)

```1071:1085:src/llm/models/qwen35_moe.cpp
// post_norm → linear_fwd(post_norm, lm_head_weight_)
// linear_fwd → linear_forward → quantized_matmul when registered
```

```127:141:include/mlx-lm/common/quantized_linear.h
// registry hit → mx::quantized_matmul(x, w, scales, biases, transpose=true, gs, bits)
```

Field package: untied `lm_head.{weight,scales,biases}` 4-bit, vocab **248320**, hidden **2048**.

### Sampling (consumes full logits today)

```191:198:src/common/generate.cpp
// temp==0 → ArgMaxSampler
// else top_p∈(0,1) → TopPSampler (ROCm: currently categorical; nucleus filter disabled)
// else CategoricalSampler
```

```40:44:include/mlx-lm/common/generate.h
// ArgMaxSampler: argmax(logits, -1)
```

There is **no** vocabulary top-k shortlist before the head matmul. Samplers only run **after** full `[…, vocab]` logits exist.

### MTP interaction (critical quality/scope constraint)

| Mode | Needs dense vocab logits? | Implication for two-stage |
|------|---------------------------|---------------------------|
| Eager greedy temp=0 | Only argmax identity | Approximate head OK **iff** argmax matches full head |
| Eager temp>0 / top-p | Softmax mass over vocab | Partial logits **change distribution** unless residual mass handled |
| MTP greedy verify | Compare draft vs target tokens | Target still needs correct argmax (or full logits if multi-token verify) |
| MTP rejection sampling | Full `q` and `p` over vocab (`mtp_residual_logits`) | **Two-stage that drops rows breaks RS** without redesign |

**Product implication:** first ship gate = **temp=0 / greedy only** opt-in. MTP RS and sampled decode stay on full head until proven.

---

## 2. Design options (ranked for this stack)

### C1 — Two-stage logits (PRIMARY algorithmic design)

**Idea:**

1. **Stage 1 (cheap shortlist):** produce scores for all vocab *or* a coarse covering set with traffic ≪ full 4-bit head; take top-K indices.  
2. **Stage 2 (exact rows):** `take` packed lm_head rows (weight/scales/biases) for those K ids; run small `quantized_matmul` / dense matmul on `[1,H]×[K,H]` → K logits.  
3. **Sample** only on the K logits (greedy: argmax among K).

**Stage-1 candidates (engineering order):**

| ID | Stage-1 method | Pros | Cons / risk |
|----|----------------|------|-------------|
| C1a | **Low-rank** `h @ A` then `(hA) @ B` with `A:[H,r]`, `B:[r,V]`, r≪H | Classic; exact trainable optional | Needs extra weights or SVD of head; quality drift |
| C1b | **Prototype / hierarchical** (cluster vocab, score C clusters then rows in top clusters) | Traffic scales with C + K | Cluster build offline; mismatch risk |
| C1c | **Reuse embed as proxy** | Weights already loaded | Embed is **dequantized BF16 at load** (`quantize_utils.cpp` embed path) → **~1 GB** dense matmul — **worse** than 4-bit head; **REJECT** on this package |
| C1d | **Frequency / static allowed-ids** | Trivial | Breaks open-ended gen; not general |

**Recommended first implement (future fire, not this one):** **C1a or C1b** behind env e.g. `MLX_LM_HEAD_TWOSTAGE=1` + `MLX_LM_HEAD_STAGE1_K=…`, **temp=0 only**.

**Stage-2 primitive:** row-gather of packed U32 + scales/biases then qmm. Engine already uses `mx::gather_qmm` for MoE experts (`examples/test_qbits.cpp`); lm_head needs **2D row subset** (may be `take` on axis 0 of packed layout — **verify pack contiguity** before implement).

### C2 — Kernel-only faster full qmm (SECONDARY)

No algorithm change: tune ROCm QMV for **M=1, N=248320, K=2048, 4-bit**. Quality risk **zero**. Upside unknown; microbench is already the workload (`bench_lm_head`). Product: mlx-rocm kernel work, not lemon-mlx-engine feature flag.

### C3 — “Quantize head more” (2-bit / lower)

| Note | |
|------|--|
| Head already 4-bit | Further bits trade quality for **at most** fraction of 3.87 ms |
| `convert.cpp` default `--lmhead-bits 8` | Field package is **4**; do not re-convert as primary win |
| **Not recommended** as first C path | Quality risk high; absolute ms room small |

### C4 — Park implement

Valid after design: free-head only ~13%; two-stage overhead may erase gains. Prefer **Lever 4 graph inventory** next unless a cheap stage-1 prototype is ready.

---

## 3. Proposed measurement protocol (when implementing)

**Do not claim success without logs.**

1. **Correctness gate (temp=0):**  
   - For ≥N decode steps (e.g. 128–256), compare `argmax(full_logits)` vs `argmax(two_stage_logits)`.  
   - **Kill implement** if mismatch rate **> 0%** on Fourier prompt @ temp=0 (strict) or define soft bar only with PM sign-off.  
2. **Latency gate:**  
   - Extend `bench_lm_head` (or env path): report `stage1_ms`, `stage2_ms`, `total_ms` vs full qmm **3.87 ms** baseline.  
   - **Fund product** only if warm mean `total_ms ≤ 0.5 × 3.87` (~**≤1.93 ms**) **or** end-to-end gen t/s **≥ +5%** vs same-session full head (prefer e2e).  
3. **e2e gen:** same protocol as `B_t1_eager_ref.txt` (SAFE fuse, 128–256 tok, temp=0).  
4. **MTP:** if e2e greedy OK, separately test `--use-mtp` greedy; **RS sampled remains full head** until redesigned.

---

## 4. Quality risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Argmax flip at temp=0 | **Ship blocker** | Strict match gate; abort flag |
| Temp>0 distribution skew | High | Do not enable two-stage when `temperature!=0` |
| MTP RS broken by sparse logits | High | XOR: two-stage off when RS path active (`!mtp_uses_greedy_spec`) |
| Stage-1 extra weights VRAM | Med | r small; or offline clusters in CPU pinned |
| Pack layout / take wrong → garbage logits | High | Golden vs full head on CPU slice |
| Overclaim TPS | Process | Cap narrative at free-head ceiling until e2e log |

---

## 5. Decision this fire (design-only)

| Item | Decision |
|------|----------|
| Design C complete? | **Yes** (this document) |
| Implement C at design fire? | **No** |
| Lever 3 status then | **DESIGN_C_DONE**; implement parked behind gates in §3 |

## 5b. Stage-2 gate update (2026-08-02T02:47Z) — **FUND_STAGE2**

Log: [`B_stage2_K_sweep.txt`](B_stage2_K_sweep.txt). Same-session full qmm mean **4.026 ms** (fund_half **2.013 ms**).

| K | take+qmm mean | stage1 budget to half |
|---|----------------|------------------------|
| 256 | 0.066 ms | +1.95 ms |
| 1024 | 0.079 ms | +1.93 ms |
| 4096 | 0.321 ms | +1.69 ms |
| 8192 | 0.561 ms | +1.45 ms |
| 16384 | 1.048 ms | +0.97 ms |

**Verdict:** Stage-2 **passes** the “not already dead” test. Residual risk was **stage-1**.  

### 5c. C1a e2e (2026-08-02) — **KILL random / close residual**

See [`C1_IMPLEMENT.md`](C1_IMPLEMENT.md). Random low-rank stage-1: **quality FAIL** + e2e **flat** (~29.4 t/s). Dense full-V stage-1 cancels stage-2 savings on gfx1150.

**Lever 3 status now:** **LEVER3_CLOSED / C4** — accept ~11.5% T₁ tax; flag research-only OFF.  
**Still forbidden:** claim e2e +% or free-head +13%; product temp0.7/RS stays full head.

---

## 6. Explicit non-claims

- No +15–25% gen t/s.  
- No “two-stage will hit 3–4 ms” beyond what **full 4-bit already is** (3.87 ms).  
- Two-stage goal is **sub-full-head** (target ≤~2 ms class **if** measured), not re-quantization.  
- No S4/C11/KV reopen.
