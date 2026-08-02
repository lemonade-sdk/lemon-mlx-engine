# exp/mtp-t1-lmhead-graph — MASTER

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Parent tip:** `fix/mtp-stream-p0` @ `875a39d`  
**Siblings:** `exp/mtp-t1-attack`, `exp/mtp-tps-ceiling`, `exp/mtp-c11-topk-close`  
**Device:** gfx1150 / Radeon 890M  
**Model (field):** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**HARD BANS:** LoopBrake / auto-disable MTP; dual-load; fake TPS; invent ms/% without logs; re-litigate S4/C11–C15/KV@256/dense_kept without new evidence.

---

## Lever board

| # | Lever | Status | Notes |
|---|--------|--------|-------|
| **2** | Batch-verify re-probe (06 §4) | **LEVER2_CLOSED / KILL** | Confirmed from `exp/mtp-tps-ceiling` S4 — **do not re-run** |
| **3** | lm_head traffic cut (T₁) | **B DONE — OPEN for design C** | Already 4-bit; qmm **~3.87 ms** ≈ **11.5%** of same-fire T₁ — kill &lt;5% T₁ **not** met |
| **4** | Graph decode MoE+GDN 35B | **PENDING** | After #3 design C or close |

---

## LEVER2_CLOSED (2026-08-02)

Stamp after confirming S4 RESULTS on sibling `exp/mtp-tps-ceiling` (not re-executed).

| Metric | Value | Source |
|--------|-------|--------|
| seq n_draft=2 gen t/s | **27.216** | `S4_seq_ndraft2.txt` |
| batch n_draft=2 gen t/s | **20.890** (−23%) | `S4_batch_ndraft2.txt` |
| batch verify_on_accept mean / med | **77.1 / 71.2 ms** | RESULTS warm stats |
| Kill line | **> 67.7 ms** | 06 §4 |
| Verdict | **KILL** | `docs/experiments/mtp-tps-ceiling/RESULTS.md` on branch `exp/mtp-tps-ceiling` |

**Cite:** `git show exp/mtp-tps-ceiling:docs/experiments/mtp-tps-ceiling/RESULTS.md`  
**Upside condition** (T₂≤60 ms for +13–50%) **failed in field.**

---

## Lever 3 — inventory A + microbench B

See [`RESULTS.md`](RESULTS.md). Headline:

- Package **already ships 4-bit `lm_head`** (U32 pack + BF16 scales/biases).
- **vocab=248320**, **hidden=2048** (not the program sketch 151936).
- Isolated `quantized_matmul` (real weights, decode T=1): **mean 3.86958 ms** (n=10 timed, warm=3) — log `B_lm_head_qmm.txt`.
- Same-fire eager SAFE fuse gen: **29.68 t/s** → T₁ ≈ **33.69 ms** — log `B_t1_eager_ref.txt`.
- Fraction: **3.87 / 33.69 ≈ 11.5%** of T₁ (inference: isolated qmm ≈ in-path head).

### Kill / fund (program rules)

| Outcome | Action | This fire |
|---------|--------|-----------|
| Already 4-bit **and** head **&lt;5% T₁** | **CLOSE** lever 3 | **No** (11.5% ≮ 5%) |
| **≥8–10% T₁** or **≥5 ms** | Design C (two-stage / further cut); **no** unmeasured win claim | **Yes on %** (11.5% ≥ 8–10%); **no** on abs (3.87 &lt; 5 ms) |
| “Quantize to 4-bit” as the win | **VOID** | already 4-bit |

**Theoretical free-head ceiling** (if head→0 and T₁ drops by mean qmm only): 33.69−3.87=29.82 ms → **~33.5 t/s** (~**+13%** vs 29.68). **Not measured**; upper-bound sketch only.

---

## Fire log

### Fire 2026-08-02T02:34Z — PROGRESS (microbench B)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | idle ~2% at start → loaded for bench + short chat |
| **Lever work** | #3 step **B** isolated qmm + same-fire T₁ denom |
| **Code** | `examples/bench_lm_head.cpp` + CMake target |

Clear Thought: sequentialthinking + decisionframework (B over C) + scientificmethod (H-qmm-expensive testing→supported on % bar) + metacognitivemonitoring (no +15–25 claim).

**Next:** design C only (two-stage sampler / vocab slice plan + quality risk); optional stop-before-lm_head delta if design needs it. Then lever 4 inventory.

### Fire 2026-08-02T02:29Z — PROGRESS

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` (created from `fix/mtp-stream-p0` @ `875a39d`) |
| **GPU** | use **~6%** idle — inventory only (no gen load) |
| **Lever work** | #2 stamp LEVER2_CLOSED; #3 step **A inventory** |
| **Not done** | microbench B; lever 4; any TPS claim |

Clear Thought: sequentialthinking + metacognitivemonitoring + decisionframework (pick A) + scientificmethod (observation: 4-bit head).

**Next:** microbench B (≥3 warm iters) isolated lm_head / full-forward vs stop-before-lm_head on gfx1150 if GPU free.

---

## Stop criteria (program)

- STOPPED if lever 3 CLOSED **and** lever 4 KILL/impossible **and** lever 2 already KILL.
- Or three consecutive fires with no implement/measure.
