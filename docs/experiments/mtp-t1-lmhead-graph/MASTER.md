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
| **3** | lm_head traffic cut (T₁) | **FUND_STAGE2 / C1 OPEN** | 4-bit tax **~4.0 ms ~11.5% T₁**; stage2 take+qmm **≪ fund bar** ([`B_stage2_K_sweep.txt`](B_stage2_K_sweep.txt)); **C1 stage-1 implement next** |
| **4** | Graph decode MoE+GDN 35B | **LEVER4_KILL** | HIP −3.6%; pure VOID; prefill-only stance [`HIP_GRAPH_STANCE.md`](HIP_GRAPH_STANCE.md) |

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

### Notable / fund (program rules) — see [`../NOTABLE_WINS.md`](../NOTABLE_WINS.md)

**Policy:** any measured performance improvement is **notable** (including &lt;5%).

| Outcome | Action | This fire |
|---------|--------|-----------|
| Head tax measured (~11.5% T₁ / ~3.87 ms) | **NOTABLE tax** — residual cut still interesting | **Yes** |
| Already 4-bit **and** head trivial (≪5% T₁) | Deprioritize residual head work | **No** (11.5% still material) |
| “Quantize to 4-bit” as the win | **VOID** | already 4-bit |
| Any future C1/C2 e2e +Δ% | **NOTABLE**; fund by size/cost | pending implement |

**Theoretical free-head ceiling** (if head→0 and T₁ drops by mean qmm only): 33.69−3.87=29.82 ms → **~33.5 t/s** (~**+13%** vs 29.68). **Not measured**; upper-bound sketch only.

### Design C (2026-08-02) — complete; stage-2 **FUNDED**

See [`DESIGN_C.md`](DESIGN_C.md). Summary:

- **Not** “quantize to 4-bit” (void). Residual is algorithmic (two-stage / hierarchical) or kernel-only qmm.
- **Ship gate:** temp=0 argmax match 100% vs full head; MTP RS stays full head.
- **Stage-2 gate (measured 2026-08-02T02:47Z):** full qmm **4.026 ms**; take+qmm K=8192 **0.561 ms** (~14% of full); stage1 budget to 0.5×full ≈ **1.45 ms** — **BUDGET_OK** all K≤16384. Log: `B_stage2_K_sweep.txt`.
- **Still unmeasured:** stage-1 shortlist cost + argmax quality.
- **Honest cap:** cannot claim &gt; free-head ~+13% from head work alone; no e2e +Δ yet.
- **Next:** dedicated C1 temp=0 implement (stage-1 + wire stage-2) **or** stage-1-only microbench.

---

## Fire log

### Fire 2026-08-02T02:47Z — PROGRESS (stage-2 K-sweep FUND)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (FUND_STAGE2) |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | ~2–3% → `bench_lm_head` only |
| **Lever work** | #3 stage-2 take+K-row qmm microbench |
| **Key** | Full **4.026** ms; K8192 stage2 **0.561** ms; all K BUDGET_OK vs 0.5×full |
| **Verdict** | Stage-2 **not kill**; implement risk = **stage-1** |
| **Next** | C1 temp=0 implement day; no L4 re-probe |

Clear Thought: sequentialthinking + decisionframework (bench over C4) + scientificmethod (H-s2-cheap supported) + metacognitivemonitoring (no e2e claim).

Log: [`B_stage2_K_sweep.txt`](B_stage2_K_sweep.txt).

### Fire 2026-08-02T02:41Z — PROGRESS (LEVER4_KILL field probe)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (L4 closed) |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | ~2% idle → three short eager/graph loads |
| **Lever work** | #4 graph decode A/B |
| **Key** | Eager **29.8084** t/s; HIP **28.733** (−3.6%, T₁ 34.8 ms); pure **INVALID** 829 t/s + Overview garble |
| **Verdict** | **LEVER4_KILL** |
| **Next** | Optional L3 C1 implement day or CLOSE residual C4 → STOPPED; do not re-probe L4 |

Clear Thought: sequentialthinking + decisionframework (probe L4) + scientificmethod (H-l4-graph-gain **refuted**) + metacognitivemonitoring (ban fake 829).

Logs: `L4_E0_eager_ctrl.txt`, `L4_E0_hip_graph.txt`, `L4_E0_pure_graph.txt`.

### Fire 2026-08-02T02:40Z — PROGRESS (temp×think matrix + design C + lever4 inventory)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **Work** | Product-mode gen matrix (temp 0/0.7 × think × eager/MTP); DESIGN_C_two_stage; LEVER4 inventory; scheduler prompt requires temp/think |
| **Key TPS** | Eager ~29.6–29.9 flat; MTP 27.1 / RS 26.1 / RS+think 25.2 |
| **Quality** | Smoke coherent E07/E07T/M07/M07T; not full Maxwell re-cert |
| **Next** | Optional lever4 graph T₁ A/B; implement C1 only if funded; do not claim +15–25% |

Clear Thought: sequentialthinking + decisionframework (matrix over implement C).


### Fire 2026-08-02T02:36Z — PROGRESS (design C)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | ~2% idle — **docs only** (no probe) |
| **Lever work** | #3 step **C design** → [`DESIGN_C.md`](DESIGN_C.md) |
| **Implement** | **None** (parked behind §3 gates) |

Clear Thought: sequentialthinking + decisionframework (two-stage primary, kernel secondary, park implement) + metacognitivemonitoring (no +15–25; free-head ceiling only).

**Next:** Lever 4 graph-decode code inventory (`src/common/graph_decode.cpp`, `MLX_DECODE_GRAPH`, `MLX_DECODE_GRAPH_PURE` in `generate.cpp`).

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
- **Now:** L2 KILL · L4 KILL · L3 **FUND_STAGE2** (stage-1 implement still open) → loop **not** STOPPED.
