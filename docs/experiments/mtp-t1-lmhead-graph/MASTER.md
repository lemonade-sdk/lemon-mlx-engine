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
| **3** | lm_head traffic cut (T₁) | **LEVER3_CLOSED / C4** | Tax **~11.5% T₁** accepted; stage2 cheap; **C1a random QUALITY_FAIL + flat t/s**; dense stage1 full-V latency void ([`C1_IMPLEMENT.md`](C1_IMPLEMENT.md)) |
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

### Design C + C1 e2e — **LEVER3_CLOSED (C4)**

See [`DESIGN_C.md`](DESIGN_C.md) + [`C1_IMPLEMENT.md`](C1_IMPLEMENT.md).

- **Not** “quantize to 4-bit” (void). Stage-2 take+qmm **FUNDED** (cheap).  
- **C1a random low-rank stage-1 e2e:** quality **FAIL** (garble); gen t/s **flat** (~29.4) — stage-1 dense `[1,r]×[r,V]` cancels stage-2 savings.  
- **Product:** accept residual **~11.5% T₁** head tax; `MLX_LM_HEAD_TWOSTAGE` research-only default **OFF**.  
- **Off-loop research only:** hierarchical shortlist (no full-V stage1) or mlx-rocm faster full QMV (C2). Not fundable as this field loop’s next step.

---

## Fire log

### Fire 2026-08-02T02:55Z — PROGRESS → **LEVER3_CLOSED / STOPPED**

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** then **STOPPED** |
| **Work** | Stamp C1a_KILL + C4 close residual; docs hygiene (code already @ `6ee1612`) |
| **Key logs** | `C1_E0_ctrl.txt` 29.378; `C1_E0_twostage.txt` 29.490 + garble; `C1_E0_twostage_K1024.txt` 29.345 + “The” loop |
| **Verdict** | L2 KILL · L3 **CLOSED** · L4 KILL → program stop criteria met |
| **Scheduler** | **STOPPED** + delete |

### Fire 2026-08-02 — C1 implement spike (quality FAIL, perf flat)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (negative result is progress) |
| **Code** | `MLX_LM_HEAD_TWOSTAGE=1` in `qwen35_moe` call_impl |
| **e2e** | ctrl 29.38 / K4096 29.49 / K1024 29.35 — **no notable win** |
| **Quality** | **FAIL** garble/loops — do not ship |
| **Next** | (superseded) C4 close residual |


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
- **Now:** L2 KILL · L3 **CLOSED (C4)** · L4 KILL → **STOPPED** (2026-08-02T02:55Z).
