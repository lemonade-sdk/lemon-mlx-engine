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
| **3** | lm_head traffic cut (T₁) | **INVENTORY DONE** → next **microbench B** | Primary BF16 premise **refuted** on this package (already 4-bit) |
| **4** | Graph decode MoE+GDN 35B | **PENDING** | After #3 microbench or #3 closed |

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

## Lever 3 — inventory A (this program's first measure fire)

See [`RESULTS.md`](RESULTS.md) §1. Headline:

- Package **already ships 4-bit `lm_head`** (U32 pack + BF16 scales/biases).
- **vocab=248320**, **hidden=2048** (not the program sketch 151936).
- BF16 full matrix would be **~1017 MB**, not 622 MB; on-disk 4-bit head total **~286.06 MB**.
- Decode path uses `linear_fwd` → `quantized_matmul` when registered (`qwen35_moe.cpp`).

**Not yet measured:** wall ms of 4-bit lm_head vs full T₁ (microbench B).  
**Do not claim** +15–25% or residual ≥8–10% T₁ until B logs exist.

### Kill / fund after B

| Outcome | Action |
|---------|--------|
| 4-bit head **&lt;5% T₁** or **&lt;5 ms** | **CLOSE** lever 3 with log |
| **≥8–10% T₁** or **≥5 ms** | Design only: further head cut (two-stage top-k / deeper quant) with quality risk — still no % claim until measured |
| BF16 dense head on some other package | Revisit A on that package |

---

## Fire log

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
