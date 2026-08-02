# MTP research master loop

**Repo:** lemonade-sdk/lemon-mlx-engine  
**Canonical map:** [`BRANCH_MAP.md`](BRANCH_MAP.md)  
**Active field branch (this loop):** `exp/mtp-t1-lmhead-graph`  
**Parent archive:** `fix/mtp-stream-p0` @ `875a39d`  
**Product PR:** [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) `fix/mtp-product`  
**HARD BANS:** LoopBrake / auto-disable MTP; dual-load; fake TPS; invent numbers without logs; re-litigate S4/C11–C15/KV@256/dense_kept without new evidence.

Program state (high level):

| Area | State |
|------|--------|
| S4 batch verify + n_draft=3 | **LEVER2_CLOSED / KILL** (`exp/mtp-tps-ceiling`) |
| C11–C15 draft fuses | **Dead** (`exp/mtp-c11-topk-close`) — do not reopen |
| T1 fuse / KV@256 / dense_kept / long-ctx KV | **Closed** (`exp/mtp-t1-attack`) — do not reopen |
| **Lever 3 lm_head traffic** | **LEVER3_CLOSED / C4** — C1a quality FAIL + flat t/s; residual ~11.5% T₁ tax accepted |
| **Lever 4 graph decode 35B** | **LEVER4_KILL** — HIP −3.6% vs eager; pure garble/fake TPS |
| Field scheduler | **STOPPED** — L2+L3+L4 settled; no open fundable on-box step |

---


## Fire 2026-08-02T02:55Z — PROGRESS → **STOPPED** (C1a_KILL + L3 C4)

| Field | Value |
|-------|--------|
| **Result** | **STOPPED** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | idle; docs + verdict (C1 e2e already logged @ `6ee1612`) |
| **Lever worked** | #3 C1 implement outcome stamp + residual **C4 CLOSE** |
| **MASTER path** | `mtp-t1-lmhead-graph/{C1_IMPLEMENT,MASTER,RESULTS}.md` |

### Clear Thought

- `sequentialthinking` — C1a already measured; this fire closes residual, no L4 thrash
- `decisionframework` — C4 close over SVD thrash (dense stage1 still full-V)
- `scientificmethod` — H-c1-random-proj **refuted** (quality + no e2e win)
- `metacognitivemonitoring` — no invent TPS; +0.4% not claimed as win (noise + garble)

### Tested (cite prior logs; no new gen invent)

| Log | gen t/s | Quality |
|-----|---------|---------|
| `C1_E0_ctrl.txt` | **29.378** | Coherent Fourier |
| `C1_E0_twostage.txt` K4096 r64 | **29.490** | **GARBLE** |
| `C1_E0_twostage_K1024.txt` | **29.345** | **“The” loop** (93 tok) |

### Decision

1. **C1a_KILL** — random low-rank stage-1 not shippable.  
2. **Dense full-V stage-1** latency **void** on gfx1150 (cancels stage-2).  
3. **LEVER3_CLOSED / C4** — accept ~11.5% head tax for product.  
4. Flag `MLX_LM_HEAD_TWOSTAGE` remains research-only default **OFF**.  
5. **STOPPED** + `scheduler_delete` — L2 KILL · L3 CLOSED · L4 KILL.  
6. Off-loop only: hierarchical shortlist or mlx-rocm QMV (not this scheduler).

### Insight

Stage-2 was never the problem; full-vocab stage-1 sketches cost ~full head wall and break greedy quality unless trained/structured better than random — out of scope for product path now.

### Confidence

**0.92** C1a quality fail (logged garble). **0.88** L3 C4 close. **0.95** stop criteria.

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| ctrl 29.378 | **OK** | `C1_E0_ctrl.txt` |
| ts 29.490 + garble | **OK** | `C1_E0_twostage.txt` |
| product two-stage win | **FORBIDDEN** | quality fail |
| free-head +13% | **NOT achieved** | — |

---

## Fire 2026-08-02T02:47Z — PROGRESS (Design C stage-2 microbench FUND)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **GPU** | ~2–3% idle → bench_lm_head only (no full chat gen) |
| **Lever worked** | #3 gated micro-opt: **stage-2** take+K-row qmm vs full head |
| **Code** | `examples/bench_lm_head.cpp` (`BENCH_STAGE2=1` K-sweep) |
| **Log** | `mtp-t1-lmhead-graph/B_stage2_K_sweep.txt` |

### Clear Thought

- `sequentialthinking` — one step = stage2 fund gate; not C1 full implement; not L4 thrash
- `decisionframework` — stage2 bench over premature C4 close
- `scientificmethod` — H-s2-cheap **supported**
- `metacognitivemonitoring` — no e2e gen t/s; stage1 unmeasured; contiguous gather best-case

### Tested (isolated, real lm_head weights)

| Cell | mean ms | vs full | stage1 budget to 0.5×full |
|------|---------|---------|---------------------------|
| Full qmm | **4.026** | 100% | fund_half **2.013** |
| stage2 K=256 | **0.066** | 1.6% | **+1.95** BUDGET_OK |
| stage2 K=1024 | **0.079** | 2.0% | **+1.93** BUDGET_OK |
| stage2 K=4096 | **0.321** | 8.0% | **+1.69** BUDGET_OK |
| stage2 K=8192 | **0.561** | 13.9% | **+1.45** BUDGET_OK |
| stage2 K=16384 | **1.048** | 26.0% | **+0.97** BUDGET_OK |

### Decision

1. **FUND_STAGE2** — row-gather stage-2 is **not** the kill; two-stage latency gate remains open.
2. Stage-1 shortlist is now the **critical path** (unmeasured; quality + ms).
3. Not STOPPED: next = dedicated **C1 temp=0** implement day **or** stage-1 microbench; **no** decode HIP.
4. Do **not** claim product gen +Δ until e2e logs.

### Insight

Exact K-row head is cheap on gfx1150 (≪1 ms for K≤8k). Residual risk is inventing a stage-1 that scores shortlist in ≲1.5 ms **and** matches argmax.

### Confidence

**0.92** on stage2 wall ms (logged). **0.0** on stage1 or e2e win.

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| full 4.026 ms | **OK** | `B_stage2_K_sweep.txt` |
| K8192 take+qmm 0.561 ms | **OK** | same |
| e2e gen +% | **NOT claimed** | — |
| stage1 free | **NOT claimed** | unmeasured |

---

## Fire 2026-08-02T02:41Z — PROGRESS (LEVER4_KILL)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` @ tip after this commit |
| **GPU** | ~2% idle → three 35B chat loads |
| **Lever worked** | #4 graph decode A/B probe |
| **MASTER path** | `mtp-t1-lmhead-graph/{LEVER4_graph_inventory,MASTER,RESULTS}.md` |

### Clear Thought

- `sequentialthinking` — one step = L4 probe (matrix/DESIGN_C already done)
- `decisionframework` — probe over re-doc; GPU free
- `scientificmethod` — H-l4-graph-gain **refuted**
- `metacognitivemonitoring` — pure 829 t/s **not claimed** (fake TPS + garble)

### Tested

| Log | Key |
|-----|-----|
| `L4_E0_eager_ctrl.txt` | gen **29.8084** t/s |
| `L4_E0_hip_graph.txt` | gen **28.733** t/s (−3.61%); T₁ 34.80 ms |
| `L4_E0_pure_graph.txt` | **INVALID** 829.673 t/s; Overview loop garble |

### Decision

1. **LEVER4_KILL** — both kill bars hit on honest HIP path; pure fail-closed.
2. Do not product-enable `MLX_DECODE_GRAPH_PURE` on this stack.
3. Not STOPPED: L3 implement still PARKED.
4. Next: C4 close residual head **or** dedicated C1 implement day; **no** L4 re-probe.

### Insight

Decode HIP graphs do not buy T₁ on 35B MoE+GDN gfx1150; pure path is quality-broken. Residual T₁ work is only parked lm_head design (~11.5% head / ~+13% free-head sketch).

### Confidence

**0.92** on LEVER4_KILL (logged). **0.95** pure TPS invalid.

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| eager 29.8084 | **OK** | `L4_E0_eager_ctrl.txt` |
| HIP 28.733 | **OK** | `L4_E0_hip_graph.txt` |
| pure 829 as product speed | **FORBIDDEN** | garble log |
| LEVER4_KILL | **OK** | kill bars |

---

## Fire 2026-08-02T02:36Z — PROGRESS (design C only)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` @ tip after this commit |
| **GPU** | ~2% idle — **GPU_IDLE docs-only** (no measure) |
| **Lever worked** | #3 step **C** design plan |
| **MASTER path** | `mtp-t1-lmhead-graph/DESIGN_C.md` + MASTER/RESULTS |

### Clear Thought

- `sequentialthinking` — C design only; no implement; no L4 expand beyond pointer  
- `decisionframework` — two-stage (C1) primary; kernel (C2) secondary; park implement  
- `metacognitivemonitoring` — +15–25% **forbidden**; free-head ~+13% is ceiling sketch from B logs only  

### Reviewed

- Sampler path: `generate.cpp` ArgMax/TopP/Categorical on **full** logits  
- Head path: `qwen35_moe.cpp` `call_impl` → `linear_forward` → `quantized_matmul`  
- MTP RS needs dense vocab (`mtp_residual_logits`) — two-stage must stay off for RS  
- Embed-as-proxy rejected (dequant BF16 embed at load)  

### Tested

- **No GPU probe** this fire.  
- No new gen t/s.  

### Decision

1. Land `DESIGN_C.md` with quality gates and fund bars.  
2. Mark lever 3 **DESIGN_C_DONE / implement parked**.  
3. **Next fire:** Lever 4 graph-decode inventory only.  
4. Not STOPPED (L4 open; L3 implement optional later).  

### Insight

Residual head room is only **~3.9 ms (~11–13% free-head ceiling)**; two-stage is the only algorithmic bet, but **must prove argmax match and beat half of 3.87 ms** before product work.

### Next step

- Lever 4: inventory `graph_decode.cpp` / `MLX_DECODE_GRAPH` / pure path; short probe only if GPU free and not mid other load.

### Confidence

**0.88** on design completeness; **0.0** on two-stage e2e win (unmeasured).

### Supervisor honesty

| Claim | Verdict |
|-------|---------|
| New ms/TPS this fire | **None** |
| +15–25% two-stage | **NOT claimed** |
| Free-head ~+13% | **Sketch from B only** |

---

## Fire 2026-08-02T02:34Z — PROGRESS (microbench B: 4-bit lm_head qmm)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` @ tip after this commit |
| **GPU** | ~2% idle → bench + one eager chat |
| **Lever worked** | #3 step **B** (isolated qmm + same-fire T₁) |
| **MASTER path** | `docs/experiments/mtp-t1-lmhead-graph/{MASTER,RESULTS}.md` |

### Clear Thought

- `sequentialthinking` — B only this fire; LEVER2 already closed  
- `decisionframework` — real-weight qmm bench (C++ `bench_lm_head`) over design C  
- `scientificmethod` — H-qmm-expensive: **supported on % fund bar**, not abs ≥5 ms  
- `metacognitivemonitoring` — no invent; free-head +13% is **sketch only**  

### Reviewed

- Prior inventory: 4-bit head, vocab 248320  
- Built `examples/bench_lm_head.cpp`  

### Tested

| Log | Key number |
|-----|------------|
| `mtp-t1-lmhead-graph/B_lm_head_qmm.txt` | qmm mean **3.86958 ms** (min 3.77, max 4.12, n=10) |
| `mtp-t1-lmhead-graph/B_t1_eager_ref.txt` | Generation **29.68** t/s (128 tok, SAFE fuse) |

Fraction: 3.86958 / (1000/29.68) = **11.48%** of T₁.

### Decision

1. **Do not CLOSE** lever 3 (kill needs &lt;5% T₁; observed ~11.5%).  
2. **Fund design C** next (two-stage / further cut) — meets ≥8–10% bar; **do not** claim win %.  
3. Absolute head is already **~3.9 ms class** (matches program’s “4-bit → 3–4 ms” sketch).  
4. Free-head ceiling sketch ~+13% gen t/s if head free — **not a claim**.  
5. Not STOPPED.  

### Insight

Residual lm_head cost after 4-bit quant is **~3.9 ms / ~11–12% of T₁** on gfx1150 — still fundable for **smarter** head reduction, **not** for “quantize the head.”

### Next step

- **Design C** only: two-stage top-k / vocab-slice sampler plan + quality risk (no implement unless design fits one fire).  
- Then lever 4 graph-decode code inventory.  

### Confidence

**0.90** on qmm wall ms (logged). **0.75** that isolated qmm ≈ in-path head. **0.95** on 29.68 gen t/s log.

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| qmm mean 3.86958 ms | **OK** | `B_lm_head_qmm.txt` |
| gen 29.68 t/s | **OK** | `B_t1_eager_ref.txt` |
| ~11.5% of T₁ | **OK** (arithmetic) | both logs |
| +15–25% product win | **NOT claimed** | — |
| free-head ~+13% | **sketch only** | arithmetic ceiling |

---

## Fire 2026-08-02T02:29Z — PROGRESS (LEVER2_CLOSED + lm_head inventory A)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` (created/checked out from `fix/mtp-stream-p0` @ `875a39d`) |
| **GPU** | use **~6%** idle — **docs + header inventory only** (no model gen) |
| **Lever worked** | #2 status confirm + stamp; #3 step **A** inventory |
| **MASTER path** | `docs/experiments/mtp-t1-lmhead-graph/MASTER.md` + `RESULTS.md` |

### Clear Thought

- `sequentialthinking` — ordered levers 2→3A; no re-run S4; no invent ms  
- `metacognitivemonitoring` — BF16 622 MB claim is **speculation** until weight map; S4 numbers are **facts** from sibling logs  
- `decisionframework` — pick **A inventory** over B/C this fire  
- `scientificmethod` — observation stage: package already 4-bit lm_head  

### Reviewed

- `git show exp/mtp-tps-ceiling:docs/experiments/mtp-tps-ceiling/RESULTS.md`  
  - seq n2 **27.216** t/s; batch n2 **20.890** t/s; verify_on_accept mean **77.1 ms** / med **71.2 ms** &gt; **67.7** kill → **KILL**  
  - Logs: `S4_seq_ndraft2.txt` Generation 27.216; `S4_batch_ndraft2.txt` Generation 20.8899  
- Safetensors header of LemonMLXE 35B MTP mlx-4bit snapshot `5f638dff…`  
- Load path: `qwen35_moe.cpp` `call_impl` / `linear_fwd`; `quantize_utils.cpp` register vs embed dequant  

### Tested

- **No GPU probe** this fire (inventory is file header + config).  
- **Did not re-run** S4 / C11 / KV / dense_kept.  
- Quality: not re-run.  

### Decision

1. **LEVER2_CLOSED** — batch-verify stay killed; no product reopen.  
2. **Lever 3A:** document that primary “BF16 lm_head ~622 MB / ~13–14 ms” sketch is **wrong for this package** (vocab 248320; head already 4-bit; store ~286 MB).  
3. **Do not close lever 3 yet** — kill needs microbench B showing head **&lt;5% T₁** or **&lt;5 ms**.  
4. **Do not** design 4-bit conversion as the win (already 4-bit).  
5. Scheduler **continues** (not STOPPED).  

### Insight

On the field 35B MTP mlx-4bit model, **lm_head is already quantized (U32 pack + scales/biases)**; funding “make lm_head 4-bit” is **void**. Remaining question is **whether the 4-bit head is still expensive enough** to justify two-stage / further-cut work — **measure next**.

### Next step (one)

- **Microbench B** on gfx1150 if GPU free: isolated quantized lm_head matmul and/or full-forward vs stop-before-lm_head; ≥3 warm iters; log wall ms + T₁ fraction hypothesis **from logs only**.  
- If GPU busy: docs-only fire, state GPU_BUSY.  
- After B: close lever 3 or design C; then lever 4 inventory.  

### Confidence

**0.90** on inventory dtypes/shapes/bytes (header parse).  
**0.95** on S4 KILL (existing logs).  
**0.0** on residual head ms/% of T₁ (unmeasured).

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| S4 batch KILL 20.89 vs 27.22; verify mean 77.1 ms | **OK** | `exp/mtp-tps-ceiling` RESULTS + S4_*.txt |
| lm_head weight U32 [248320,256] 254279680 B | **OK** | safetensors header |
| lm_head total store 286064640 B | **OK** | sum weight+scales+biases |
| vocab 248320 hidden 2048 | **OK** | config.json text_config |
| BF16 full would be ~1017 MB | **OK** (arithmetic from dims) | RESULTS §1 |
| lm_head wall ms / % of T₁ | **NOT claimed** | needs B |
| +15–25% from quantizing head | **VOID** on this package | already 4-bit |
| Any new gen t/s this fire | **NONE** | — |

---

## Stop criteria

- **STOPPED** + `scheduler_delete` when: lever 3 CLOSED **and** lever 4 KILL/impossible **and** lever 2 already KILL; **or** three consecutive fires with no implement/measure.  
- This fire is **not** that condition.

## Fire 2026-08-02T02:40Z — PROGRESS (temp×think product matrix)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **Work** | Ran temp×think×eager/MTP matrix; design C residual head; lever4 inventory |
| **Insight** | Eager t/s flat across temp/think; MTP RS+think ~25.2; Design C must cover product modes not just greedy |
| **Next** | Lever4 graph probe or C1 fund decision |
| **Logs** | `mtp-t1-lmhead-graph/T_*.txt` |


## Fire 2026-08-02 — POLICY: all measured gains are NOTABLE

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (criteria update, not a new probe) |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **Directive** | User: any/all performance improvement shall be notable |
| **Doc** | `docs/experiments/NOTABLE_WINS.md` |
| **Change** | Split NOTABLE (log any +Δ) vs FUND (implement cost) vs NOISE; re-label fuse +2–3%, long-ctx KV +1–1.4%, MTP C7 +4.6% as notable; &lt;5% is not “ignore” |
| **Unchanged** | HARD BAN: no invented TPS; no +15–25% without measure; S4 batch still a notable **regress** |


## Fire 2026-08-02 — HIP graph stance (prefill-only)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **Advice reviewed** | HIP graphs only for prefill; decode/pure not stable / not worth it |
| **Clear Thought** | sequential + meta + argument + scientific conclusion |
| **Our measure** | HIP decode 28.73 vs 29.81 (−3.6% **notable regress**); pure VOID (garble + 829 t/s) |
| **Decision** | **Agree directionally** with field data: product decode HIP **OFF**; pure **OFF**; prefill **opt-in only** |
| **Doc** | `docs/experiments/mtp-t1-lmhead-graph/HIP_GRAPH_STANCE.md` |


## Fire 2026-08-02 — further improve review (still open)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (roadmap, not re-probe killed levers) |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **User** | Still willing to review and improve further |
| **Clear Thought** | decisionframework remaining levers + sequential open/closed map |
| **Doc** | `mtp-t1-lmhead-graph/FURTHER_IMPROVE.md` |
| **Still open** | Design C1 lm_head residual (largest on-box tax ~11.5% T₁); product hygiene; H1/H2 |
| **Closed** | L2 batch, L4 decode HIP, C11–C15, pure graph, “quantize head” |
| **Next if continue** | C1 temp=0 two-stage / shortlist implement + e2e measure (any +Δ notable) |


## Fire 2026-08-02 — C1v2 continue resolve (partial win)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` |
| **Change** | Range-finder stage-1 (not random proj); QR on CPU |
| **Notable** | **+2.28%** gen t/s vs full head (temp0, 128 tok) |
| **Quality** | Coherent; ~2.3% argmax mismatch under CHECK |
| **Status** | Partial resolve; flag opt-in; not product default |

