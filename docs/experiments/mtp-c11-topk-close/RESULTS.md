# C11 close-out — `MLX_MTP_DRAFT_TOPK` is dead on this machine

**Branch:** `exp/mtp-c11-topk-close`  
**Parent (sibling of `exp/mtp-tps-ceiling`):** `fix/mtp-stream-p0` @ `875a39d`  
**Scope:** Hygiene + documentation for a **clean negative** field result — **not** a re-measure or product enable.  
**Primary evidence (already on parent tip):** `docs/experiments/mtp-stream-p0/C11_TPS_probe_ndraft2_topk2.txt`, CRITICAL_ANALYSIS C11 row.

---

## Record (do not re-litigate without new evidence)

| Config | gen t/s | accept | joint draft= ms | log |
|--------|---------|--------|-----------------|-----|
| C7 trained k=8 | **27.34** | **0.85** | **~38** | `C7_TPS_probe_ndraft2.txt` |
| C11 `MLX_MTP_DRAFT_TOPK=2` | **26.94** (−0.4) | **0.72** | **~60** | `C11_TPS_probe_ndraft2_topk2.txt` |

Opt-in flag stays **default off**.

---

## Dead three independent ways (any one kills)

### 1. Premise falsified — for this head

Comment formerly claimed speculative draft can “often keep high accept with fewer experts (routing shortcut).”

**Measure:** k=2 → **−13 pp accept** (0.85→0.72).  
**C14 corroboration:** skip shared expert → accept 0.85→0.71, −1.7 t/s. Same cliff from the other side.

This Qwen3.6 MTP head leans on full routed+shared structure. Routing-shortcut theory does **not** hold here.

### 2. Savings do not exist on gfx1150

Joint step inflated **38→60 ms** despite **strictly less** expert FLOP.

In the **launch-bound 8-CU** regime, currency is launches / routing dispatch / full-vocab `lm_head`, not expert matmul FLOPs. Shrinking k changes GEMM shapes into worse kernels without cutting fixed costs.

Same mechanism as **C13** (draft QKV fuse REGRESS) and **C14** (shared skip REGRESS). Three strikes.

### 3. Structurally inert even if draft were faster

From **06-tps-ceiling §2** (identity under sequential verify + C4 overlap on greedy path):

- Draft wall is hidden under T₁ when C4 parallel works.
- A pure draft speedup surfaces as **≈ +0 t/s**.
- Accept penalty **does** surface — and is exactly the **−0.4 t/s** (+ joint inflation) measured.

The cancellation identity predicts the outcome in advance.

### Corner case not funded: RS serial draft

`mtp_speculative_step_sampled` runs the draft chain **serially** (no C4 overlap), so draft ms sit on the wall and contribute to RS’s −7% vs greedy.

A cheaper draft *could* help there **only if** accept held. At 0.72 accept, extra rejections buy more serial T=1 trunk calls (~T₁ each) that **eat** any draft saving several times over.

**Do not fund** RS×top_k A/B without evidence the accept penalty disappears at k=4+ — C14 says it will not for this head.

---

## Code changes on this branch (hygiene only)

| ID | Change |
|----|--------|
| **R-11 (P3)** | Rewrite `mtp_moe.cpp` C11 comment: cite measured REGRESS; flag = research opt-in |
| **P3 hygiene** | `effective_draft_top_k`: `getenv`+`atoi` once via `static const` lambda (not every forward) |
| Log string | One-shot banner notes “measured REGRESS on 35B gfx1150” when override active |

No default-on. No new TPS claim. No LoopBrake.

---

## Product / experiment status

| Item | Status |
|------|--------|
| Ship `MLX_MTP_DRAFT_TOPK=2` as default | **NO** |
| Fund C11 re-measure / k=4 ladder | **NO** (unless new head/architecture) |
| Fund RS×top_k | **NO** without accept-recovery theory |
| Keep env flag for research | **YES** (opt-in, documented dead on 35B/890M) |
| Sibling stack | Parent `fix/mtp-stream-p0`; sibling of S4 `exp/mtp-tps-ceiling` (batch also KILL) |

---

## Bottom line

Well-run experiment, **clean negative**. Dead on this machine for three independent reasons. Close C11 as a product lever; leave research flag + honest comments.
