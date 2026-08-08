# Path A — Research complete / freeze status (Redline kernel-launch)

**Date:** 2026-08-08  
**Branch (lemon-mlx):** `exp/redline-kernel-launch`  
**Redline fork:** [`antmikinka/redline`](https://github.com/antmikinka/redline) · branch `exp/hip-stream-bridge` (base: pwilkin; not warpfront)  
**Decision framework:** Path **A** = pause product gen pursuit and record honest status. Path **B** = optional further async PRE work (see §7). **User priority: B first, then A** — this doc **is** deliverable A.

---

## 1. One-line status

**Ownership research for OWN_GLUE + packed OWN_RMSNORM is done (opt-in, default OFF).**  
**Product gen speed-up via Redline is not achieved** (~**−3.5%** B1 vs B0 after phase1/phase2 fixes).  
**Do not product-default ON.** Prefer freeze (A) unless continuing async 2b (B).

---

## 2. What is done (keep)

| Item | Status | Evidence |
|------|--------|----------|
| Redline session, RUNPATH, XOR pure-graph | Done | P2+ |
| **OWN_GLUE** retained PM4 | Done | ≈ baseline gen |
| **OWN_RMSNORM** packed (~31/37) | Done | owns product HIP when armed |
| Launch inventory P11 | Done | ~395 L1 dispatches; QMM dominant |
| PRE/POST knobs P12b–d | Done | post ≈ free; join tax remains |
| HIP stream bridge **phase1** | Done | `rl_pm4_replay_after_hip_stream` · host StreamSynchronize |
| Phase2b WAIT_REG_MEM encoding fix | Done | gfx9+ mem_space; `phase2-used` proven |
| Measure discipline B0/B1/B2 | Done | no ≥2% win |
| Commits pushed | Done | lemon-mlx + antmikinka/redline |

### Recommended operator stacks (unchanged)

| Intent | Env |
|--------|-----|
| **Product / max gen** | all `MLX_REDLINE_*` **unset** |
| **Least-harm ownership** | `DECODE=1` `LIB` `OWN_GLUE=1` `GLUE_HSACO` only |
| **RMSNorm ownership lab** | + `OWN_RMSNORM=1` `RMS_HSACO` · PRE/POST unset · **no** `PHASE2` unless debugging |
| **Forbidden** | `DECODE=1` + `MLX_DECODE_GRAPH_PURE=1` |
| **Not for speed** | all-flags B2; `MLX_REDLINE_PHASE2=1` as default |

---

## 3. What is *not* done (honest)

| Item | Status |
|------|--------|
| Gen t/s ≥ **+2%** vs eager | **Not met** |
| Product default ON | **Forbidden** until gate met |
| PRE/join tax removed | **Still ~2.3 ms / 31 owns** host wall (join+RL) |
| Phase2 faster than phase1 | **No** (~−3.9% vs −3.4%) |
| All-flags + bridge matrix | Not required for A; historically B2 **~−9%** |
| qmm / full forward on Redline | Out of scope (E3) |
| Async host-free OWN (submit + consumer WaitValue wired in lemon) | Path **B** residual |
| Strided RMSNorm / CustomKernel ownership | Not owned |

---

## 4. Latest PRE / gen numbers (owning paths)

**TS 20260808-141111** · redline `9df1dfe` · both paths log `phase*-used`  
See also [`PRE_RETEST_20260808.md`](PRE_RETEST_20260808.md).

| Stack | Mean gen t/s (0.8B) | vs B0 |
|-------|--------------------:|------:|
| B0 baseline | **116.7** | — |
| B1 phase1 | **112.7** | **−3.4%** |
| B1 phase2 | **112.1** | **−3.9%** |

Host profile n=31 (bridge folds join into `replay` bucket):

| | set_k | pre timer | **join+RL (replay)** | post |
|--|------:|----------:|---------------------:|-----:|
| phase1 / phase2 | ~4 µs | ~124 µs | **~2.2–2.5 ms** | ~3 µs |

≈ **72–80 µs per OWN_RMSNORM** for producer wait + Redline kernel (not empty sync).

Earlier fail-open phase2 (before encoding fix): no OWN log, prompt ~10s — fixed by `9df1dfe` (gfx9+ `mem_space`).

---

## 5. Why gen still loses (first principles)

1. Product HIP chains RMSNorm on the **same stream** as matmul — host does not join each op.  
2. Redline is a **different queue** — need explicit order (phase1 StreamSynchronize or phase2 WriteValue+WAIT_REG_MEM).  
3. Host still waits for **producer tail + RL** before returning from OWN (sync phase2).  
4. Token time is still mostly **qmm/attn** — fixing RMSNorm launch alone may never hit +2%.  
5. Additive flags (SMALL_OP/SIDECAR) make gen **worse** (B2 ~−9%).

---

## 6. Path A actions (this document’s job)

| # | Action | Status |
|---|--------|--------|
| A1 | Record research-complete / freeze posture | **This file** |
| A2 | Keep product defaults **OFF** | Policy |
| A3 | Point operators at glue-only / baseline stacks | §2 |
| A4 | Point bridge work at antmikinka remote | §8 |
| A5 | Optional: open PR antmikinka → pwilkin for stream-bridge | Human |
| A6 | Optional: lemon-mlx PR as research-only, not product ON | Human |
| A7 | Stop thrashing flags / long empty loops | Policy |

**Path A does *not* delete code.** It freezes **product claims** and **default ON**.

---

## 7. Path B (deferred / optional after A)

If gen pursuit continues (user may do B first in a given session):

1. Wire lemon-mlx to `rl_pm4_submit_after_hip_stream_phase2` + `rl_gpu_consumer_wait_hip_stream`  
2. Host returns without full Redline `wait_signal`  
3. Remeasure B0 vs B1 — **kill** if still &lt; +2% with real `phase2-used` / async log  
4. Commit+push redline to **antmikinka** and lemon-mlx as required  

Symbols already exist on redline (`2922fac`+); lemon wire + measure may still be open.

---

## 8. Remotes and commits (reference)

| Tree | Remote / branch | Notes |
|------|-----------------|-------|
| redline | `origin` = antmikinka/redline · `exp/hip-stream-bridge` | Push required when bridge advances |
| lemon-mlx | lemonade-sdk · `exp/redline-kernel-launch` | Research branch |
| warpfront/redline | Reference only (~8 commits ahead on hipgraph) | Not primary |

Key redline commits (stream bridge era): phase1 `d65fd44` · phase2 API · WAIT_REG_MEM · encoding fix `9df1dfe` · async APIs `2922fac`.

---

## 9. Stop / kill criteria (program)

| Outcome | Action |
|---------|--------|
| B1 gen ≥ **+2%** vs B0, quality OK, real own log | Discuss product default ON for that path only |
| Async B still ≤ −2% or no own log | **Kill gen pursuit**; keep ownership opt-in |
| 3 empty fires / no new lever | Pause; revise roadmap |

---

## 10. Related docs

- [`PRE_RETEST_20260808.md`](PRE_RETEST_20260808.md) — latest PRE/gen  
- [`TRY_PHASE2B_20260808.md`](TRY_PHASE2B_20260808.md) — phase2 fail-open then fix  
- [`P13_STREAM_BRIDGE_PR.md`](P13_STREAM_BRIDGE_PR.md) — PR-A/B/C checklist  
- [`BRANCH_FREEZE_GEN.md`](BRANCH_FREEZE_GEN.md) — earlier gen freeze note  
- [`ROADMAP.md`](ROADMAP.md) · [`MASTER.md`](MASTER.md)  
- redline: `docs/HIP_STREAM_BRIDGE.md`

---

## 11. Bottom line (Path A)

**Ship posture:** baseline product path; Redline **off**.  
**Research posture:** glue ownership safe to demo; RMSNorm ownership works but **slows gen ~3–4%**.  
**Next only if chasing gen:** Path B async — not more flags.
