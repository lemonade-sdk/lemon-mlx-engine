# Redline continuous roadmap (living)

**Branch:** `exp/redline-kernel-launch`  
**Host target:** gfx1150 (890M) · lemon-mlx-engine ROCm  
**Revised:** 2026-08-08 (P12b POST_SYNC fence policy + redline-conscious-loop workflow)

---

## 0. North star (honest)

**Goal:** Make Redline **own real product decode launches** that are **launch-bound**, so gen t/s can rise vs eager HIP — without inventing TPS, without product default ON until measured ≥2%.

**Not the goal:** Flip every `MLX_REDLINE_*` flag (that adds work and often **slows** gen).  
**Not the goal yet:** Full model / qmm on Redline (E3: AOT qmm is pointer-launch, not drop-in HSACO).

---

## 1. First principles

| Fact | Implication |
|------|-------------|
| Token time ≈ **matmul + launch + sync** | Only replacing **costly** launches (or many small ones) moves gen t/s |
| E1/E2/P2: retained AQL ~**1.5–1.9×** vs system/HIP on **toy multi-dispatch** | Mechanism is real |
| E3: **qmm not drop-in** | “Redline everywhere” ≠ load qmm CO next week |
| All-flags ON gen A/B: **slower** | SMALL_OP/SIDECAR are **additive**; still full `call_fn` |
| P9–P10: **OWN_GLUE** replaces real product HIP glue | First true product-path ownership |

**Pareto:** ~80% of 35B token time is still compute (qmm/attn). Glue is **necessary infrastructure**, not the gen win by itself.

---

## 2. Done (keep; do not re-litigate)

| Phase | Status | Product ownership? |
|-------|--------|--------------------|
| E0–E4 | DONE | Design + floors |
| P0–P4 | DONE | Env OFF, floors, design |
| P5–P7b | DONE | In-proc session + sidecar correctness |
| **P8** SMALL_OP | DONE | Uses product VRAM; still **extra** PM4 |
| **P9–P10** OWN_GLUE retained | DONE | **Replaces** product pos/token glue HIP |
| **P11** launch inventory | DONE | 0.8B L=1 ≈ **395** dispatches; table in [`P11_LAUNCH_INV.md`](P11_LAUNCH_INV.md) |
| **P12** OWN_RMSNORM packed | DONE | Replaces **~31/37** RMSNorm product HIP (strided residual 6); [`P12_OWN_RMSNORM.md`](P12_OWN_RMSNORM.md) |
| Gen A/B 0.8B / 35B / all-flags | RUN | No win when additive flags on |
| Gen A/B OWN_GLUE only (M1) | RUN | ≈ baseline (glue too small vs 395) |
| Gen A/B OWN_RMSNORM only (M2) | RUN | 0.8B B1 ~−3–5% vs stable B0; B2 slower; [`GEN_AB_OWN_RMSNORM_20260808.md`](GEN_AB_OWN_RMSNORM_20260808.md) |
| Clean + retry2 rebench | RUN | B1 ~−2–3.5% / B2 ~−8–9%; lemonade still holds VRAM; [`GEN_AB_CLEAN_20260808.md`](GEN_AB_CLEAN_20260808.md) [`GEN_AB_RETRY2_20260808.md`](GEN_AB_RETRY2_20260808.md) |
| **P12b** POST_SYNC fence policy | CODE | `MLX_REDLINE_POST_SYNC=device\|stream\|off` (default **device**); tax A/B: post-sync **not** primary tax; [`P12B_SYNC_TAX.md`](P12B_SYNC_TAX.md) |
| **P12c** PRE_SYNC / set_k-before-pre | RUN | set_k-before-pre; profile pre~1.8ms/31; B1 −3.0%; PRE=off −1.6% (not ship); [`P12C_PRE_SYNC.md`](P12C_PRE_SYNC.md) |

---

## 3. Living roadmap (revise each loop)

### Track A — **Own more product launches** (primary)

| ID | Work | Success | Kill if |
|----|------|---------|---------|
| **P11** | **Launch inventory** per L=1 token | **DONE** — 395/token 0.8B; QMM 187, CustomKernel 90, RMSNorm 37, … | — |
| **P12** | Own **packed RMSNorm** multi-instance product launches (`OWN_RMSNORM=1`) | **DONE** — arm smoke PASS; inv 37→6; gen text OK; M2 DONE (no gen win) | Strided residual; mid-eval sync tax |
| **P12b** | Cut mid-eval **POST_SYNC** tax (`MLX_REDLINE_POST_SYNC=device\|stream\|off`) | **CODE** — default device; stream/off research-only; B1-off≈B1-device (~−2.6%) → post-sync **not** primary tax; [`P12B_SYNC_TAX.md`](P12B_SYNC_TAX.md) | Dual-queue races if off without events |
| **P12c** | Cut **pre**-stream / host set_k tax on OWN_RMSNORM (set_k-before-pre; `PRE_SYNC` knob) | **MEASURED** — pre dominates; B1 −3.0%; PRE=off recovers half (−1.6%) not shippable; [`P12C_PRE_SYNC.md`](P12C_PRE_SYNC.md) | — |
| **P12d** | Ordering: post=auto (replay wait); pre=query-then-sync | **CODE** — [`P12D_ORDERING.md`](P12D_ORDERING.md); same-queue still blocked | Residual pre_wait = dual-queue host bubble |
| **P13** | **HIP stream/event bridge** (drop host PRE) — Redline C-API + lemon wire | Spec: [`P13_STREAM_BRIDGE_PR.md`](P13_STREAM_BRIDGE_PR.md); **PR-A phase1** on redline `exp/hip-stream-bridge` (`rl_pm4_replay_after_hip_stream` host wait) — gen win needs phase2 | Phase2 blocked / no device wait |
| — | **Gen freeze** on this branch without bridge | [`BRANCH_FREEZE_GEN.md`](BRANCH_FREEZE_GEN.md) | — |
| **A** | **Research-complete / product freeze status** | [`STATUS_RESEARCH_COMPLETE_A.md`](STATUS_RESEARCH_COMPLETE_A.md) | — |
| **P13b** | **Encoder / CommandEncoder shim** (E4 option B) for JIT module launches only | Measured launch cut or KILL | Too invasive without win |
| **P14** | Revisit **qmm** only via recompile/export plan (E3 high friction) | Explicit design gate | Drop-in still impossible |

### Track B — **Measure honestly** (always)

| ID | Work | Notes |
|----|------|-------|
| **M1** | Gen A/B **OWN_GLUE only** (no SMALL_OP/SIDECAR) | Isolates ownership tax/benefit |
| **M2** | Gen A/B after P12 OWN_RMSNORM | **DONE** 0.8B B0/B1/B2 — B1 no win (sync tax); 35B optional later |
| **M3** | Never claim microbench µs as gen t/s | Hard ban |

### Track C — **Hygiene** (secondary)

| ID | Work |
|----|------|
| **H1** | Default all research flags OFF; document recommended stacks |
| **H2** | Rebuild checklist after every code change |
| **H3** | Deprecate “turn everything on” as a performance strategy |

---

## 4. Recommended stacks (operators)

| Intent | Env |
|--------|-----|
| **Product default** | *(all unset)* |
| **Own glue only** | `DECODE=1` `OWN_GLUE=1` `GLUE_HSACO=…` `LIB=…` |
| **Own packed RMSNorm** | `DECODE=1` `OWN_RMSNORM=1` `RMS_HSACO=…` `LIB=…` |
| **Correctness lab** | + `HSACO` + `SMALL_OP=1` (expect gen **slower**) |
| **Forbidden combo** | `DECODE=1` + `MLX_DECODE_GRAPH_PURE=1` (XOR) |

---

## 5. Stop / kill criteria (program)

| Outcome | Action |
|---------|--------|
| Owned product path gen t/s ≥ **+2%** vs eager, quality OK | Discuss product default / PR |
| Owned path regresses gen without quality gain | KILL that path; document |
| 3 empty fires / hard blocker ×2 | Pause loop; revise roadmap |
| qmm-only strategy | **Rejected** until export tool exists |

---

## 6. Continuous loop mandate

Each fire must:

1. Clear Thought (sequential + decision/scientific + metacog)  
2. Quintuple domain check (or simulated supervisor review)  
3. **One** net-new item from Track A or B (prefer A)  
4. Update this ROADMAP if priorities change  
5. Commit; no force-push; no fake TPS  

**Slogan (accurate):** *Replace product launches, don’t pile flags.*

---

## 7. Immediate next (this revision)

1. ~~**M1** OWN_GLUE-only gen A/B~~ — DONE (≈ baseline).  
2. ~~**P11** launch inventory~~ — DONE (395/L1 on 0.8B).  
3. ~~**P12** OWN_RMSNORM packed~~ — DONE (31/37 owned; strided 6 residual; mid-eval sync tax).  
4. ~~**M2** gen A/B OWN_RMSNORM~~ — DONE 0.8B: B1 **no ≥2% win** (~−3–5% stable pairs); B2 ~−13%; keep default OFF.  
5. ~~**P12b** POST_SYNC fence policy + tax isolation~~ — **CODE** (default device; off≈device → post not primary tax).  
6. ~~**P12c** set_k-before-pre + PRE_SYNC A/B~~ — **DONE** GTT-clear host: B1 −3.0%; PRE=off −1.6% (not ship); pre host ~1.8ms/31 owns.  
7. ~~**P12d** query-pre + POST auto~~ — **CODE**; measure B0/B1; same-queue still upstream.  
8. Own next residual (CustomKernel / strided) **or** Redline HIP-stream bind if API appears — still default OFF.  
8. Optional 35B B0/B1 when claiming 35B relevance (GPU free).

“Redline everywhere” = **grow the set of product ops that fall through to Redline**, op by op — not enable every research env at once.
