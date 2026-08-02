# Quality review — redline-kernel-launch docs

**Reviewer:** Taylor Kim (quality pass)  
**Date:** 2026-08-02  
**Scope:** `RESEARCH.md`, `SUBAGENT_ROCM_DISPATCH.md`, `SUBAGENT_ENGINE_LAUNCH_MAP.md` (+ README/MASTER for consistency)  
**Method:** Cross-check vs local Redline trees (`/tmp/redline-warpfront`, `/tmp/redline-pwilkin`), MLX ROCm `use_hip_graphs()`, prefill-hip-graph F1–F3, `generate.cpp` pure-graph comments, host ROCm layout.

**Overall verdict:** **PASS (with residual risks)** — research docs are honest, non-productizing, and correctly separate Redline retained-PM4 from killed HIP-graph product paths. No gen t/s win is claimed for this GPU.

---

## Checklist

| # | Criterion | Result | Notes |
|---|-----------|--------|-------|
| 1 | **pwilkin vs warpfront identity** | **PASS** | README/RESEARCH/MASTER: typo `pwilikin` → **pwilkin** fork; **warpfront** upstream. Local trees confirm warpfront has `redline-hipgraph/src/shims/*` + ownership redesign doc; pwilkin lacks both. Prefer-warpfront guidance correct. |
| 2 | **Redline claims accuracy** | **PASS** | Hipfire gfx1151 three-way: 185/240 firsts, RL/HIP median **0.3730**, 233/240 RL>HIP match `REPORT.md`. HipEngine gfx1151: 164/224 vs Vulkan, 182/224 vs HIP match. Historical ~1.8× BoundarySerialized / ~1.6× vs hipGraph / ~10–12× PM4 host floor match `DISPATCH-FLOOR.md` and are labeled **historical / methodology**, not current product scorecard. |
| 3 | **ROCm 7.13 vs ≥7.14 honesty** | **PASS** | Host has TheRock-style **`/opt/rocm/core-7.13`**. Docs treat Redline **≥ 7.14** as **blocker** (E0 pending); confidence medium-low until upgrade; no claim that 7.13 already runs Redline productively. Matches warpfront README hard requirement. |
| 4 | **No overclaim of gen t/s wins** | **PASS** | RESEARCH: hypothesis unmeasured; counter-hypothesis; confidence **0.55** on near-term gen-t/s; **HARD BAN** fake TPS / wins without this-GPU logs. SUBAGENT success bar (≥5% / ≥10%) is **future criteria**, not a result. MASTER: “no fake speed claims.” |
| 5 | **HIP graph product OFF alignment** | **PASS** | `use_hip_graphs()` default **OFF** (env opt-in only) confirmed in `device.cpp`. Engine map non-goals kill default graphs, pure decode (`MLX_DECODE_GRAPH_PURE`), prefill F1–F3 productization. README: “Decode HIP graphs remain product OFF (separate from Redline retained-PM4).” Path B/E5 preload framed optional / after proof, not product default. |
| 6 | **gfx1150 ≠ gfx1151 transfer caution** | **PASS** | Radiowave `from_arch("gfx1150")` is `None` (tests reject bleed from Gfx1151). Docs require re-certify on 890M; PM4 family `gfx11*` mapping stated correctly. |
| 7 | **Prefill / pure-graph prior evidence** | **PASS** | F1 ~+2.7% pp/s, F2 ~+3.6%, F3 fail ≥10% bar — consistent with prefill experiment results. Pure ~68 eager vs ~64 pure matches `generate.cpp` product comment. |
| 8 | **Citation hygiene for sibling L4 −3.6%** | **FAIL (minor)** | RESEARCH cites `exp/mtp-t1-lmhead-graph` **L4 −3.6%** / pure VOID, but **no probe log or RESULTS doc is in this tree** under that name. Pure-graph regression is independently backed; the **−3.6% L4 figure is not verifiable here**. Fix: attach log path or soften to “sibling branch measured decode-graph regress; pure path regressed vs eager (~68→~64).” |
| 9 | **“~1.4k inline launches”** | **FAIL (minor)** | Engine map asserts ~1.4k graph-split inline ops “historically”; **no in-repo comment/probe found** under this review. Direction (residuals split graphs) is plausible; **count should be sourced or marked estimate**. |
| 10 | **Dispatch-count wording** | **PASS w/ risk** | “Thousands of host→device dispatches per token” is qualitative, not measured on 35B in these docs. Acceptable as thesis framing if E1/E3 will count; do not promote to hard fact in product copy. |

---

## What the pack does well

1. **Mechanism vs product:** Redline = retained IB + fence policy over public ROCr; **not** “turn on HIP graphs.”  
2. **Version gate first-class:** 7.13 host vs 7.14 requirement; E0/E1 pending explicit.  
3. **Identity / provenance:** fork lag called out; trademark/NOTICE notes present.  
4. **Kill criteria:** no re-opening killed HIP-graph product decode; success vs **eager** baseline only.  
5. **Scorecard discipline:** microbench ratios separated from end-to-end gen t/s.

---

## Residual risks (do not block doc PASS)

| Risk | Severity | Mitigation |
|------|----------|------------|
| No live E0/E1 on this host yet | High for *implementation* | Keep docs research-only until 7.14 + gfx1150 smoke |
| gfx1151 micro wins may not transfer to 35B MoE T=1 | High | E2 fixed chain → E4 subset only; kill if &lt;2% gen t/s |
| Data-dependent MoE experts break single retained IB | Med | Docs already flag multipath / re-record |
| Path B `LD_PRELOAD` hipGraph still tempting | Med | Keep ranked **last**; prefer C ABI |
| Commit counts (44 vs 36) will drift | Low | Re-check remotes before pin |
| Parent SHA `875a39d` is branch cut point, not tip of full MTP archive | Low | Document as fork-point, not product tip |

---

## Required doc fixes (optional before merge)

1. **RESEARCH.md §1 / §5:** Source or soften **L4 −3.6%**.  
2. **SUBAGENT_ENGINE_LAUNCH_MAP.md:** Source or qualify **~1.4k** inline launches.  
3. (Nice) Add one line: “Dispatch counts and gen t/s: **unmeasured on gfx1150 for Redline as of this research pass.**”

---

## Decision

| Gate | Status |
|------|--------|
| Safe as **research synthesis** for `exp/redline-kernel-launch` | **PASS** |
| Safe as **product performance claim** | **N/A / FAIL if used that way** — docs correctly refuse this |
| Blockers before engine wire | ROCm ≥ 7.14, E0–E1 on gfx1150, no HIP-graph product default-on |

**Sign-off:** Docs package is **quality-acceptable** for experiment research with **two minor citation FAILs** that do not reverse the main honesty gates (identity, ROCm version, no gen t/s overclaim, HIP graph product OFF).
