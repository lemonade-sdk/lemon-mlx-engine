# Redline × lemon-mlx-engine — research synthesis

**Branch:** `exp/redline-kernel-launch`  
**Parent:** `fix/mtp-stream-p0` @ `875a39d` (sibling of `exp/mtp-t1-lmhead-graph`, same parent)  
**Date:** 2026-08-02  
**Method:** Clear Thought (sequentialthinking, metacognitivemonitoring) + domain subagents (ROCm dispatch explore, engine launch map) + source read of **pwilkin/redline** (fork) / **warpfront/redline** (upstream).

**Local hardware context:** gfx**1150** (Radeon 890M / Strix Point), ROCm Core **7.13.0** under `/opt/rocm`.

---

## 0. Repo identity (important)

User URL typo: `github.com/pwilikin/redline` → **404**.

| URL | Status |
|-----|--------|
| **https://github.com/pwilkin/redline** | **Exists** — personal **fork** of the project |
| **https://github.com/warpfront/redline** | **Upstream** (Kaden Schutt / Hipfire lineage; more commits, hipGraph shims) |
| `pwilikin/redline` | Does not exist |

**Both** implement the same product: *“Lightning-fast kernel dispatch for ROCm”* via retained PM4 / ROCr queues. Prefer **warpfront** for latest integration work; pwilkin is fine for reading the same core design.

Subagent detail: [`SUBAGENT_ROCM_DISPATCH.md`](SUBAGENT_ROCM_DISPATCH.md).

---

## 1. What Redline is (and is not)

### Is

- Attacks the **HIP dispatch floor** ([ROCm/ROCm#6409](https://github.com/ROCm/ROCm/issues/6409)): at **tiny multi-kernel** workloads, HIP pays too much **per-launch fence + submit**.
- **Record once, replay many:** build a retained indirect buffer (PM4) / AQL path over **public ROCr/HSA queues**.
- Derives **minimal fences** from declared memory hazards (`BoundarySerialized` ~**1.8×** vs system-fence-every-dispatch in historical floor microbench — `docs/DISPATCH-FLOOR.md`).
- Integration surfaces:
  - **C ABI** `redline-capi` — best for C++ engines
  - Rust `redline-dispatch`
  - Python bindings
  - **`LD_PRELOAD` hipGraph interposer** (`redline-hipgraph`) — fall through for unsupported ops

### Is not

- Not a drop-in replacement for all of HIP.
- Not “turn on HIP graphs and go” — our own **decode HIP graph** path already **regressed** on this stack (`exp/mtp-t1-lmhead-graph` L4 −3.6%; pure VOID). Redline is a **different** lever (retained IB + fence policy), not product HIP-graph default-on.
- Not free for **data-dependent** graphs (MoE expert choice per token) without re-recording or multi-path IBs.

---

## 2. Why it matters for lemon-mlx-engine

From engine map ([`SUBAGENT_ENGINE_LAUNCH_MAP.md`](SUBAGENT_ENGINE_LAUNCH_MAP.md)):

| Today (product decode) | Implication |
|------------------------|-------------|
| MLX default **`use_hip_graphs() == false`** | Each op → **immediate `hipLaunchKernel` / module launch** |
| Commit ~every 2000 ops + host funcs | Many submissions / token |
| MoE: many small `gather_qmm` / layer | **Launch-bound friendly to Redline’s thesis** on 8 CU APU |
| lm_head full vocab qmm | Large compute; still sits in multi-kernel token DAG |
| HIP graph product decode | **Already killed** as default |

**Hypothesis (research, unmeasured here):** If a **fixed T=1 decode subgraph** (or fused expert set) can be retained and **kernargs patched** each token, Redline-style replay may cut host/fence overhead vs N HIP launches — **especially** where launch floor dominates (our MTP sequential T=1 story, MoE expert storms).

**Counter-hypothesis:** MLX already fuses some work; compute (not fence) may dominate for 35B matmuls → Redline win **smaller** than microbench no-ops (~1.8× on no-op chains).

---

## 3. Fit to **this** machine (gfx1150 / ROCm 7.13)

| Requirement | Local status | Risk |
|-------------|--------------|------|
| ROCm Core SDK **≥ 7.14** (TheRock) | Host **7.13.0** — **compile OK** (E0 2026-08-02) | 7.14 still preferred for optional HIP FFI + upstream product cert; **not** hard compile gate for dispatch/capi/hipgraph |
| Public AQL / ROCr | `libhsa-runtime64` present; **E1/E2 AQL measured** | Optional 7.14-only FFI still untested |
| Retained PM4 GFX11 | `gfx11*` → Gfx11 encoder | **Likely maps gfx1150** |
| Radiowave ArchProfile | Subagent: may treat **gfx1150 ≠ gfx1151** | Re-certify on 890M; don’t assume Strix Halo numbers |
| Published gfx1151 benches | Strong (Strix Halo) | **Transfer, not proof**, for gfx1150 |
| Floor HSACO gfx1150 | **Built** + **E1/E2 exercised** | PM4 example path still gfx12-only |

**Confidence we can *build* on 890M today:** **high** (E0 log).  
**Confidence of *dispatch-floor* win on no-op:** **high** (E1 GPU-span ~1.91×; E2 host wall ~1.5–1.6× vs HIP eager).  
**Confidence of *productive* gen-t/s win:** still **medium-low** until E3/E4 engine-shaped HSACO work (no-op ≠ 35B).

---

## 4. Integration paths (ranked for this engine)

| Path | Fit for lemon-mlx-engine | Notes |
|------|--------------------------|-------|
| **1. C ABI (`redline-capi`)** | **Best long-term** | Own buffers + HSACO + fixed dispatch list; `finalize` once, `set_kernargs` + `replay` per step |
| **2. Extract HSACO from MLX/hipcc for hot kernels** | Hard | Need symbols for QMM / fused packs; Radiowave optional cert |
| **3. hipGraph preload** | **Poor first choice** | We already avoid product HIP graphs; interposer complexity + fallthrough |
| **4. Full MLX backend rewrite** | Out of scope | Multi-quarter; not this experiment branch |

### Conceptual decode hook (research only)

```text
prepare fixed T=1 graph (or multi-path set for MoE routing buckets)
each token:
  pack kernargs (weights, KV pos, activations)  // stable addresses help (we already have graph_decode_* buffers)
  redline_replay(ib)
  host sample / argmax
```

Stable address work in `graph_decode.cpp` is **aligned** with Redline’s “patch kernargs between replays” model — even though HIP graph product is OFF.

---

## 5. Relation to prior MTP / T₁ experiments

| Experiment | Lesson for Redline |
|------------|-------------------|
| Sequential MTP T=1 | Launch/dispatch cost is real; free draft under verify |
| S4 batch verify KILL | Multi-token / topology change is expensive — Redline wants **fixed** DAG |
| Decode HIP graph −3.6% | Don’t re-enable product HIP graphs; Redline is **not** that path |
| lm_head two-stage +2–4% | Complementary (less work per launch), not a substitute for dispatch floor |
| MoE top_k / expert set | **Data-dependent** experts break single retained IB unless multipath or re-record |

---

## 6. Experiment plan (this branch)

| ID | Experiment | Pass / fail |
|----|------------|-------------|
| **E0** | Build Redline (warpfront) against local ROCm; note 7.13 vs 7.14 | **BUILD_OK** — see [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md) |
| **E1** | Run `dispatch_floor` / hipfire-6409 microbench on **gfx1150** if ROCm allows | **AQL MEASURED** — see [`E1_FLOOR.md`](E1_FLOOR.md) (~1.91× BoundarySerialized vs system-every); PM4 example tail N/A on gfx1150 |
| **E2** | MoE-shaped **fixed** N-launch chain (toy) retained vs HIP | **MEASURED** — [`E2_MULTI.md`](E2_MULTI.md) (~1.5–1.6× host wall BoundarySerialized vs HIP eager; hipGraph ≈ eager) |
| **E3** | Inventory whether MLX can export HSACO for one QMM | **DONE** — drop-in **not** feasible; JIT CO yes; see [`E3_HSACO.md`](E3_HSACO.md) |
| **E4** | Design-only `MLX_REDLINE_DECODE=1` hook sketch (no product default) | Design doc only until E1 green |

**HARD BAN:** No fake TPS; no claiming Redline wins without logs on **this** GPU; no re-opening killed HIP-graph product decode.

---

## 7. Decision (now)

| Question | Answer |
|----------|--------|
| Is Redline relevant? | **Yes** — same problem class as APU launch-bound decode |
| Immediate plug-in to product decode? | **No** — ROCm version, arch cert, MLX HSACO ownership |
| Worth a sibling experiment branch? | **Yes** — this branch |
| Prefer upstream remote | **warpfront/redline** (pwilkin = fork for reading) |
| Parallel with lm_head C1? | **Yes** — orthogonal (compute cut vs dispatch cut) |

**Overall confidence:** **0.82** on mechanism understanding; **0.90** on compile feasibility on 7.13 (E0); **0.55** on near-term gen-t/s win on gfx1150 (unmeasured).

---

## 8. Artifacts

| File | Role |
|------|------|
| [`SUBAGENT_ROCM_DISPATCH.md`](SUBAGENT_ROCM_DISPATCH.md) | Redline architecture + claims |
| [`SUBAGENT_ENGINE_LAUNCH_MAP.md`](SUBAGENT_ENGINE_LAUNCH_MAP.md) | lemon/MLX launch path map |
| [`RESEARCH.md`](RESEARCH.md) | This synthesis |
| Local clones (not in git) | `/tmp/redline-pwilkin`, `/tmp/redline-warpfront` |
| [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md) | E0 compile + HSACO evidence |
| [`E1_FLOOR.md`](E1_FLOOR.md) | E1 AQL µs/dispatch on gfx1150 |
| [`E2_MULTI.md`](E2_MULTI.md) | E2 HIP wall vs retained AQL |
| [`E3_HSACO.md`](E3_HSACO.md) | MLX qmm / JIT HSACO feasibility |
| [`harness/`](harness/) | E2 HIP + AQL host-wall sources |
| [`INSTALL_UPGRADE.md`](INSTALL_UPGRADE.md) | 7.13 vs 7.14 upgrade notes |
| [`logs/`](logs/) | E0–E2 logs + floor CO |

## 9. References

- https://github.com/warpfront/redline  
- https://github.com/pwilkin/redline (fork)  
- https://github.com/ROCm/ROCm/issues/6409  
- Redline `docs/INTEGRATION.md`, `docs/DISPATCH-FLOOR.md`
