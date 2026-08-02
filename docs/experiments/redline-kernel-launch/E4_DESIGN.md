# E4 — `MLX_REDLINE_DECODE` design sketch (env **OFF** by default)

**Date:** 2026-08-02  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **DESIGN ONLY** — no product enablement, no gen t/s claim, no decode HIP-graph re-open.  
**Prerequisites satisfied:** E0 BUILD_OK · E1 AQL measured on gfx1150 · E2 host wall measured · E3 HSACO inventory.

---

## 0. One-sentence goal

Provide an **opt-in research hook** (`MLX_REDLINE_DECODE=1`) that can eventually replay a **fixed T=1 decode subgraph** via Redline **retained AQL + BoundarySerialized**, while **leaving product qmm on HIP** and keeping **all product defaults OFF**.

---

## 1. Evidence that bounds the design

| ID | Fact used by design |
|----|---------------------|
| **E0** | Redline dispatch/capi builds on ROCm Core **7.13** / gfx1150 ([`E0_HOST_BUILD.md`](E0_HOST_BUILD.md)) |
| **E1** | AQL BoundarySerialized ~**1.91×** vs system-every on no-op GPU span ([`E1_FLOOR.md`](E1_FLOOR.md)) |
| **E2** | Host wall BoundarySerialized ~**1.5–1.6×** vs HIP eager; **hipGraph ≈ eager** ([`E2_MULTI.md`](E2_MULTI.md)) |
| **E3** | Product **qmm is not drop-in HSACO**; JIT `.hsaco` exists; doors = recompile slice / encoder shim ([`E3_HSACO.md`](E3_HSACO.md)) |
| **Prior** | Product decode HIP graphs **killed**; pure capture regressed — Redline is **not** re-enabling that path |

**Implication:** v0 must **not** depend on unbundling `qmm.o`. Win surface is **launch-floor** for **fixed multi-dispatch** chains; large matmuls stay HIP until a later explicit recompile path.

---

## 2. Architecture choice (decision)

| Option | Role |
|--------|------|
| **A. redline-capi + fixed small-op subgraph** | **Primary v0** |
| **B. CommandEncoder launch shim** | Phase-2 research (deeper MLX patch) |
| **C. `LD_PRELOAD` redline-hipgraph** | **Rejected first path** (graph-shaped, product risk) |
| **D. qmm offload unbundle** | Deferred (E3 high friction) |

**Primary path A** matches E1/E2 (proven AQL) and E3 (no qmm export).

```text
                    ┌─────────────────────────────────────┐
  TokenIterator     │  MLX_REDLINE_DECODE != 1 (default)  │──► product eager HIP
  step() L=1  ──────┤                                     │
                    │  =1 → RedlineDecodeSession (opt-in) │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
   patch graph_decode_*     Redline retained AQL      HIP qmm / rocBLAS
   input + pos (stable)     BoundarySerialized        (unchanged product)
           │                 small-op chain only
           │                       │
           └──────────► logits sample / MTP verify (host) ◄─┘
```

---

## 3. Environment contract (all default OFF)

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_REDLINE_DECODE` | **unset/0** | Master switch; anything else only when exactly `1` (match pure-graph discipline) |
| `MLX_REDLINE_HSACO_DIR` | unset | Directory of prebuilt CO files for v0 subgraph |
| `MLX_REDLINE_POLICY` | `BoundarySerialized` | Fence policy name; never default Independent for decode |
| `MLX_REDLINE_LOG` | `0` | If `1`, log record/replay counts (no TPS invention) |
| `MLX_REDLINE_FALLBACK` | `1` | On any Redline error → silent product HIP path for that step |

**Hard exclusions (must not couple):**

- Do **not** set `MLX_USE_HIP_GRAPHS` / `MLX_HIP_GRAPH_DECODE` / `MLX_DECODE_GRAPH_PURE` as part of Redline enable.  
- Do **not** interpret Redline as “turn graphs on.”  
- XOR with pure-graph if both set: **prefer fail-closed** (log once, stay eager) until measured.

---

## 4. Engine integration points (existing code)

| Site | File | Role in design |
|------|------|----------------|
| L=1 step | `src/common/generate.cpp` `TokenIterator::step` | After `gpu_set_graph_decode_mode(Lstep==1)`, gate optional Redline session |
| Stable token buf | `src/common/graph_decode.cpp` `graph_decode_input` / `set_graph_decode_input_from` | Kernarg patch source (stable device addr) |
| Stable pos | `graph_decode_pos` / `set_graph_decode_pos` / `advance_graph_decode_pos` | Same for RoPE/KV device pos |
| Prefill | `prepare` paths | **Out of scope v0** — Redline decode only when L=1 gen |
| MTP | draft/verify loops | Optional later; v0 = main `step` only |

**Non-integration (explicit):**

- `device.cpp` `use_hip_graphs()` remains independent and default **false**.  
- No change to product default launch policy without a measured A/B log on gfx1150.

---

## 5. Component sketch (pseudocode — not shipped)

### 5.1 Session lifecycle

```text
class RedlineDecodeSession:  // process-lifetime, lazy init
  enabled = (getenv("MLX_REDLINE_DECODE") == "1")
  if !enabled: no-op forever

  init_once():
    load libredline_dispatch / redline-capi (dlopen)
    load HSACO set from MLX_REDLINE_HSACO_DIR  // fixed list
    build SingleQueueBatchGraph OR rl_pm4_* only if arch supports
       prefer AQL path on gfx1150 (E1: PM4 example was gfx12-only)
    policy = BoundarySerialized
    record fixed dispatch list (see §6)
    state = READY | FAILED → if FAILED and FALLBACK, disable for process

  on_token_step(token_array, pos):
    set_graph_decode_input_from(token)
    set/advance graph_decode_pos as product does
    for each node: set_kernargs(ptr pack)  // stable addresses preferred
    replay_and_wait()
    // then product still runs MLX forward for ops NOT in subgraph
    // v0: subgraph is *partial* — see §6
```

### 5.2 Partial-forward reality (critical)

v0 **cannot** replace full `context_.call_fn` until all kernels in the DAG are Redline-owned.

**v0 mode = “sidecar / partial”:**

| Subgraph | Owner |
|----------|--------|
| Fixed list of **launch-heavy small** kernels we recompiled or JIT-captured for smoke | Redline AQL |
| **qmm / gather_qmm / lm_head / flash** | **HIP product path unchanged** |

That means v0 may be:

1. **Instrumentation-first:** record timing of a synthetic Redline chain between tokens (dev only), or  
2. **True partial:** replace a **named engine-owned** micro-sequence (if any) — lemon-mlx has almost no custom HIP decode kernels (E3), so partial product win needs MLX patches (phase-2 shim) or recompiled slices called from a thin engine path.

**Honest sequencing:**

| Phase | Deliverable | Success bar |
|-------|-------------|-------------|
| **P0** | Env parse + log “enabled but no-op” in `generate.cpp` behind `#if MLX_BUILD_ROCM` | Binary with `=1` prints once; default silent |
| **P1** | Out-of-process or tool path: load one MLX JIT `.hsaco` or toy CO in Redline (already E1/E2 class) | Correctness gate only |
| **P2** | Engine links `redline-capi`; session init; **no** change to forward | Init smoke on gfx1150 |
| **P3** | Replace **one** fixed recompiled micro-op OR encoder shim for JIT module launches only | ≥ measurable host launch cut **or** kill |
| **P4** | Optional multipath MoE / re-record | Separate experiment |

**This E4 document completes design through P0–P3 sketch; implementation of P0+ is future work (not this fire).**

---

## 6. Fixed subgraph content (what to put in the IB)

### v0 candidate set (launch-floor oriented)

Prefer ops that:

- Are **module/HSACO** launchable (JIT path) **or** recompiled with known `.kd`  
- Have **stable** grid/block for T=1  
- Are **not** data-dependent expert choice

Examples (illustrative, not committed):

- Fused elementwise from JIT cache (format-feasible, E3 path A)  
- Tiny custom “token glue” if engine adds them later  
- **Not** first: full `gather_qmv_expert_batched_kernel` (routing-dependent)

### Kernarg patching

Reuse product stable buffers:

```text
each token:
  set_graph_decode_input_from(prev)
  set_graph_decode_pos / advance_graph_decode_pos
  redline set_kernargs(node_i, {act_ptr, w_ptr, ...})  // only for nodes we own
  redline replay
```

Addresses that reallocate every step **break** retained IB — same constraint that motivated `graph_decode_*`.

---

## 7. Fence policy

| Policy | Use |
|--------|-----|
| **BoundarySerialized** | **Default decode** (dependency-safe; E1/E2 winner class) |
| SystemEveryDispatch | Debug baseline only |
| BoundaryIndependent | **Forbidden** for real decode chains (E1 doc) |

PM4 IB: library supports gfx11 family; **do not** use the `dispatch_floor` example’s hardcoded Gfx12 path on gfx1150 (E1 fail). Prefer **AQL `SingleQueueBatchGraph`** for 890M v0.

---

## 8. Build / link sketch (future)

```cmake
# conceptual — NOT in product CMake yet
option(MLX_LM_WITH_REDLINE "Link redline-capi for experiment" OFF)
if(MLX_LM_WITH_REDLINE)
  # CARGO_TARGET_DIR=/tmp/redline-warpfront-target
  # target_link_libraries(... redline_dispatch)
endif()
```

Runtime without link: **dlopen** `libredline_dispatch.so` only when env=1 (keeps default binary free of Redline).

---

## 9. Kill / pass criteria (when someone implements)

| Gate | Rule |
|------|------|
| Correctness | Greedy or fixed-seed match vs eager for N tokens on test model |
| Perf | Report gen t/s **only** from this GPU log vs **eager** baseline same build |
| Pass (research) | ≥ **2%** gen t/s **or** clear launch-count drop with no quality loss — else **KILL** |
| Pass (product discuss) | ≥ **5–10%** + stability — out of scope until measured |
| Ban | No claiming E1 1.91× or E2 1.5× as gen t/s |

---

## 10. Stub pseudocode for `generate.cpp` (P0 only)

```cpp
// DESIGN SKETCH — do not treat as committed product code.
// Placement: TokenIterator::step, ROCm only, before/around call_fn when Lstep==1.
static bool redline_decode_enabled() {
  const char* v = std::getenv("MLX_REDLINE_DECODE");
  return v && v[0] == '1' && v[1] == '\0';
}
// if (redline_decode_enabled()) {
//   static std::once_flag once;
//   std::call_once(once, [] {
//     std::cerr << "[redline] MLX_REDLINE_DECODE=1: session not implemented (design only)\n";
//   });
//   // future: RedlineDecodeSession::get().on_token_step(...);
// }
```

**Default path unchanged** when env unset. No HIP graph side effects.

---

## 11. Risks & non-goals

| Risk | Mitigation |
|------|------------|
| Partial subgraph adds sync without removing enough launches | Measure; FALLBACK=1; kill if regress |
| MoE expert set changes topology | Multipath / re-record later; v0 avoid expert kernels |
| Conflating with HIP graphs | Separate env; docs ban; quality review |
| ROCm 7.14 optional FFI | Not required for AQL path (E0/E1) |
| Shipping stub that looks like a feature | Log “not implemented” until P2+ |

**Non-goals:** product default ON; full model on Redline; hipGraph preload; LoopBrake/MTP changes; lm_head C1 thrash.

---

## 12. Board completion for E4

| Item | Status |
|------|--------|
| Design doc | **This file** |
| Env name + default OFF | **Specified** |
| Integration sites | **Cited** (`generate.cpp`, `graph_decode.cpp`) |
| Phased plan | **P0–P4** |
| Product wire / stub in binary | **Not shipped** (design sketch only) |
| Gen t/s claim | **None** |

**E4 healthy/done criteria:** design landed behind env-OFF semantics — **met** by this document.
