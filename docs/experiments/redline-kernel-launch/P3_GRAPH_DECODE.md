# P3 — `graph_decode` kernarg-patch integration (design doc)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **DESIGN DOC** (required for continuous-loop stop A with quality PASS)  
**Depends on:** P0 GREEN · P1 PASS · P2 PASS · E3/E4

---

## 1. Goal

Define how a **future** Redline retained-AQL subgraph would patch **stable device addresses** already owned by product decode (`graph_decode_input` / `graph_decode_pos`) without re-enabling product HIP graphs.

**Not this doc:** shipping a measured gen t/s win; replacing qmm; product default ON.

---

## 2. Product stable buffers (existing)

| Buffer | API | Lifetime / use |
|--------|-----|----------------|
| Input token | `graph_decode_input()`, `set_graph_decode_input_from(token)` | Fixed-address `[1,1] int32`; device copy each step |
| Position | `graph_decode_pos()`, `set_graph_decode_pos`, `advance_graph_decode_pos` | Fixed-address `[1] int32`; RoPE/KV offset |
| Mode | `gpu_set_graph_decode_mode(L==1)` | ROCm backend graph-decode flag (product path) |

**Invariant (same as pure-graph research):** retained IB / AQL packets bake **device pointers**. Any buffer that reallocates each token **breaks** replay. Prefer only pointers into:

1. `graph_decode_*` fixed arrays  
2. Model weights (immutable after load)  
3. KV slots written in-place at device pos (if product already does this)

---

## 3. What P3 can replace (honest)

| Candidate | Feasibility | Notes |
|-----------|-------------|-------|
| Toy floor dual-dispatch (P1 class) | **High** for microbench only | No model quality; proves session replay from engine |
| Small JIT elementwise (E3 path A) | **Medium** | Need symbol + grid for T=1; load from `MLX_REDLINE_HSACO_DIR` |
| Encoder launch shim (E4 option B) | **Medium–Hard** | Patch MLX CommandEncoder; deeper fork |
| qmm / gather_qmm / flash | **Low / deferred** | E3: not drop-in HSACO |
| Full `context_.call_fn` | **Out of scope** | Needs entire DAG ownership |

**v0 product-adjacent path:** keep `call_fn` for real compute; optionally interleave a **sidecar** Redline batch (instrumentation) between tokens when `MLX_REDLINE_DECODE=1` **and** session READY **and** `MLX_REDLINE_SIDECAR=1` (new, default OFF). Sidecar must not be reported as gen t/s win.

**v1 path (later):** replace one named engine micro-sequence if one exists — lemon-mlx has few custom HIP decode kernels, so expect MLX-side work.

---

## 4. Per-token sequence (design)

```text
TokenIterator::step L=1, MLX_REDLINE_DECODE=1, session READY:

  1. set_graph_decode_input_from(prev)     // product
  2. advance/set graph_decode_pos          // product (if external pos)
  3. if subgraph owns nodes:
       for node in fixed_list:
         rl_graphexec_set_node_kernargs(node, {ptrs...})  // C-API
       rl_graphexec_launch(gpu, exec)                     // or AQL replay
  4. context_.call_fn(...)                 // product — still owns qmm etc.
  5. sample                                // product
```

**Fence policy:** BoundarySerialized only (E1/E2). Never BoundaryIndependent for real decode.

**Fallback:** any Redline error → log once if `MLX_REDLINE_LOG=1`, continue product only (`MLX_REDLINE_FALLBACK=1` default).

---

## 5. P2 residual that blocks tight in-process bind

P2 measured: `rl_gpu_new(0)` **null** inside MLX-linked `chat` even pre-load; **non-null** in standalone C `dlopen` smoke.

| Mitigation | Use when |
|------------|----------|
| Out-of-process Redline worker + IPC | If HSA/HIP cannot coexist |
| Init ROCr before any HIP symbol load | May need separate launcher binary |
| Stay on AQL Rust harness for measure (P1 style) | Short-term truth for launch floor |
| C-API graph path after HIP already up | Needs Redline/ROCm investigation |

**P3 code must not claim in-process GPU bind until a smoke shows `gpu_new=ok` in the engine binary.**

---

## 6. Kill / pass (when implemented)

| Gate | Rule |
|------|------|
| Correctness | Greedy/fixed-seed match vs eager for N tokens |
| Perf | Gen t/s **only** from same-build A/B log vs eager |
| Research pass | ≥2% gen t/s **or** clear launch-count drop, no quality loss — else KILL |
| Ban | Do not re-label E1/E2 µs as model TPS |

---

## 7. Stop-rule A checklist

| Item | Status |
|------|--------|
| P0 green | **YES** |
| P1 green | **YES** |
| P3 doc | **THIS FILE** |
| Quality PASS on P3 doc | See `QUALITY_REVIEW_P3.md` |

Implementation of P3 micro-op remains **optional after** stop A; continuous loop may continue for P4 MoE multipath design or stop per scheduler rules.
