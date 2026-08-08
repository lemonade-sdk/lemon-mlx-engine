# P3 — `graph_decode` buffers & kernarg-patch integration

**Date:** 2026-08-07  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **DESIGN DOC COMPLETE** + measured micro-op **PASS** ([`P3_MICRO_OP.md`](P3_MICRO_OP.md))  
**Depends on:** P0 GREEN · P1 GREEN · P2 N-sweep GREEN · E3/E4  
**Not shipped:** product forward replacement, gen t/s win, product default ON, decode HIP-graph re-enable.

---

## 1. Goal

Document how a **future** Redline **retained AQL** subgraph must integrate with lemon-mlx **fixed-address** decode buffers (`graph_decode_input` / `graph_decode_pos`) so kernargs stay valid across replays — without treating Redline as product HIP graphs.

---

## 2. Product stable buffers (source of truth)

### 2.1 Headers

[`include/mlx-lm/common/graph_decode.h`](../../../include/mlx-lm/common/graph_decode.h):

| API | Role |
|-----|------|
| `graph_decode_pos()` | Persistent `[1] int32` device scalar |
| `set_graph_decode_pos(int)` | In-place write absolute offset |
| `advance_graph_decode_pos(int)` | In-place increment (between replays) |
| `graph_external_pos()` / `set_graph_external_pos(bool)` | Loop owns pos vs eager |
| `graph_decode_input()` | Persistent `[1,1] int32` token buffer |
| `set_graph_decode_input_from(array&)` | Device copy sampled token → fixed buffer |
| `graph_capturing()` / `set_graph_capturing` | Capture/replay bookkeeping |

### 2.2 Implementation — address stability

[`src/common/graph_decode.cpp`](../../../src/common/graph_decode.cpp):

| Lines | Fact |
|-------|------|
| 22–28 | `graph_decode_pos()`: **lazy static** `new array(zeros({1}))` + `eval` — device addr fixed after first use |
| 31–42 | `set_graph_decode_pos`: ROCm `gpu_kv_pos_set` **in-place** (no realloc) |
| 46–52 | `advance_graph_decode_pos`: ROCm `gpu_kv_pos_increment` in-place |
| 60–66 | `graph_decode_input()`: lazy static `zeros({1,1})` — fixed after first use |
| 69–76 | `set_graph_decode_input_from`: ROCm `gpu_scalar_copy_i32` into fixed dst |
| 7–8 | Comment: mutate **without reallocating** so captured graph baked addresses stay valid |

**Redline invariant (same as pure-graph research):** AQL/IB packets bake **device pointers**. Any per-token reallocation **breaks** retained replay.

### 2.3 Where the product loop patches them

[`src/common/generate.cpp`](../../../src/common/generate.cpp) `TokenIterator::step_pure_graph` (research pure-graph path; **not** Redline):

| Lines (approx) | Action |
|----------------|--------|
| 636 | Feed `LMInput::Text(graph_decode_input())` into forward |
| 640 | `gpu_set_graph_decode_mode(false)` for loop-owned immediate ops |
| 646 | `set_graph_decode_input_from(prev_tok)` each token |
| 648–655 | Warmup: `set_graph_external_pos(true)` + `set_graph_decode_pos(off)`; else `advance_graph_decode_pos(1)` |

Eager `TokenIterator::step` (~598–610): only `gpu_set_graph_decode_mode(Lstep==1)` + product `call_fn` — **does not** patch fixed buffers unless pure path is on.

Redline opt-in today (P0/P2b): `maybe_log_redline_session_status()` on L=1 / `next()` — **no** kernarg patch yet.

---

## 3. What may enter a Redline subgraph vs stays HIP

| Class | Owner | Rationale |
|-------|--------|-----------|
| Fixed T=1 small ops (JIT elementwise / toy floor CO) | **Redline AQL** candidate | P1/P2 measured load+replay; launch-floor oriented |
| Product **qmm / gather_qmm / flash / lm_head** | **HIP product** | E3: AOT `hipLaunchKernel`, not drop-in HSACO |
| Full `context_.call_fn` DAG | Product until full ownership | lemon-mlx has few custom decode HIP kernels |
| MoE expert choice | **Not** single retained IB | Topology data-dependent → P4 multipath later |

**v0 integration shape (sidecar / partial):**

```text
TokenIterator L=1, MLX_REDLINE_DECODE=1, session READY:

  // Product stable patch (required if Redline nodes read these ptrs)
  set_graph_decode_input_from(prev)          // graph_decode.cpp:69
  set/advance graph_decode_pos as pure path  // :31 / :46

  // Redline-owned nodes only (future)
  for node in fixed_list:
    patch kernargs → { input_ptr, pos_ptr, weight_ptr, act_ptr, ... }
  replay BoundarySerialized batch            // E1/E2/P2 policy

  // Product still owns bulk compute
  context_.call_fn(...)                      // qmm etc. HIP
  sample
```

Optional future env (default OFF, **not implemented**): `MLX_REDLINE_SIDECAR=1` for instrumentation-only dual path — **must not** be reported as gen t/s.

---

## 4. Kernarg patch contract

| Requirement | Detail |
|-------------|--------|
| Prefer stable ptrs | `graph_decode_input()` / `graph_decode_pos()` data_ptr after first eval |
| Weights | Immutable after load — safe to bake |
| Activations / KV | Only if product writes **in-place** at fixed slots (pure-graph GDN/KV discipline) |
| Fence | **BoundarySerialized** only for real decode (E1/E2; P2 N-sweep) |
| N dispatches | Profiling AQL batch needs **N≥2** (P1 `InvalidBatchShape` lesson) |
| Fallback | `MLX_REDLINE_FALLBACK=1` default → product HIP on any error |

**In-process residual (P2b):** `rl_gpu_new(0)` may be **null** inside MLX-linked `chat` while standalone `dlopen` succeeds — see [`P2_INIT.md`](P2_INIT.md). **Do not claim engine-bound GPU replay until `gpu_new=ok` smoke exists.** Out-of-process harness (P1/P2) remains the measured launch-floor truth.

---

## 5. Explicit non-goals / hard bans

| Ban | Why |
|-----|-----|
| Product default ON | E4 + loop rules |
| Re-enable product decode HIP graphs via Redline | L4 KILL; separate envs |
| Couple to `MLX_USE_HIP_GRAPHS` / pure-graph without XOR fail-closed | P0/P2b |
| Claim E1 1.91× / E2 1.59× / P2 1.80× as model gen t/s | Wrong metric |
| qmm unbundle without recompile path | E3 |

---

## 6. Kill / pass when someone implements code

| Gate | Rule |
|------|------|
| Correctness | Greedy or fixed-seed match vs eager, N tokens, same model |
| Perf | Gen t/s **only** from this-GPU same-build A/B vs eager |
| Research pass | ≥2% gen t/s **or** clear launch-count drop, no quality loss — else **KILL** |
| Ban | No re-label of host-µs floors as TPS |

---

## 7. Stop-rule A checklist

| Item | Evidence |
|------|----------|
| P0 green | [`P0_STUB.md`](P0_STUB.md) + `logs/p0-*-20260807-215209.err` |
| P1 green | [`P1_LOAD.md`](P1_LOAD.md) + `logs/p1-load-hsaco-20260807-215318.log` |
| gfx1150 logs | E0–E2, P0–P2 under [`logs/`](logs/) |
| **P3 doc** | **This file** |
| Quality PASS | [`QUALITY_REVIEW_P3.md`](QUALITY_REVIEW_P3.md) |
| **P3 measured micro-op** | [`P3_MICRO_OP.md`](P3_MICRO_OP.md) + [`QUALITY_REVIEW_P3_MICRO.md`](QUALITY_REVIEW_P3_MICRO.md) — host µs + correctness PASS |

Out-of-process micro-op **landed** (acc_k patch+replay). Product TokenIterator wire remains research-only and never product-default-on; P4 MoE multipath is separate.
