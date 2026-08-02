# Engine launch map — ROCm decode (Redline experiment prep)

**Branch:** `exp/redline-kernel-launch` (parent: `fix/mtp-stream-p0`)  
**Scope:** how kernels actually leave the host on **decode (T=1)** today, where launch/dispatch cost concentrates, and where a Redline-style **tiny-dispatch / command-buffer replay** would hook in.  
**Stack:** lemon-mlx-engine + FetchContent MLX under `build/_deps/mlx-src` (ROCm backend).  
**Target class:** gfx1150 / 890M APU (8 CUs) and launch-bound dGPUs.

---

## 0. Executive summary

| Question | Answer on this stack (product defaults) |
|----------|------------------------------------------|
| How do decode kernels launch? | **Eager immediate** `hipLaunchKernel` / `hipModuleLaunchKernel` per MLX primitive via `CommandEncoder` |
| Are HIP graphs on? | **No** — `use_hip_graphs()` default **OFF** (env opt-in only) |
| Is pure stream-capture decode on? | **No** — `MLX_DECODE_GRAPH_PURE=1` only; **regresses** on gfx115x APU vs eager |
| Does `gpu_set_graph_decode_mode(true)` batch launches? | **Only if** graphs env is also on; otherwise the flag is largely **inert** for launch batching |
| Dominant cost class (35B MoE) | **Quantized matmuls** (attn, MoE experts, lm_head) — compute + per-op launch count |
| Secondary launch tax | MoE `gather_qmm` × experts × projs; elementwise/norm; worker host callbacks |
| Redline-relevant bottleneck | **Thousands of host→device dispatches per token**, many short-lived kernels on few CUs |

**Non-goal (hard):** re-enable or productize **decode HIP graphs** / pure capture-once as the default path — already measured and killed on this stack (see §5).

---

## 1. Current path (product decode)

### 1.1 Engine entry (lemon-mlx-engine)

```
TokenIterator::operator++ / next
  └─ step(previous)                         # src/common/generate.cpp
       ├─ StreamGuard(generation_stream())
       ├─ gpu_set_graph_decode_mode(Lstep==1)   # ROCm only; L=1 during gen
       ├─ context_.call_fn(...)                 # model forward (lazy MLX arrays)
       ├─ maybe_quantize_kv_cache(...)
       └─ convert_to_token(logits) → sample
  materialization: eval / async_eval on token
```

Key call sites for `gpu_set_graph_decode_mode`:

| Site | Mode | Purpose |
|------|------|---------|
| `TokenIterator::step` | `true` iff `Lstep==1` | Mark single-token decode for backend |
| `TokenIterator::prepare` | default `false` (opt-in `MLX_PREFILL_ONE_GRAPH=1`) | Prefill stays memory-capped / split |
| MTP draft / verify loops | `true` | Treat draft + sequential verify as L=1 |
| `step_pure_graph` | **forces `false`** | Capture must see immediate launches |

**Product pure-graph gate** (`generate.cpp` ~1927–1954):

- Env: `MLX_DECODE_GRAPH_PURE=1` exactly.
- Default **OFF**. Comment: eager ~68 t/s vs pure ~64 t/s on gfx1151 4-bit.
- When on: `step_pure_graph` → `decode_capture_*` + fixed `graph_decode_input` / `graph_decode_pos`.

### 1.2 Fixed buffers (graph_decode helpers)

| API | File | Role |
|-----|------|------|
| `graph_decode_pos()` | `src/common/graph_decode.cpp` | Resident `[1] int32` device pos |
| `set_graph_decode_pos` / `advance_graph_decode_pos` | same | In-place via `gpu_kv_pos_set` / `gpu_kv_pos_increment` (`indexing.hip`) |
| `graph_decode_input()` | same | Resident `[1,1] int32` token buffer |
| `set_graph_decode_input_from` | same | `gpu_scalar_copy_i32` device copy |
| `graph_decode_enabled()` | same | `MLX_DECODE_GRAPH` env (bring-up; not product) |
| Device-pos RoPE/KV | e.g. `src/llm/models/qwen35_moe.cpp` | When `graph_external_pos()` or `MLX_DECODE_DEVICE_POS` + L==1 |

These exist so a **captured** whole-forward graph can bake stable addresses. They are **not** on the eager product path unless pure-graph / external-pos is active.

### 1.3 MLX GPU eval → CommandEncoder (ROCm)

```
mx::eval / async_eval
  └─ gpu::eval(array)                       # mlx/backend/rocm/eval.cpp
       ├─ get_command_encoder(stream)
       ├─ prim.eval_gpu(inputs, outputs)    # kernel(s) recorded or launched
       ├─ add_temporary(inputs/siblings)
       └─ maybe_commit() / commit()         # flush policy
  finalize(stream) → encoder.commit()
```

**Default (`use_hip_graphs() == false`):**

1. `CommandEncoder::add_kernel_node*` → `add_kernel_node_raw` → **`hipLaunchKernel` immediately** on the encoder’s HIP stream.  
2. JIT/custom kernels → `launch_module_kernel` → **`hipModuleLaunchKernel`** (eager branch).  
3. Library / residual ops → `launch_kernel(lambda)` → direct HIP / hipBLASLt on stream.  
4. `maybe_commit()` when `node_count_ >= max_ops_per_buffer` (default **2000**) → `Worker::commit` → **`hipLaunchHostFunc`** completion signaling + deferred free handlers.

**There is no command-buffer replay on the product path.** Each MLX primitive pays a full host launch (plus host-func signaling on commit boundaries).

### 1.4 HIP graph path (exists, default OFF)

```
build/_deps/mlx-src/mlx/backend/rocm/device.cpp
  use_hip_graphs()
    MLX_USE_HIP_GRAPHS=1     → both prefill + decode-mode
    MLX_HIP_GRAPH_PREFILL=1  → when graph_decode_mode == false
    MLX_HIP_GRAPH_DECODE=1   → when graph_decode_mode == true
    default → false
```

When ON:

| Mechanism | Behavior |
|-----------|----------|
| Manual-node graphs | `hipGraphAddKernelNode` during eval; `commit()` → instantiate / `hipGraphExecUpdate` → `hipGraphLaunch` |
| Decode-mode (`graph_decode_mode`) | **No mid-forward split** (`needs_commit` always false); whole L=1 forward one graph + ExecUpdate |
| Prefill caps | Split on ops/MB; Concatenate is graph-split unless decode-mode |
| `launch_kernel` residual | **Graph-split**: commit accumulated graph, run op inline (~1.4k “inline” launches historically on decode) |
| Pure capture | `decode_capture_begin/end/replay` — stream capture with graphs **OFF**, bake everything including library ops |

Historical outcome (docs under `docs/experiments/prefill-hip-graph/`): rebuild-every-eval was a **net loss**; RDNA3.5 prefill SEGV history; F1–F3 prefill graphs **&lt;10%** pp/s bar (often flat/regress). Upstream hard-off `9c5f1d0d`; local tree restores **env opt-in only** for measurement.

---

## 2. Hot paths by operator family

### 2.1 Quantized matmul (dominant)

| Layer | Engine → MLX | Backend |
|-------|--------------|---------|
| Dense linear | `linear_forward` / `linear_fwd` → `mx::quantized_matmul` (`include/mlx-lm/common/quantized_linear.h`) | `mlx/backend/rocm/quantized/qmm.hip` etc. |
| Registry | `QuantizedWeightRegistry` — packed uint32 at load; no dequant-at-load | `eval_gpu` → `add_kernel_node` (QMV/QMM/WMMA) |
| MoE experts | `SwitchLinear` → `mx::gather_qmm` | same quant backend + gather indexing |
| Fuse opt-in | `MLX_ENABLE_QUANT_FUSE=1` (attn QKV, MLP gate\|up); GDN in_proj needs `MLX_ENABLE_QUANT_FUSE_GDN=1` | fewer, fatter launches |

Decode T=1 often hits **QMV-style** small-M kernels: short device work, **high launch:compute ratio** on 8 CUs.

### 2.2 MoE expert launches

```
SwitchGLU::operator()                     # src/common/switch_layers.cpp
  expand_dims ×2
  optional gather_sort (MLX_MOE_SORT_MIN, default 64 — OFF for T=1 top_k typically)
  gate_up fused? ensure_gate_up_fused()   # default ON unless MLX_NO_EXPERT_FUSION
      → 1× gather_qmm (gate+up) + split
    else 2× gather_qmm (up, gate)
  swiglu elementwise
  1× gather_qmm down_proj
  optional scatter_unsort
```

Per MoE layer, per token (sorted indices = top_k experts):

- **~2–3 `gather_qmm` launches** after gate/up fusion (else ~3–4), plus routing/argsort/take/add noise.
- 35B-A3B class: many MoE layers × top_k (e.g. 8) still means **many small expert matmuls**, not one fat GEMM — classic **tiny-dispatch** surface.
- Prefill large-T: sort path + expert-batched kernels; decode does **not** amortize the same way.

`moe_swiglu.cpp` fused sorted-MoE SwiGLU + hipBLASLt path is a **separate** primitive (ZERO_SYNC pack); engine MoE models primarily use **SwitchGLU + gather_qmm**, not that fused primitive, unless wired elsewhere.

### 2.3 GDN (gated delta)

| Piece | Path | Launch behavior |
|-------|------|-----------------|
| Fused2 decode | `gdn_fused_decode` custom HIP (`src/common/gated_delta.cpp`) | **1** JIT module launch (auto-on; opt-out `MLX_GDN_FUSED2=0` / `MLX_GDN_NO_FUSED2`) |
| Conv step | `gdn_conv_step` custom kernel | 1 launch (vs many mxops) |
| Fallback | `gated_delta_update` / ops loop | Many small launches (rms_norm, softplus, matmuls, …) |
| Inplace state | state-out aliases state-in | Pure-graph friendly; eager benefits from less copy traffic |

Field note: fused2 is **modest** TPS (~0–2 t/s on 35B iGPU), not a 2× lever — launch reduction helps, but **QMM/MoE** still dominate.

### 2.4 Attention / residual / norms

- RoPE, RMSNorm, SDPA/flash attention, residual add, reshape/transpose/slice: each is typically **one or more** `add_kernel_node` launches.
- Compiled MLX fuse groups can collapse some elementwise chains; residual graph still large.
- KV: inplace update default ON (`MLX_KV_INPLACE_OFF` to disable) — reduces copies, not kernel count of the matmul spine.

### 2.5 Host-side completion tax

`Worker::commit` posts `hipLaunchHostFunc` after each encoder commit batch (skipped only during pure `g_decode_capturing`). High op count + frequent commits → **host callbacks** interleaved with launch traffic. Relevant to any command-buffer design that still free/temp via completion handlers.

---

## 3. Launch-bound bottlenecks (Redline-relevant)

Ranked for **tiny-dispatch / command-buffer replay** (not “make GEMM 2% faster”):

1. **Per-primitive host launch**  
   Product path: every quant kernel, elementwise, gather, copy = `hipLaunchKernel` from the eval thread. On 8 CUs, many kernels are shorter than launch overhead.

2. **MoE expert fan-out**  
   top_k × (gate_up|gate+up) × down per MoE layer. Fusion already cut gate+up; still **O(layers × top_k)** small QMMs.

3. **No durable command buffer**  
   Even historical HIP graphs either rebuilt topology every eval (CPU tax) or required fixed addresses (pure-graph arena) and still regressed on APU. There is **no** lightweight “record once, rebind args, submit packet” product path.

4. **Graph-split residuals (if graphs forced on)**  
   hipBLASLt / some JIT / memset go through `launch_kernel` → flush graph + inline launch → fragments the batch; comments cite ~**1.4k** inline ops on full decode under manual-node mode.

5. **Commit / host-func cadence**  
   Eager: flush every ~2000 ops + finalize. Each commit = host callback machinery. Not the #1 cost, but shows up in dispatch-bound profiles.

6. **Contiguous / layout fixups**  
   `ensure_row_contiguous` / `contiguous_copy_gpu` inject extra launches when intermediates aren’t row-major.

7. **MTP sequential verify**  
   Does not reduce launches per *accepted* token under sequential T=1 verify (each verify ≈ full trunk). Launch amortization needs batch verify or dGPU headroom — orthogonal experiment (`docs/analysis/mtp-review/06-tps-ceiling.md`).

8. **APU physics**  
   Prefill is often **compute-bound** (WMMA QMM) → graphs +3%. Decode has higher **launch:compute** share, but pure-graph still lost to eager on this APU (capture/replay tax + fixed arena). Redline must beat **eager hipLaunch**, not just beat broken HIP graphs.

---

## 4. Integration hooks (where Redline would land)

### 4.1 Primary (MLX ROCm CommandEncoder)

| Hook | File | Why |
|------|------|-----|
| `CommandEncoder::add_kernel_node_raw` | `device.cpp` / `device.h` | Single choke point for manual kernels (eager `hipLaunchKernel` vs graph node) |
| `CommandEncoder::launch_kernel` | `device.h` template | Library GEMM / memcpy / residuals; must not graph-split if replaying a whole buffer |
| `launch_module_kernel` | `jit_module.h` | Custom/JIT (GDN fused, etc.) |
| `CommandEncoder::commit` / `maybe_commit` | `device.cpp` | Submit boundary; worker host funcs |
| `gpu::eval` / `finalize` | `eval.cpp` | Op isolation, graph_decode_mode, split policy |

A Redline-style layer would typically:

1. **Record** a sequence of launch descriptors (func, grid/block, smem, arg blob or indirection) during a warmup token.  
2. **Rebind** per-token pointers (or use fixed arenas like `graph_decode_*`).  
3. **Replay** via a small host submit (command buffer / packet stream / single dispatcher kernel), avoiding N× `hipLaunchKernel` round-trips.

### 4.2 Engine (lemon-mlx-engine)

| Hook | File | Why |
|------|------|-----|
| `TokenIterator::step` / pure path | `generate.cpp` | Mode flags; when to record vs replay |
| `graph_decode.cpp` | fixed input/pos | Already stable addresses for capture-style replay |
| `gpu_set_graph_decode_mode` | mlx `eval.cpp` wrapper | Existing “L=1 epoch” signal |
| MoE `SwitchGLU` / quant fuse flags | `switch_layers.cpp`, env | Reduce descriptor count before replaying |
| GDN fused2 | `gated_delta.cpp` | Already collapses multi-op GDN to 1 kernel |

### 4.3 What *not* to hang Redline on

- Reusing **manual-node HIP graphs + ExecUpdate** as the product mechanism (measured loss / complexity / residual splits).  
- Prefill-only graph experiments as a decode launch fix (different shape stability + compute-bound).  
- Expecting MTP sequential verify to hide launch tax on 8 CUs.

---

## 5. Non-goals (decode HIP graphs already killed)

Do **not** spend this experiment re-litigating:

| Item | Evidence | Status |
|------|----------|--------|
| Default `use_hip_graphs()` ON | Net loss; historical RDNA3.5 SEGV; hard-off then env opt-in only | **Product: OFF** |
| Pure decode stream capture as default | `MLX_DECODE_GRAPH_PURE`; APU ~68 eager vs ~64 pure | **Product: OFF** |
| Prefill HIP graphs for gen TPS | F1–F3: +0–4% pp/s, missed ≥10% bar; more mem | **Killed for product** |
| Manual-node graphs without arena | Pointers/temps reallocate; ExecUpdate needs topology + stable params | Dead without arena |
| “Just set `gpu_set_graph_decode_mode(true)`” | Already set on L=1; **inert** without graph env | Not a launch fix |

Pure-graph / HIP-graph code paths remain in-tree for archaeology and dGPU profiling only.

---

## 6. Env cheat-sheet (launch-related)

| Env | Default | Effect |
|-----|---------|--------|
| *(unset graphs)* | eager | Product decode |
| `MLX_USE_HIP_GRAPHS` / `MLX_HIP_GRAPH_{PREFILL,DECODE}` | off | Manual-node graphs |
| `MLX_GRAPH_PREFILL_REPLAY` | off | Prefill ExecUpdate + capture rejected kernels as child graphs |
| `MLX_GRAPH_DECODE=0` | enabled when mode flag set | Disable decode-mode semantics even if flag true |
| `MLX_DECODE_GRAPH_PURE=1` | off | Stream capture build-once replay |
| `MLX_DECODE_GRAPH` | off | `graph_decode_enabled()` bring-up |
| `MLX_DECODE_DEVICE_POS` | off | Device-pos RoPE/KV without full pure machine |
| `MLX_ENABLE_QUANT_FUSE` (+ `_GDN`) | off | Fewer/fatter QMM launches |
| `MLX_NO_EXPERT_FUSION` | off | Disable MoE gate\|up fuse |
| `MLX_GDN_FUSED2=0` / `MLX_GDN_NO_FUSED2` | fused on | Multi-launch GDN fallback |
| `MLX_SYNC_DECODE` | off | Full barrier each step (debug) |
| `MLX_MAX_OPS_PER_BUFFER` | 2000 | Eager commit cadence |

---

## 7. File index (absolute)

| Path | Role |
|------|------|
| `/home/antmi/lemon-mlx-engine/src/common/generate.cpp` | Decode loop, mode flags, pure-graph, MTP L=1 |
| `/home/antmi/lemon-mlx-engine/src/common/graph_decode.cpp` | Fixed pos/input buffers |
| `/home/antmi/lemon-mlx-engine/include/mlx-lm/common/graph_decode.h` | API |
| `/home/antmi/lemon-mlx-engine/src/common/switch_layers.cpp` | MoE SwitchGLU / gather_qmm |
| `/home/antmi/lemon-mlx-engine/src/common/gated_delta.cpp` | GDN fused2 / conv |
| `/home/antmi/lemon-mlx-engine/include/mlx-lm/common/quantized_linear.h` | Registry + `quantized_matmul` |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/device.cpp` | `use_hip_graphs`, commit, capture |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/device.h` | CommandEncoder, `launch_kernel` |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/eval.cpp` | `gpu::eval`, `gpu_set_graph_decode_mode` |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/worker.cpp` | Host completion |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip` | Quant kernel launches |
| `/home/antmi/lemon-mlx-engine/build/_deps/mlx-src/mlx/backend/rocm/moe_swiglu.cpp` | Fused MoE primitive (alt path) |
| `/home/antmi/lemon-mlx-engine/docs/experiments/prefill-hip-graph/` | Graph A/B results (killed) |
| `/home/antmi/lemon-mlx-engine/docs/ROCM_TPS_OPTIMIZATION_OPERATORS_KV.md` | Operator TPS context |
| `/home/antmi/lemon-mlx-engine/docs/analysis/mtp-review/06-tps-ceiling.md` | Launch-bound MTP ceiling |

---

## 8. Suggested Redline experiment framing

**Hypothesis:** A large fraction of T=1 wall on ROCm is **host dispatch + many short kernels**, not only arithmetic intensity of QMM. A tiny command buffer (record launch descriptors once; rebind args; multi-launch or single replay submit) can reduce per-token host cost vs eager `hipLaunchKernel` without the HIP-graph build/ExecUpdate tax.

**Success criteria (sketch):**  
- Gen t/s lift vs **eager product baseline** (not vs broken graph mode).  
- Correctness vs golden / temp=0 bitwise or policy-defined numeric.  
- Peak mem ≤ baseline + small fixed arena.  
- Kill if only helps pure synthetic microbench but not 35B end-to-end.

**Kill criteria:**  
- Reimplements HIP graphs with same rebuild/split pathologies.  
- Requires pure-graph arena + still loses to eager on APU with no dGPU win.  
- Only works with graphs env on and fails default product config.

---

*Generated for `exp/redline-kernel-launch` — engine launch map only; no implementation in this doc.*
