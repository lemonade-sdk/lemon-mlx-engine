# HIP-Graph Reuse — Implementation Plan

Self-contained plan to finish graph reuse so decode beats eager on the R9700/TB5.
Builds on `HIP_GRAPH_REUSE_REPORT.md`. Backend: NripeshN/mlx `rocm-support`,
files under `build/_deps/mlx-src/mlx/backend/rocm/`.

## Goal & success criteria

Steady-state decode must, per token:
- **NOT** rebuild the graph (no per-token `hipGraphAddKernelNode` storm), and
- **NOT** reinstantiate the exec (no per-token `hipGraphInstantiate`).
Only: refresh the changed params (or nothing), then `hipGraphLaunch` per chunk.

Done when, on a 50-token run (`MLX_GRAPH_REUSE_STATS=1`):
- `reinst≈0`, `addnode_time≈0` after the first token (build-once), and
- output is **bit-identical to eager** at temp 0, coherent over 1000 tokens, on
  **both** gfx1151 (dev 0) and gfx1201 (dev 1), and
- R9700 decode TPS **> eager** (eager ≈ 45), peak VRAM bounded (< 34 GB).

Two independent pieces: **(A) exec-refresh** (skip reinstantiate) and **(B)
build-once** (skip rebuild). Both required. Land A first (smaller, unblocks the
TB5 exec-upload win), then B (removes the 554ms reconstruction).

---

## Phase A — Exec refresh without reinstantiate

Pick the refresh mechanism. Primary = **deterministic SetParams** (no arena, the
user's preference). Fallback = **stable-address relaunch**. Try A1; if the OOB
proves unfixable in a bounded effort, switch to A2.

### A1. Crack the SetParams OOB (primary)

Status: `hipGraphExecKernelNodeSetParams` per-node refresh is correct in every
standalone (`/tmp/setparams_loop.hip`, `sp_byval_change.hip`, `modnode_setparams.hip`)
but OOBs in the engine (`copy_gg_byval`, grid 2097152) under `MLX_GRAPH_SETPARAMS=1`.
The grid/block-**product** key collision was found & fixed (full dims) but it still
OOB'd, so there is a second cause.

Steps:
1. **Poisoned-free bisection in the engine.** Re-enable the per-gen poison mode
   (`MLX_GRAPH_POISON_FREE`, already in `allocator.cpp`) but bounded: fill freed
   buffers with a sentinel and KEEP them mapped only for the slot being SetParams'd
   (not a global leak — that OOM'd). Run `MLX_GRAPH_SETPARAMS=1` + poison: if the
   OOB turns into garbage (mapped sentinel) the failing node reads a buffer we
   refreshed wrong; if it stays an unmapped fault, the param points at a freed/
   reused address.
2. **Dump the failing node's params** right before launch when SetParams is on:
   for each `slot.src_nodes[i]`, log `build_node_params_[i]` (func, grid, block,
   smem) AND the dereferenced kernarg pointer values (the device addresses in
   `Pack->ptrs`). Compare against what `hipGraphExecKernelNodeGetParams(exec,
   src_nodes[i])` reports after SetParams. A mismatch names the bug:
   - addresses differ → SetParams didn't take (kernarg-layout issue),
   - addresses match but grid differs → grid not applied,
   - addresses match but point at freed memory → Pack lifetime.
3. **Suspect kernarg layout for by-value-struct kernels.** `copy_gg_byval` takes
   `hip_array<int,10>`/`hip_array<long,10>` by value. Verify the engine's `Pack`
   (`device.h` `add_kernel_node_ex`) lays these out identically to what the node
   was instantiated with. If SetParams marshals by the node's *original* arg sizes
   and our Pack passes a differently-sized blob, it corrupts. Fix = ensure
   `build_node_params_[i].kernelParams` exactly mirrors the instantiate-time layout.
4. **Verify `src_nodes[i]` ↔ `build_node_params_[i]` alignment across commits.**
   `src_nodes` is captured at first instantiate; `build_node_params_` is this
   commit's. They must be the same kernel at each index. Assert
   `node_sig(build_node_params_[i]) == slot.src_sig[i]` (already have `src_sig`)
   for ALL i before applying SetParams; if any differ, fall back to reinstantiate
   for that commit and log it — that catches residual key ambiguity.
5. Once SetParams is bit-identical at temp 0 on dev 0, repeat on dev 1.

Exit: `MLX_GRAPH_SETPARAMS=1` gives bit-identical output, both devices,
`update=4043 reinst=0`, no OOB, coherent 1000 tokens.

### A2. Stable-address relaunch (fallback, if A1 stalls)

If addresses are stable token-to-token, skip SetParams entirely — just relaunch
the cached exec.

Steps:
1. **Deterministic per-forward allocator for auto-batch intermediates.** Add a
   bump arena reset at the start of each decode forward so op N's k-th allocation
   always returns the same address. This is the DecodeArena idea but wired into the
   auto-batch malloc path (NOT the engine capture path the user dislikes). Gate on
   a flag; only for decode (single-token) forwards.
2. **Detect address stability** in `commit()`: compare this build's
   `build_node_params_` kernarg values against the slot's stored values. If all
   match → `graph_exec = slot.exec; hipGraphLaunch` (no SetParams). Add a
   `relaunch=` counter.
3. **KV position** is the one thing that legitimately changes each token. Either
   keep the KV write at a stable base with a device-side position the kernel reads
   (device-position decode, already prototyped via `set_graph_decode_pos`), or
   SetParams *only* the handful of KV-touching nodes. Everything else relaunches.

Exit: `relaunch≈all`, `reinst≈0`, bit-identical, both devices.

---

## Phase B — Build-once (skip the per-token rebuild)

Removes the 554ms/token `hipGraphAddKernelNode` reconstruction. Independent of
whether A1 or A2 landed.

The obstacle: the topology key is known only after nodes are added. Defer node
creation.

Steps:
1. **Record, don't build, during eval.** In `add_kernel_node_raw` /
   `add_module_kernel_node`, when graphs on, push `{func, grid, block, smem,
   params-ptr, keepalive}` into `build_node_params_` and the dep info, but DO NOT
   call `hipGraphAddKernelNode` yet. Move the real-dep + chain-edge bookkeeping to
   operate on op indices (record `(from_idx, to_idx)` pairs) instead of
   `hipGraphNode_t`.
2. **At `commit()`:** compute `key = graph_nodes_key_ + ":" + graph_deps_key_`
   from the recorded sequence (it already is). Look up `exec_pool_[key]`:
   - **hit (drained slot):** apply Phase-A refresh (SetParams or relaunch) using
     the recorded params + `slot.src_nodes`; `hipGraphLaunch`. **No graph built.**
   - **miss:** NOW build `build_graph_` (loop the recorded ops →
     `hipGraphAddKernelNode`), wire deps from the recorded index pairs,
     `hipGraphInstantiate`, store `slot.src_nodes` + `source_graph` + packs.
3. **Keep the dep/edge recording faithful** to today's chain-edge fix (serialize
   each node behind the previous, deduped vs real deps) so a miss-built graph is
   still bit-correct.
4. Clear the recorded vectors each commit (as today).

Exit: after the first decode token, `addnode_time≈0` and `new≈0` for the steady
topologies; `graph-time-ms` dominated by `launch` only.

---

## Phase C — Device-length attention (only if static-shape route)

Needed only if reuse ends up requiring a fixed-shape (static-KV) graph (so the
topology is constant every token). Today's auto-batch uses growing-KV shapes, so
attention commits change shape and won't reuse — they'd reinstantiate. Options:
- keep growing-KV and accept attention commits reinstantiate (they're a minority),
  OR
- go static-KV full-cap and add device-length flash attention so the bandwidth
  scales with the live length, not capacity.

Step (if pursued): extend the decode SDPA vector kernel
(`scaled_dot_product_attention.hip kernel_sdpav_1pass`) to take a device-side
`const int* kv_len` and bound the KV loop by `*kv_len` instead of `params.kL`.
Update the dispatch + the model to pass the device length. Validate the captured
graph is constant-shape and attention bandwidth tracks length (TPS independent of
ctx_size).

---

## Phase D — Validation & measurement

1. Bit-identical to eager (temp 0, fixed prompt) on dev 0 AND dev 1, for: A-only,
   then A+B.
2. Coherent 1000-token generation, both devices, under poisoned-free (no UAF/leak,
   bounded VRAM).
3. Walk op cap 50→2000: stable, bit-identical.
4. **R9700 honest measurement:** record per token — graph launches, inline
   launches, `hipGraphInstantiate` count (must be ~0 steady state), `addnode`
   time, decode TPS, peak VRAM. Compare vs eager (45 tok/s). Confirm exec-upload
   over TB5 is gone.
5. Cross-check against the **mlx-lm reference** (now building for ROCm) on the same
   standard model, both devices, to validate engine-independent behavior.

---

## Env flags (existing)

`MLX_USE_HIP_GRAPHS=0` (off), `MLX_GRAPH_SETPARAMS`, `MLX_GRAPH_RELAUNCH`,
`MLX_GRAPH_EXECUPDATE`, `MLX_GRAPH_REINST_SLOT`, `MLX_GRAPH_REUSE_STATS`,
`MLX_GRAPH_POISON_FREE`, `MLX_GRAPH_FREE_LAG`, `MLX_GRAPH_DEFER_MAX_MB`,
`MLX_ROCM_NO_ASYNC_POOL`, `MLX_GRAPH_NODEFER`, `MLX_EVENT_BLOCKING`,
`MLX_MAX_OPS_PER_BUFFER`, `MLX_NO_CONCAT_SPLIT`.

## Files

- `device.{h,cpp}` — commit(), exec_pool_/ExecSlot, add_kernel_node_raw,
  add_module_kernel_node, build_node_params_/build_nodes_/src_sig, refresh logic.
- `allocator.cpp` — deferred-free, free_graph_generation_async, pool, (A2) arena.
- `scaled_dot_product_attention.hip` — (C) device-length SDPA.
- `src/common/generate.cpp`, `src/llm/models/qwen35_moe.cpp` — (A2/C) device
  position / KV length plumbing.

## Build & run (7.13)

```
cd build && HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/core-7.13/lib/llvm/bin ninja chat
LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib ./build/chat ~/mtp_convert/qwen36_4bit_v6 \
  --device {0|1} --no-think --ignore-eos --max-tokens 50 --temperature 0
# Python mlx (built): PYTHONPATH=build/_deps/mlx-src/build/lib.linux-x86_64-cpython-312
#   LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib:<that>/mlx/lib  HIP_VISIBLE_DEVICES={0|1}
#   python3 -m mlx_lm generate --model mlx-community/Qwen3.6-35B-A3B-4bit ...
```

## Sequencing (recommended)

1. Phase A1 (crack SetParams OOB) — unblocks instantiate-free reuse, biggest TB5
   win per effort. Timebox; if stuck, A2.
2. Phase B (build-once) — removes the rebuild; the other half of the win.
3. Phase D (measure on R9700). Decide if Phase C (device-length attention) is
   warranted based on whether attention commits dominate the residual cost.
