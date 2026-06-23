# ROCm backend graph-batching: mirror CUDA, delete the engine hack

## Root cause (confirmed by CUDA/Metal/Swift/ROCm agent review)

The ROCm backend issues **every kernel as an individual `hipLaunchKernel`, inline, immediately**
(`backend/rocm/device.h:187` `launch_kernel`). It has **no per-eval graph batching**.

CUDA, by default, accumulates each eval's kernels into a `cudaGraph_t` and submits the whole
chunk with **one `cudaGraphLaunch` per 20–100 ops**, reusing the instantiated exec graph across
decode steps via **`cudaGraphExecUpdate`** keyed on graph topology
(`backend/cuda/device.cpp:243-288` add_kernel_node, `:458-536` commit). Metal does the same with
one `MTLCommandBuffer` per ~40 kernels. Our ROCm path is a faithful copy of CUDA's
**graphs-DISABLED fallback** — i.e. CUDA's slow path is our only path.

For decode (~2000 kernels/token, launch-bound) this is THE reason ROCm decode is slow, and why
the engine needed a fragile capture-once hack to claw back the graph benefit manually.

## Decision

Mirror CUDA's backend graph pipeline exactly. Delete the engine capture-once hack and all
workarounds it required. No env gates kept "just in case."

---

## Part A — ADD: CUDA's graph pipeline ported to ROCm

Port these CUDA pieces (`backend/cuda/`) into `backend/rocm/` 1:1 (HIP API equivalents in parens):

1. **CommandEncoder graph state** (`cuda/device.h`): `graph_` (hipGraph_t accumulator),
   `LRUCache<string, hipGraphExec_t> graph_cache_` (cap 400), `node_map_`, `active_deps_`,
   `active_outputs_`, `graph_nodes_key_`, `graph_deps_key_`, `node_count_`, `bytes_in_graph_`,
   `max_ops_per_graph_`, `max_mb_per_graph_`.

2. **Kernel-node construction** (`cuda/device.cpp:243-288`, `device.h:52-79`):
   `add_kernel_node()` marshals params → `hipGraphAddKernelNode` (was cudaGraphAddKernelNode).
   The existing `launch_kernel(lambda)` call sites migrate to record into the graph. Mechanism:
   - Primary: rewrite kernel launches to `add_kernel_node(func, grid, block, smem, args...)`.
   - For JIT/custom kernels (`hip_kernel`) and opaque library calls (hipBLASLt/rocBLAS), use a
     per-call **`hipStreamBeginCapture`(ThreadLocal)/EndCapture → `hipGraphAddChildGraphNode`**,
     mirroring CUDA's `CaptureContext` (`cuda/device.cpp:424-438`, `cublas_utils.cpp:198`).

3. **Dependency tracking** (`cuda/device.cpp:142-176`, `224-241`): `set_input_array` /
   `set_output_array` build `node_map_` (buffer→producer node) and the topology key strings;
   `insert_graph_dependencies` batches edges → `hipGraphAddDependencies` in commit.

4. **Commit / exec-update** (`cuda/device.cpp:458-536`): `needs_commit()` = `node_count_ >
   max_ops_per_graph_ || bytes`. `commit()`: add deps, `make_current` once, topology-key the
   `graph_cache_`, **`hipGraphExecUpdate`** the cached exec graph (fallback
   `hipGraphInstantiate`), **one `hipGraphLaunch`**, reset graph_, `worker_->commit`.

5. **Per-arch graph limits** (`cuda/device.cpp:181-203` get_graph_limits): pick gfx1151/gfx1201
   op/MB caps (start ~50 ops / 100 MB), env override `MLX_MAX_OPS_PER_BUFFER`.

6. **Worker on a side stream** (`cuda/worker.cpp:44-57`): event recorded on work stream → waited
   on a dedicated `signal_stream_` → `hipLaunchHostFunc` on the **signal stream** (not the work
   stream). Removes the per-commit work-stream stall we have now.

7. **Blocking event wait + event pool** (`cuda/event.cu:26-94`): replace the busy spin-poll
   (`rocm/event.hip:69-83`) with `hipEventSynchronize` on a `hipEventBlockingSync` event; pool
   events per (device,flags). Stops APU memory-controller contention.

8. **Cached device select** is already present (`make_current` thread-local) — call it once per
   commit (not per kernel) like CUDA (`cuda/device.cpp:483`); drop the double `make_current` +
   map lookups in `rocm/device.cpp` `get_command_encoder`.

---

## Part B — DELETE: old code made irrelevant by Part A

Once the backend auto-batches every eval, the manual engine hack and its scaffolding are dead.
Remove them outright (no env gating):

Engine (`src/`):
- `src/common/generate.cpp`: the entire `MLX_DECODE_GRAPH` capture-once block (warmup, arena
  begin/reset/pause, begin/end capture, replay loop, `g_input`/`g_logits`, capture_eval of mamba
  states, contiguity force, write_token, advance/replay).
- `src/common/graph_decode.{cpp,h}`: device-position pos buffer, `set/advance_graph_decode_pos`,
  `graph_external_pos`, `graph_capturing`, `graph_decode_enabled`.
- `src/llm/models/qwen35_moe.cpp`: the `graph_external_pos()` attention branch (full-CAP read +
  per-layer mask), the GDN `gdn_dbuf` double-buffer (ridx/widx, slice/slice_update, promotion,
  parity eval). Revert attention + GDN to the plain eager path only.
- `src/common/kv_cache.cpp`: `update_at_pos` / static-CAP graph plumbing if unused by eager.

MLX backend (`backend/rocm/`):
- Old whole-stream capture-once API: `begin_capture`/`end_capture`/`replay`/`reset_graph` and the
  `gpu_graph_*` / `gpu_arena_*` wrappers in `eval.cpp` (replaced by automatic per-commit graphs).
- `DecodeArena` (allocator.h/.cpp) and arena routing in `malloc_async`.
- hipBLASLt capture special-casing (`is_hipblaslt_available` stream_capturing branch,
  `MLX_HIPBLASLT_NO_CAPTURE`) — GEMM is captured as a child node like CUDA does; the warm-cache
  workspace pre-alloc stays (needed so capture doesn't hipMalloc).
- DynamicSliceUpdate `graph_active()` donation hack (`backend/gpu/primitives.cpp`) if the device
  capture flag is gone.

Engine decode loop: rewrite to the reference async-pipeline (Swift mlx-lm `TokenIterator`):
compute token N+1's forward, `async_eval`, then read the **stale** token N (`.item()` overlaps
GPU). Keep sampling on-device. The KV cache returns to normal in-place append (no static CAP).

---

## Phasing

1. **Backend graph pipeline** (Part A 1–5): land auto-batching + exec-update. Validate decode is
   coherent and faster on device 0 (target ~2× eager).
2. **Worker side-stream + blocking events** (A 6–8): APU contention + per-commit stall.
3. **Delete engine hack** (Part B): remove capture-once + device-pos + double-buffer + arena;
   simplify the model and the decode loop to plain eager + async pipeline.
4. **Bench** graph-batched eager vs old: confirm 76+ on APU, and re-check R9700.

## Risks
- hipGraphExecUpdate fragility on topology change (CUDA falls back to instantiate; mirror that).
- hipStreamBeginCapture(ThreadLocal) must wrap library calls cleanly (workspace pre-alloc’d).
- Per-arch op/MB caps need tuning on gfx1151/gfx1201.
