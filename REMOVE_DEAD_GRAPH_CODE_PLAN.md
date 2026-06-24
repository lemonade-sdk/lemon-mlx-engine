# Plan: remove dead/non-working HIP-graph code (engine + mlx rocm backend)

Keep the shipped decode-mode path (graphs default-on; single-token forward = one
graph refreshed via hipGraphExecUpdate). Remove the capture-once/DecodeArena/
device-position path and dead reuse/diagnostic knobs. Two repos: engine
(geramy/graph-prefill) + mlx backend (rocm-support, NripeshN/mlx) kept in sync via
CMake FetchContent GIT_TAG rocm-support.

## Env-guard classification
KEEP: MLX_USE_HIP_GRAPHS, MLX_GRAPH_DECODE, MLX_NO_CONCAT_SPLIT,
MLX_ROCM_NO_ASYNC_POOL, MLX_GRAPH_DEFER_MAX_MB, MLX_GDN_NO_FUSED (+ unrelated
MLX_SYNC_DECODE/MLX_GEN_OWN_STREAM/MLX_SDPA_DECODE_FLASH).
REMOVE: MLX_GRAPH_REUSE_STATS, MLX_GRAPH_NO_REUSE, MLX_GRAPH_REINST_SLOT,
MLX_GRAPH_SETPARAMS, MLX_GRAPH_SP_DEBUG, MLX_GRAPH_RELAUNCH, the MLX_GRAPH_EXECUPDATE
env disjunct (keep mechanism via graph_decode_mode), MLX_GRAPH_EU_MIN/MAX,
MLX_HIP_GRAPH_DUMP, MLX_GRAPH_DEBUG, MLX_GRAPH_FORCE_FROM/TRACE, MLX_DECODE_GRAPH,
MLX_DECODE_DEVICE_POS, MLX_GRAPH_WARM, MLX_NO_ARENA, MLX_GRAPH_NO_INPLACE,
MLX_KV_STATIC. (Optional: MLX_GRAPH_NODEFER/POISON_FREE/FREE_LAG debug knobs.)

## Engine removals
- DELETE src/common/graph_decode.cpp + include/mlx-lm/common/graph_decode.h (+ CMakeLists.txt:113).
- generate.cpp: drop capture-once extern block (keep gpu_set_graph_decode_mode), the
  capture-once step() block (~343-516); KEEP normal path + decode-mode wiring + prepare() reset.
- qwen35_moe.cpp: remove gmode/device-pos attention branches (keep scalar-offset RoPE +
  cache->update+sdpa), gdn_dbuf double-buffer (keep simple (*cache)[i]= writes), inplace_state/
  MLX_GRAPH_NO_INPLACE, want_static graph_decode/MLX_KV_STATIC tie-in (keep eager cap/reserve).
- kv_cache.cpp/.h: remove update_at_pos + gpu_kv_row_write extern (no callers after above).
- DELETE stale docs: GRAPH_DECODE_PLAN, GRAPH_REUSE_PLAN, HIP_GRAPH_REUSE_REPORT,
  RESUME_GRAPH_DECODE, ROCM_GRAPH_BACKEND_PLAN, HIP_KV_FINDINGS.

## MLX backend removals
- device.h/.cpp: remove reuse/stats globals+prints, arg-hash (build_arghash_/pending_arghash_/
  KernelArgs::arg_hash + add_module_kernel_node arghash param), ExecSlot.src_sig/last_args/node_sig,
  SetParams branch + sig guard, relaunch branch, eu_min/max gate, no_reuse, MLX_HIP_GRAPH_DUMP.
  KEEP: needs_commit decode-mode, ExecUpdate branch, reinstantiate fallback, add_kernel_node_kp +
  add_child_graph_node FLATTEN (delete only pending_arghash_ lines), exec_pool_/ExecSlot core,
  set_graph_active(true) ctor (keep-path deferred-free), graph_decode_mode/set.
- Remove capture API: begin_capture/end_capture/replay/reset_graph/capturing_/graph_/graph_exec_/
  capture_held_ + if(capturing_) block in commit(); untangle event.hip:383 capturing() branch.
- eval.cpp: remove gpu_arena_*/gpu_graph_* shims + FORCE_FROM/TRACE; keep is_graph_split_op,
  finalize, synchronize, gpu_set_graph_decode_mode.
- allocator.{h,cpp}: remove DecodeArena class + arena_.active() branches; KEEP deferred-free.
- indexing.hip: remove gpu_kv_pos_* / gpu_kv_row_write kernels.

## Order + verification
Phase 0 docs → 1 leaf diagnostics (backend) → 2 capture-once engine → 3 capture-once backend → 4 grep sweep.
After each: `source /tmp/rocm713.env && HIP_PLATFORM=amd ninja -C build chat`, then temp-0 decode
bit-identical to eager on dev 0 (gfx1151) and dev 1 (gfx1201). Backend commits land on rocm-support
first (engine re-fetches it), then engine.
