# RESUME — ROCm HIP-graph decode (auto graph batching)

## Goal
Make HIP-graph decode produce coherent 1k-in/1k-out output AND beat eager
(~41 tok/s) on gfx1151 (APU, device 0), q4 model `~/mtp_convert/qwen36_4bit_v6`.

## TL;DR status
- **Production path that WORKS: eager decode, 41 tok/s, coherent.** Default build
  (graphs-OFF). Verified on both 7.12 and 7.13 runtimes. Ship this.
- **graphs-ON (`MLX_USE_HIP_GRAPHS=1`) is WIP and produces garbage.** Runs
  end-to-end on the 7.13 runtime but output is wrong. Two compounding bugs
  isolated (below). Gated OFF by default — does not affect production.

## Build & run (ALWAYS use 7.13 — see memory `rocm_must_use_7_13_for_graphs`)
```
cd build && HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/core-7.13/lib/llvm/bin ninja chat
# run (7.13 runtime is REQUIRED for graphs; 7.12 segfaults hipGraphLaunch):
LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib ./build/chat ~/mtp_convert/qwen36_4bit_v6 \
  --device 0 --no-think --ignore-eos --max-tokens 50 --temperature 0
# eager = default; graphs = add  MLX_USE_HIP_GRAPHS=1
```
Two ROCm installs: `/opt/rocm` = 7.12 (HIP 7.3, rocBLAS 5.3); `/opt/rocm/core-7.13`
= 7.13 (HIP 7.13, rocBLAS 5.4, hipBLASLt 1.3) — **full stack**. The build links 7.12
by default (CMakeCache AMDHIP64_LIB etc. -> /opt/rocm); LD_LIBRARY_PATH overrides at
runtime (sonames compatible). TODO: reconfigure CMake fully onto core-7.13.

## What's DONE and committed (NripeshN/mlx branch `rocm-support`, latest 90f557b9)
1. **Full kernel migration** to CUDA-style `add_kernel_node` (~25 files: unary,
   binary, binary_two, ternary, copy + copy/ subdir, arange, rms_norm, layer_norm,
   softmax, logsumexp, scan, arg_reduce, sort, rope, indexing, random, attention
   sdpa/flash/flash_wmma, reduce/ subdir, quantized affine/fp/convert, qmm ~63
   sites, gemv). Builds clean; **graphs-OFF (eager) unchanged at 41 tok/s.**
2. **Backend graph infra** in `backend/rocm/device.{h,cpp}` (mirrors backend/cuda):
   `add_kernel_node`/`_ex`/`_raw`, `build_graph_`, `insert_graph_dependencies`,
   `commit()`, `synchronize()`. Gated behind `use_hip_graphs()` = `MLX_USE_HIP_GRAPHS`.
3. **Graph-split** in `launch_kernel`: un-graphable residuals (JIT module kernels via
   `hipModuleLaunchKernel`, library GEMM, memsets) flush+launch the kernel-node graph
   then run immediately on the same stream (HIP `hipKernelNodeParams` has NO
   module-function field in 7.13 — confirmed — so JIT/GEMM can't be kernel nodes).
4. **rocBLAS forced in graph mode** (`is_hipblaslt_available` returns false when
   `MLX_USE_HIP_GRAPHS`): hipBLASLt's lazy `hipblasLtCreate`/AlgoGetHeuristic/workspace
   hipMalloc are non-capturable and `exit()` the process under graph activity.
5. **Linear-chain dependencies**: graph nodes chained in submission order (matches
   eager stream order) instead of the incomplete set_input/output_array edges.
6. **Func-keyed node IDs**: cache/topology key includes the kernel func ptr + dims
   (type-only "K" key collided distinct kernels -> ExecUpdate mis-reuse -> garbage).
7. **Engine** (geramy/optimizations): full kernel-migration state; the OLD capture-once
   path (`MLX_DECODE_GRAPH`, device-position KV, GDN double-buffer) is still present
   but separate/unused by the auto-graph path — slated for deletion once graphs-ON works.

## RULED OUT (with standalone repros / bisection) — do NOT re-investigate
- HIP graph kernel nodes work on 7.13: `/tmp/graphtest.hip` (hipGraphAddKernelNode ==
  hipLaunchKernel output). Recreate if needed.
- MLX-style tuple/pack param marshaling works: `/tmp/graphtest2.hip` (MATCH YES).
- NOT async/timing: garbage persists under `HIP_LAUNCH_BLOCKING=1`.
- NOT dependency edges: linear chain still garbage.
- NOT KV realloc: `MLX_KV_STATIC=1` still garbage.
- NOT exec-cache reuse alone: fresh-instantiate (no cache) still wrong/segfaults.

## THE TWO REMAINING BUGS (the actual blockers)
**BUG 1 — buffer lifetime (causes segfault):** graph nodes execute later (at commit),
but the allocator frees intermediate buffers at eval time -> pool reuses them before
the graph runs -> graph reads/writes recycled memory -> segfault. Deferring frees
(`set_graph_active(true)` in the encoder ctor + `flush_graph_deferred_frees()` in
synchronize) PREVENTS the segfault but balloons memory to ~30 GB (frees never released
mid-eval). Proper fix: tie each graph-referenced buffer's lifetime to that graph's
completion (like CUDA's per-commit completion-handler temporaries), not a global defer.

**BUG 2 — computation garbage (separate, the hard one):** with BUG 1 worked around
(buffers kept alive, non-aliased), output is STILL garbage. So some kernel(s) compute
wrong results in the full multi-kernel graph despite the single-kernel repro being
correct. NOT reproduced in isolation yet.

## DEFINITIVE NEXT STEP (start here)
Build a **per-array eager-vs-graph output bisection**: run the SAME single forward
once eager and once graphs-ON, dump a checksum (sum-abs) of every array's output in
tape order, diff to find the FIRST diverging op. That pinpoints the exact kernel
whose graph-node output != its eager output, turning BUG 2 from a needle-hunt into a
one-kernel fix. (Hook into `mlx/transforms.cpp` eval loop or `backend/rocm/eval.cpp`
gpu::eval to dump `sum(abs(out))` per array under an env flag, run both modes, diff.)

After BUG 2: fix BUG 1 with proper per-graph buffer lifetime; then benchmark vs eager
(graph-split fragments around JIT/GEMM, so expect modest gains for this GDN+MoE model
unless those residuals are reduced); then delete the old capture-once engine hack
(task #22) and revert the model to plain eager.

## Key files
- backend/rocm/device.{h,cpp} — graph infra, commit, synchronize, launch_kernel
- backend/rocm/gemms/hipblaslt_gemm.cpp — is_hipblaslt_available (rocBLAS-in-graph)
- backend/rocm/eval.cpp — gpu::eval needs_commit/commit wiring
- backend/rocm/allocator.cpp — graph_active() deferred-free path (BUG 1)
- src/llm/models/qwen35_moe.cpp, src/common/generate.cpp — engine (old capture-once)
- repo: ROCM_GRAPH_BACKEND_PLAN.md (original plan), this file (resume)
- memory: rocm_must_use_7_13_for_graphs, rocm_no_graph_batching_rootcause
