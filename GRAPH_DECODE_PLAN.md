# HIP-Graph Decode — Correct Kernarg/Buffer Lifetime Ownership (MLX ROCm backend)

Self-contained plan. Read only this file to start; it does not depend on any prior notes.

## Goal

Make HIP-graph-batched decode produce output **bit-identical to eager** on gfx1151 (APU,
device 0) at graph sizes large enough to actually batch kernels, then **measure decode
throughput vs eager honestly**. No predetermined outcome — we get it correct first, then let
the measurement decide whether batching wins.

Model: `~/mtp_convert/qwen36_4bit_v6` (q4). Eager is the current production path and is
coherent; record its TPS as the baseline at the start.

## Premise (what is and isn't the problem)

HIP graphs and the ROCm runtime are **working as documented**. The graph kernarg/replay
semantics match CUDA's. The garbage output and the need to keep graphs tiny come from **how
our backend manages the lifetime of the memory that graph kernel nodes reference** — i.e.
our design, not a runtime bug. This plan fixes that.

### The API contract we must honor (the spec)

- A **kernel node owns its kernel params and the argument values they point to.** That memory
  is valid until the node is destroyed or its params are modified. Callers must not modify it
  directly, and must not point a node's `kernelParams[]` entries at caller-owned, stack, or
  per-replay-transient memory.
- An **instantiated graph exec dereferences memory during `hipGraphLaunch`.** Every byte it
  touches (kernarg values, input/output/intermediate buffers, dynamic-value buffers) must
  stay valid and unmodified until that launch completes.
- For values that change between replays (token position, sequence length), the supported
  pattern is a **stable device buffer** the node reads through a pointer, whose **contents**
  are updated before each replay (memcpy node or in-place write) — never by reassigning the
  node's param pointer to fresh memory per replay.

Implication: every allocation a graph node dereferences must be owned with a lifetime that
extends until the last in-flight replay referencing it has completed.

## Root-cause hypotheses (confirm by diagnosis in Phase 1 — do not assume)

- **H1 — Kernarg value memory freed early.** The kernarg storage we marshal (the "Pack") is
  freed/recycled while an instantiated exec still references it at replay → stale reads.
  (Symptom on record: keeping that memory alive removed the garbage.)
- **H2 — Source graph destroyed early.** The source `hipGraph_t` that an exec was instantiated
  from / updated against is destroyed per-commit while the exec still references it.
- **H3 — Intermediate buffers freed early.** Op-output buffers feeding later nodes are freed by
  the allocator at eval time but read by the deferred graph at launch → use-after-free.
- **H4 — Pointer reassigned per replay.** We reassign a node's `kernelParams[]` entry to point
  at transient memory each token (the unsupported pattern) instead of updating a stable
  buffer's contents.
- **H5 — Missing dependency edges.** Consumer nodes lack edges to their producer nodes, so at
  larger graph sizes replay order diverges from eager → races / nondeterministic output. This
  is an *ordering* fault and needs a different fix than the *lifetime* faults above.

## Phase 0 — Clean baseline + master diagnostic

1. **Baseline.** Build, run eager, confirm coherent, record TPS and a reference token stream
   (temp 0, fixed prompt). This is the bit-identical target.
2. **Graphs-ON smoke.** Confirm the graph path builds and runs (it will currently misbehave;
   that's expected).
3. **Per-op eager-vs-graph checksum harness (the master diagnostic).** Under an env flag, dump
   `sum(abs(out))` of every array output in tape order. Run the same single forward eager and
   graphs-ON; diff to find the **first diverging op**. Everything in Phase 1 keys off this.
4. **Poisoned-free mode.** Add an allocator debug mode that fills freed buffers with a sentinel
   (e.g. NaN / 0xDEAD) instead of recycling. If graph output changes between normal-free and
   poisoned-free, a node is reading freed memory → use-after-free (and which op diverges names
   the buffer).

## Phase 1 — Diagnosis (pin the exact memory and the exact point its lifetime ends)

Concrete steps, in order; stop when the fault is precisely located.

1. **Empirically establish add-time copy semantics (standalone HIP test).** AddKernelNode with
   args in a heap buffer → free/overwrite that buffer → launch (no update). Correct output ⇒
   the node copied the values at add-time (our Pack is safe to free after add). Wrong output ⇒
   the node references our buffer until launch (lifetime must extend to launch). Settles
   whether H1 applies to the add path.
2. **Same test for the exec/update path.** Build graph → instantiate → free/overwrite the
   source graph and arg buffers → `hipGraphExecUpdate` → launch. Pinpoints H1/H2 on the reuse
   path specifically.
3. **Allocator reference audit.** Tag each alloc/free with whether a pending (not-yet-launched
   or in-flight) graph node references it. Run graphs-ON at cap 2 and at a larger cap; flag any
   referenced buffer freed before its graph's launch completes (H1/H3).
4. **Poisoned-free run** (from Phase 0.4) under `HIP_LAUNCH_BLOCKING=1`: confirms use-after-free
   and, via the checksum, names the first array read after free.
5. **Checksum bisection at the breaking cap.** Use the per-op checksum + `MLX_GRAPH_FORCE_FROM`
   to bisect to the first wrong op. Inspect what memory that op's node dereferences and whether
   it is alive at launch.
6. **Dependency-edge audit (H5).** Dump nodes/edges (`MLX_HIP_GRAPH_DUMP`); compare graph edges
   against the true producer→consumer relationships; list any consumer node missing an edge to
   its producer. Distinguish *wrong-because-freed* (lifetime) from *wrong-because-raced*
   (ordering) — confirm which by re-running with all frees deferred: if output is correct with
   frees deferred, it's lifetime; if still wrong, it's ordering (H5).

**Phase 1 output:** a precise statement, e.g. "node X reads buffer Y; Y is freed at point Z
before the launch at W completes" (lifetime) and/or "node A consumes node B's output with no
edge B→A" (ordering).

## Phase 2 — Design: completion-gated lifetime ownership (no deep copy)

Principles:

- **No deep copy** of kernargs or buffers per commit/update. Instead, **own** the referenced
  memory and tie its lifetime to **graph-replay completion**.
- Each instantiated exec carries a **live set**: every allocation its nodes dereference
  (kernarg/Pack storage, dynamic-value device buffers, and any input/output/intermediate
  buffers still needed at launch).
- A **replay generation** is created per commit/update binding; each `hipGraphLaunch` is tagged
  with its generation. A completion signal (event / host-func on the worker side stream) marks
  a generation done.
- A generation's live set is reclaimed **only after** its launches complete (inflight → 0). On
  rebind (`ExecUpdate`), the prior generation's live set moves to a pending-reclaim list, freed
  by the completion handler once drained — **never synchronously inside the update call.**
- **Dynamic per-replay values:** one stable device buffer owned for the exec's life, contents
  updated before replay (memcpy node / in-place) — per the contract; no per-token pointer
  reassignment.
- **Intermediate buffers:** extend lifetime to the owning graph's completion via the
  per-generation live set, replacing any global "defer all frees" approach that balloons memory.

This is the same ownership idea that was observed to remove the garbage (keep referenced memory
alive), made **correct** (freed exactly when safe, no leak) and **scalable** (works at any
graph size, so the op cap can rise to where batching is meaningful).

## Phase 3 — Implementation

- `backend/rocm/device.{h,cpp}` — exec/generation live set, commit, synchronize, completion
  handler, op-cap.
- `backend/rocm/allocator.cpp` — per-generation buffer lifetime instead of global defer.
- `backend/rocm/worker.cpp` + event path — completion signal on a side stream (so reclamation
  doesn't stall the work stream).
- `backend/rocm/eval.cpp` — commit wiring.
- If **H5** confirmed: complete producer→consumer dependency registration on every graph-node
  kernel so replay order matches eager.
- If **H4** confirmed: replace any per-replay pointer reassignment with the stable-buffer
  update pattern.

## Phase 4 — Correctness validation

1. Standalone HIP lifetime tests from Phase 1 pass on the fixed path.
2. Per-op checksum bit-identical to eager across the full forward.
3. Walk the op cap 2 → 8 → 16 → 32 → 64: bit-identical + coherent 1000-token generation at each
   step. The cap at which anything breaks (if any) localizes a residual issue.
4. Long-generation stress under poisoned-free: stable memory, no leak, no OOM, no use-after-free.

## Phase 5 — Measure and decide

1. With correctness locked at a high cap, sweep the op cap and measure decode TPS vs the eager
   baseline. Record kernels-per-graph and `hipGraphLaunch` count per token to confirm batching
   actually collapsed the per-token launch count.
2. Add the worker side-stream + blocking event wait and re-measure (these only help once
   launches are batched).
3. Report the honest batched-vs-eager result. If batching wins, keep it; if not, the result is a
   clean, defensible measurement taken with correct lifetime management at full graph size.

## Build & run

Graphs require the 7.13 runtime.
```
cd build && HIP_PLATFORM=amd HIP_CLANG_PATH=/opt/rocm/core-7.13/lib/llvm/bin ninja chat
# eager (baseline):
LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib ./build/chat ~/mtp_convert/qwen36_4bit_v6 \
  --device 0 --no-think --ignore-eos --max-tokens 50 --temperature 0
# graphs: prefix the run with  MLX_USE_HIP_GRAPHS=1
```

## Env guards / diagnostics

- `MLX_USE_HIP_GRAPHS=1` — enable the graph path
- `MLX_MAX_OPS_PER_BUFFER=N` — cap kernel nodes per graph
- `MLX_GRAPH_NO_REUSE=1` — fresh instantiate each commit (isolate the update/reuse path)
- `MLX_GRAPH_FORCE_FROM=N` — force ops ≥ N to eager (bisection)
- `MLX_GRAPH_CHECKSUM` — per-op eager-vs-graph sum-abs dump (master diagnostic)
- `MLX_HIP_GRAPH_DUMP=1` / `MLX_GRAPH_DEBUG=1` — node/edge dump + tracing
- `MLX_NO_CONCAT_SPLIT=1` — keep Concatenate in-graph
- `HIP_LAUNCH_BLOCKING=1` — serialize launches while diagnosing

## Files / entry points

- `backend/rocm/device.{h,cpp}` — graph infra: add_kernel_node, commit, synchronize, launch.
- `backend/rocm/allocator.cpp` — buffer lifetime / free path.
- `backend/rocm/worker.cpp`, event path — completion signaling.
- `backend/rocm/eval.cpp` — gpu::eval commit wiring.
- `src/llm/models/qwen35_moe.cpp`, `src/common/generate.cpp` — engine decode loop.
