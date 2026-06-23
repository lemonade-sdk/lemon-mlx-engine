# HIP-Graph Reuse — Status & What's Left

ROCm MLX backend (NripeshN/mlx `rocm-support`). Model: Qwen3.6-35B-A3B q4.
Devices: gfx1151 APU (HIP dev 0) + gfx1201 R9700 over TB5 (HIP dev 1).

## 1. Why reuse matters

The auto-batch path collapses a token's ~2000 individual `hipLaunchKernel` calls
into ~80 `hipGraphLaunch` + ~30 inline library launches. That launch reduction is
the whole point on a **discrete GPU over a slow link (TB5 eGPU)**: each launch is a
command crossing PCIe, so fewer submissions = less link traffic and latency. On the
unified-memory APU it doesn't help (launches are cheap, decode is bandwidth-bound),
which is why graphs ≈ or < eager there.

**But we don't actually reuse the graph today.** Every token we *rebuild and
reinstantiate* it from scratch, which (a) wastes CPU and (b) on TB5 re-uploads the
exec over the link every token — partly cancelling the launch savings.

## 2. Current state (measured)

Per 50-token run, default path (`MLX_GRAPH_REUSE_STATS=1`):

```
reuse-stats: update=0  reinst=4043  new=63 | graph_launches=4106 inline=30 kernel_nodes=70138
graph-time : addnode=554ms  deps=5ms  reinstantiate=420ms  launch=63ms
```

- `update=0, reinst=4043` → **zero exec reuse**: every commit reinstantiates.
- `addnode=554ms` → we re-run ~70k `hipGraphAddKernelNode` (rebuild the graph)
  every token.
- `launch=63ms` → the actual launches are ~6% of graph CPU. Build + instantiate
  is the other ~94%.

Throughput (correct, stable today):

| | APU dev0 | R9700 dev1 |
|---|---|---|
| eager | 41 tok/s | 45 tok/s |
| graphs (reinstantiate) | 38 | 35.7 |

So graphs are currently **slower than eager** — entirely because of the per-token
rebuild + reinstantiate, not the launches.

## 3. The two things that must change

There are **two independent costs**, both paid every token:

1. **Reconstruction** (`addnode` 554ms): we call `hipGraphAddKernelNode` ~70k×
   per token to rebuild an identical graph. Fix = **build-once**: build the graph
   for a topology one time, then on recurring tokens *don't* rebuild it.
2. **Instantiation** (`reinstantiate` 420ms / re-upload on TB5): we make a fresh
   `hipGraphExec_t` each commit. Fix = **reuse the exec** (refresh its changed
   params, or just relaunch it).

A perfect fix needs BOTH. Build-once without exec reuse still re-instantiates;
exec reuse without build-once still pays the 554ms rebuild.

## 4. Exec-reuse: three mechanisms tried, why each fails

To reuse an exec across tokens, its kernel-node params must be refreshed (the
per-token intermediate buffers land at **new addresses**, so the baked device
pointers change). Three ways, all currently blocked:

### (a) `hipGraphExecUpdate` — mis-maps params
Returns `hipGraphExecUpdateSuccess` every time (4479/4479) but produces garbage in
the model's complex DAG. **Not a keying bug on our side** — instrumented the
per-node signature (func + full grid + full block) of the matched slot vs the
current build: **0 collisions**, topology is identical, and it still mis-maps. It
also works fine in a standalone unambiguous chain. Conclusion: ROCm's ExecUpdate
node-matching is unreliable for graphs with many same-(func,dims) nodes. Dead end.

### (b) `hipGraphExecKernelNodeSetParams` (deterministic per-node) — OOBs
The right idea: we build nodes in a known order, so refresh each node's params by
*handle*, bypassing ExecUpdate's matching. **Proven correct in every standalone
test** (pointers, by-value structs, changing values, module funcs, repeated reuse,
8-iter loops). In the engine it still faults: a launch reads out-of-bounds
(`copy_gg_byval`, grid 2097152). Partially root-caused — the exec-pool key used the
grid/block **product**, colliding distinct shapes; fixed to full dims, but it
**still OOB'd** after that. **Not fully root-caused.** This is the most promising
path if the remaining OOB is cracked (probably another kernarg-layout / lifetime
detail specific to the engine's Pack storage).

### (c) Pure relaunch (no update at all) — needs stable addresses
If the per-token buffers landed at the **same addresses** every token, the cached
exec stays valid and we just `hipGraphLaunch` — zero per-node calls, the minimal-
call ideal. Measured: relaunch drops `reuse/instantiate` from 420ms → **6ms**. But
the `hipMallocAsync` pool hands out **fresh addresses** each token, so the cached
exec reads last-token's memory → garbage. Needs deterministic per-token addresses
(a per-forward reset arena, or `mx.compile`'s buffer reuse).

## 5. The build-once problem (the 554ms)

Even with a working refresh, we still rebuild the graph every token. To skip it we
must **not** call `hipGraphAddKernelNode` when an exec for this topology already
exists. The obstacle: the topology key is only known *after* adding the nodes
(chicken-and-egg). Solution shape:

- During eval, **record** each op's `{func, grid, block, smem, params, deps}`
  without creating graph nodes; accumulate the key incrementally.
- At commit, look up the exec pool by key:
  - **hit** → refresh params (SetParams) or relaunch (stable addrs) + launch.
    No `hipGraphCreate` / `AddKernelNode` / `Instantiate`.
  - **miss** → now build the graph from the recorded ops, instantiate, cache.

Steady-state decode = all hits → near-zero per-token graph CPU, one launch per
chunk. This is the structural change that removes the 554ms.

## 6. Bandwidth caveat (only matters on the APU)

On the APU, even a *perfect* reuse barely helps: removing instantiate (relaunch
6ms) moved TPS ~1 tok/s, because decode is **memory-bandwidth-bound** there and the
graph CPU is already overlapped with the GPU. So the reuse win is specifically for
the **R9700/TB5** (PCIe launch + exec-upload traffic), not local APU throughput.
The static-KV / capture-once route additionally pays **full-capacity attention
bandwidth** (scales with ctx_size: 34.7→23.4 tok/s as ctx 256→4096), so if reuse
is pursued via a fixed-shape captured graph, it also needs a **device-length
flash-attention** kernel (read a device-side KV length, skip positions beyond it)
to avoid that.

## 7. What's left to finish — concrete checklist

In priority order for the R9700/TB5 win:

1. **Crack the SetParams OOB** (mechanism b). Reproduce with a poisoned-free /
   per-node bisection in the engine; the standalone is clean, so it's an engine
   integration detail (kernarg Pack layout/lifetime, or a node whose grid the
   pool-key still aliases). This unblocks instantiate-free reuse.
   - *Alternatively*, **stable per-token addresses + relaunch** (mechanism c):
     route auto-batch decode intermediates through a per-forward-reset bump
     allocator so token N+1 reuses token N's addresses → relaunch, zero per-node
     calls. (User dislikes the engine's DecodeArena specifically; a clean
     deterministic allocator for the auto-batch path is the same idea done right.)
2. **Build-once** (section 5): defer node creation, key incrementally, only
   build+instantiate on cache miss. Removes the 554ms/token rebuild. Independent
   of which refresh mechanism (1) lands.
3. **Device-length flash attention** — only if reuse is pursued via static-shape
   captured graphs; lets a fixed-shape graph attend over the live length without
   full-capacity bandwidth.
4. **Re-measure honestly on the R9700** with build-once + working reuse: confirm
   per-token launch *and* exec-upload counts drop, and whether decode now clears
   eager over TB5.

## 8. What's already solid (committed, `rocm-support`)

- Chain-edge ordering fix; JIT/Custom kernels as graph nodes (module `hipFunction_t`
  IS a valid graph node on 7.13 → eliminated ~320 inline splits/token).
- Default-on graphs; hipBLASLt-under-graphs crash fix; deferred-free leak fix.
- Thread-safety: `encoders_` map race + async-pool race (fixed the R9700
  load/decode crashes that looked device-specific).
- Async stream-ordered frees (no pipeline drain) + memory bound → R9700 graph
  decode peaks 20.4GB (was 34/OOM), stable, coherent.
- Clean-build fixes (Pack default-ctor, uint2/uint4 nontemporal hack) → the Python
  `mlx` bindings now build for ROCm and run mlx-lm coherently on both devices.

So the platform is correct and stable; **graph reuse is the one remaining lever**
for making graph decode beat eager on the R9700/TB5.
