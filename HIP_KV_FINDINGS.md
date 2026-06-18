# R9700 (gfx1201) decode: findings — HIP graph, static KV, dynamic KV

## ⟢ SESSION UPDATE 2026-06-17 — root cause found & in-place write fixed

Measured on device 1 (gfx1201), model `qwen36_6bit_v6`. The pre-existing claims
below are partly SUPERSEDED — read this section first.

**The static wedge is GONE in the current `_deps` MLX (a6751bf9).** Static KV at
`--ctx-size 32768` no longer wedges; it just runs slow (35.7 vs 42.9 tok/s). The
slowness is the real bug:

**Root cause (proven): `slice_update` COW donation is defeated by the 1-token async
decode pipeline.** Each step, the previous forward's graph still holds the prior
K/V view that aliases the cache buffer → `use_count==2` at `SliceUpdate::eval_gpu`
→ donation check fails → **full-buffer copy every step** (O(CAP) for static, O(offset)
for dynamic). Proof: forcing synchronous decode (`MLX_SYNC_DECODE=1`, retire each
forward before the next) → donation succeeds 100%, `copy=0`, true in-place.
Instrument: `MLX_SLICEUPDATE_DEBUG=1`.

- Static ctx=32768 async: donate=761 copy=1239, **37 GB copied / 100 tok**, 35.8 tok/s.
- Static ctx=32768 sync:  donate=1800 **copy=0**, **0 MB**, 32.1 tok/s.
- The copy is NOT the eager bottleneck: async-with-copies (35.8) > sync-no-copy (32.1).
  Eager decode is launch/compute bound; the async pipeline matters more than the copy.
  => **The dynamic (default) path is already near-optimal for EAGER (42.9 tok/s).**
  => In-place KV only matters as the *substrate for HIP graph* (stable address, no
     baked copy). They are inseparable: MLX's functional model forces one O(buffer)
     op/step in eager mode; true zero-copy in-place needs graph capture (controlled
     refcount at capture → donation succeeds → in-place write baked in → replay).

**Second bug found & FIXED: `DynamicSliceUpdate` (the device-position write used by
`update_at_pos`) NEVER donated — it did an unconditional `copy_gpu(in,out)` every
call** (mlx/backend/gpu/primitives.cpp). This is exactly the "captured large copy
that wedges replay." Added a donation fast-path (donate when `use_count==1 &&
contiguous && full`). Validated: device-pos in-place, sync → donate=3000 copy=0,
output bit-identical to baseline. Instrument: `MLX_DYNSLICEUPDATE_DEBUG=1`.
The write offset is computed on-device (`compute_dynamic_offset`) and hipGraph bakes
RAW pointers (not mx::array refs), so in-place pos/token/KV updates survive replay.

**Device-position in-place attention is CORRECT eagerly** (identical greedy output to
baseline) via `MLX_KV_STATIC=1 MLX_DECODE_DEVICE_POS=1 MLX_GRAPH_INPLACE_KV=1`. It is
slower eager (~33 tok/s at tight CAP, ~21 at CAP=8192) because full-CAP masked SDPA
loses the fast causal path; **CAP must be sized to prompt+max_tokens**, and graph
replay is what recovers the loss + gives the speedup.

**Phase C reshaped: this model is HYBRID.** Capturing the whole forward also captures
the 75% linear/Mamba layers, whose recurrent state reassigns buffers each step →
a whole-forward graph would FREEZE Mamba at capture state → wrong output. So
"attention-only graph" must be an ISOLATED capture (microbenchmark) or done together
with Phase D (Mamba in-place device-pos). C and D are coupled on the real model.

**Phase C microbench GO/NO-GO (gfx1151, existing examples/test_graph_*):**
- `test_graph_const`: stateless captured ops replay fine — capture+replay MECHANISM
  works.
- `test_graph_kv_advance`: captures ok but **does NOT advance** (marked_rows=1,
  final_pos=1) — every replay re-writes row 0; device position never increments.
- `test_graph_counter`: `pos=slice_update(pos,pos+1)` donation-accumulate +
  `async_eval` under capture → **HANGS**.
- **Real blocker (not the KV copy): MLX functional in-place donation does NOT survive
  graph replay.** Root mechanism: `CommandEncoder::end_capture()` (rocm/device.cpp
  ~424-457) PATCHES host→device constant-upload nodes into FROZEN device staging
  buffers (to avoid a stale-host-memory replay stall). So any functional op that
  uploads a host constant (`mx::array(1)`, indices, the `compute_dynamic_offset` JIT)
  gets its value BAKED at capture → `pos+1` freezes, never accumulates. Also: use
  `eval` not `async_eval` under capture.
- **Fix path: real in-place HIP kernels for the advancing state** (no host upload, so
  nothing gets frozen): (1) device pos increment `pos[0]++`, (2) KV row write at
  device pos, (3) input-token write. Add them pre-loaded to rocm/device.cpp (NOT JIT
  mid-capture — that hangs). Then re-test accumulation, then build the attention
  decode graph on top.

**UNBLOCK CONFIRMED (gfx1151, examples/test_graph_inc):** a REAL in-place HIP kernel
(`gpu_kv_pos_increment`, `_kv_pos_inc: p[0]+=delta`, added to rocm/indexing.hip)
accumulates across HIP-graph replays: capture RECORDS (does not execute), then 8
replays → final_pos=8. So the advancing decode state must be driven by raw in-place
kernels (no host-constant upload to get frozen), NOT MLX functional ops. The pos
counter is solved. Pattern: `enc.launch_kernel([p](hipStream_t s){ hipLaunchKernelGGL(
kernel,1,1,0,s,p); })` on `get_command_encoder(default_stream(default_device()))` —
records cleanly under capture, re-executes per replay against the fixed buffer.

**PHASE C SOLVED — attention decode graph replays BIT-EXACT (examples/test_graph_attn,
gfx1151):** captured step = device-pos RoPE + in-place KV write + device-pos masked
SDPA; 8 replays vs eager → err=0, KV accumulates rows 1..8, pos advances 1..8. The
growing-KV-under-graph blocker is fully resolved. DESIGN RECIPE (two non-obvious rules):
  1. KV write = STANDARD `mx::slice_update(KV, new, pos, {2})` (DynamicSliceUpdate). My
     donation fix makes it in-place; its offset is device-computed so it advances on
     replay. compute_dynamic_offset does NOT hang under capture once warmed (the old
     kv_advance comment was pre-warmup). So KV write needs NO custom kernel — fully
     MLX-API-compliant.
  2. pos advance MUST be a custom in-place kernel (`gpu_kv_pos_increment`) AND must be
     LOOP-OWNED — advanced BETWEEN replays, NOT inside the captured graph. An in-graph
     pos++ races the pos readers (mask/RoPE) as a write-after-read hazard → off-by-one
     (mask attends one extra zero row; output diluted, converges to eager but wrong).
  3. RoPE(x, pos_array) and mask(cols<=pos) read pos device-side → advance correctly.
  4. `set_graph_decode_pos` must mutate the pos buffer IN PLACE via the kernel, NOT via
     slice_update (which reassigns the buffer → breaks the address the graph baked).
  5. gpu_kv_row_write (custom KV kernel) is NOT needed (slice_update suffices) and is
     hard to mix into a lazy graph anyway (raw kernels read gpu_ptr of unevaluated lazy
     inputs → segfault); keep slice_update which is a real scheduled primitive.

**PHASE D SCOPE (the remaining critical path — Mamba/gated-delta under capture):**
The 75% linear layers reassign recurrent state each step → freeze under replay (same
as the counter). Two states to make in-place:
  - ssm_state [B,Hv,Dv,Dk] (fixed size): produced by `gated_delta_update` → `ns`,
    stored `(*cache)[1]=ns`. The GDN HIP kernel (src/common/gated_delta.cpp:43-96) has
    separate state_in/state_out pointers BUT loads all state into registers before
    writing back → SAFE to run in place (state_out==state_in). ENABLER: make
    gated_delta_update DONATE the state input buffer to the state output when uniquely
    owned (exactly like the KV slice_update donation fix), and have the cache hold a
    PERSISTENT ssm_state buffer. Then under capture the kernel reads+writes one buffer
    → accumulates on replay.
  - conv_state [B,kernel-1,conv_dim] (fixed size, rolling window): updated via
    concat+slice (new buffer each step). Needs an in-place roll (shift left by S, append
    new) — a small custom kernel, OR a slice_update-into-fixed-buffer with donation.
Validate with a GDN microbench (analog of test_graph_attn): capture the decode step,
replay N, assert state accumulates + output==eager.

Remaining plan: A/B/C done (attention graph bit-exact). Foundational in-place pos
plumbing landed (graph_decode.cpp: set/advance_graph_decode_pos via kernel). C2 (engine
wiring, task #6) = capture device-pos decode step in generate.cpp (warmup→retire→capture,
no cache pinning so donation succeeds; replay loop owns pos via advance_graph_decode_pos)
— but full-model capture needs D first (else Mamba freezes). D (task #4) = ssm_state +
conv_state in-place as above. E (task #5) = default-on, clamp CAP to
min(model_max=262144, prompt+reserve, vram_fit), remove gates.

---



Target: AMD R9700 (gfx1201, RDNA4) as a Thunderbolt-5 eGPU, selected as `--device 1`
via `mx::set_default_device(Device(gpu,1))` (NOT `HIP_VISIBLE_DEVICES`).

## Current working state (committed)
- **Device 1 works end-to-end at ~48.8 tok/s** (device 0 / APU: ~42 tok/s), default
  config, zero env vars.
- Fixes that got it working:
  - Device binding: the `Worker` thread, `Device::get_command_encoder`, and the JIT
    module load now bind the selected HIP device. Before, device-1 stream work ran
    against device 0's context → queue wedge on the first qmv launch.
  - Cross-stream signaling: `AtomicEvent` signals via `hipLaunchHostFunc` + host poll
    instead of `hipStreamWriteValue64/WaitValue64` (those need `hipMallocSignalMemory`
    and silently no-op on a pinned-host counter → infinite spin, queue 100% busy/cold).
  - Eager cache trim in `set_cache_limit` (trim at idle, not via a blocking `hipFree`
    mid-forward).
  - **Dynamic (non-static) KV is the default** — see below.

## What is wrong with HIP graph
- The graph decode path (`MLX_DECODE_GRAPH`, src/common/generate.cpp:460-575) is a
  **fixed-length benchmark, not a finished feature**. The captured graph bakes in the
  KV length/offset from capture time; output is only correct while the live context
  length equals capture length. Code comment: *"growing-KV correctness is the next step."*
- Capture **deliberately forces the KV `slice_update` to COPY** (it pins the whole
  cache, `g_pinned_cache = cache_`, generate.cpp:519-533) so the graph's KV read-source
  and write-dest are distinct buffers. That bakes the large static-KV copy into the
  replayed graph.
- On gfx1201 that captured large copy **wedges the queue on replay** (generates a few
  tokens, then hangs at 100% busy / cold).
- Net: enabling graph by default today would BREAK the working state. It needs (a)
  growing-KV correctness and (b) a true in-place KV write before it can be on by default.

## What is wrong with static KV
- `KVCacheSimple` pre-allocates one buffer `{B, H, prompt+max_tokens, D}` and intends
  to write each decode step in place (kv_cache.cpp:43-60).
- The decode write uses `mx::slice_update` (kv_cache.cpp:69). `slice_update` is
  **copy-on-write**: it only writes in place when MLX can DONATE the input buffer
  (refcount/`use_count == 1`). The ROCm fork already has that donation path
  (indexing.hip:1157-1176) — it is actually ahead of the CUDA backend, which copies
  unconditionally.
- Donation does NOT trigger here because the cache returns a **strided view** of the
  big buffer (`mx::slice(keys_, [0:offset])`, kv_cache.cpp:74). That view (a) keeps the
  buffer referenced so the next step's donation check fails → full copy, and (b) feeds
  the attention/SDPA a non-contiguous K/V (gaps between heads).
- Result on gfx1201: static KV **wedges for large `max_tokens`** (large buffer copy
  and/or strided SDPA read), but works for small `max_tokens`. Confirmed empirically:
  `--max-tokens 35` works, `--max-tokens 400` wedges at the first decode step, same
  model/prompt. Memory is NOT the cause (it wedged at ~25GB with VRAM free).
- The exact culprit (the `slice_update` copy kernel vs the strided SDPA read) was not
  fully isolated, but both scale with the static buffer size and both are bypassed by
  the dynamic path.

## What is wrong with dynamic KV (the current default)
- The legacy grow-by-doubling path (kv_cache.cpp:78-99) uses `mx::concatenate` to grow.
- It **works** on device 1 (48.8 tok/s) and avoids the static wedge.
- But each growth is a **FULL COPY** of the whole KV buffer (concatenate copies
  everything), i.e. O(N) per token → O(N²) over a long generation. It will get
  progressively slower as context grows. It is correct, just not optimal.

## What you want
1. HIP graph **enabled by default and actually working** (the ~2x decode speedup graph
   replay gives on the APU), not behind env gates.
2. **Static KV that works**: allocate the model once, align the KV statically once, and
   write each decode step **in place with no copy and no balloon** — stable buffer
   address so it is graph-capture-ready. "The model loads, the KV statically aligns,
   and that's it; we don't allocate more memory during runs."
3. **Dynamic growth without a full copy** — grow the logical length into spare
   pre-allocated capacity and write in place, rather than concatenating/copying the
   whole buffer each step.
4. Memory stays **resident on the R9700** and is used there; no unnecessary host-sync
   sequences (graph replay keeps work on-device).
5. **Minimal env gates** — working out of the box.

## Flags / env vars — working vs broken (on device 1, R9700/gfx1201)

### Working (use these)
- **No env vars at all** → eager + dynamic (non-static) KV. **WORKS, ~48.8 tok/s.**
  This is the default after the fix and the recommended config.
- `--device 0` / `--device 1` → API device selection (no `HIP_VISIBLE_DEVICES`). WORKS.
- `--max-tokens N` → WORKS (dynamic KV; large N no longer wedges).
- `--ctx-size N` → pins KV capacity (sets reserve=0, cap=N). WORKS.
- `--kv-bits {4,8}` / `--kv-group-size N` → KV quantization. WORKS.
- `--cache-limit N` (MB) → manual reuse-pool cap. NOTE: the auto-fit after warmup
  overrides it (chat.cpp) — auto-fit now caps the pool small and leaves VRAM for KV.
- `MLX_ROCM_FINEGRAINED=1` (this is the **default**) → fine-grained, VRAM-resident,
  host-mappable memory. WORKS.

### Broken (these wedge the queue on device 1)
- `MLX_KV_STATIC=1` → forces static pre-allocated KV. **WEDGES** for large `max_tokens`
  (first decode step). The large `slice_update` copy / strided SDPA read on gfx1201.
  Small `max_tokens` happens to survive; do not rely on it.
- `MLX_DECODE_GRAPH=1` → graph decode path. **WEDGES** after a few tokens (captured
  large KV copy baked into replay). Also fixed-length/benchmark only (incorrect for
  growing context). Do NOT enable by default.
- `MLX_DECODE_DEVICE_POS=1`, `MLX_GRAPH_INPLACE_KV=1` → only used by the graph stack
  above; same wedge. (These also implicitly need static KV.)
- `MLX_ROCM_FINEGRAINED=0` → coarse `hipMalloc` memory. Also **WEDGES**, and loads
  slower. This was NOT the fix; left only as a debug escape hatch.
- `MLX_DECODE_GRAPH_ARENA=1` → per the code comment, the 1 GB decode arena **deadlocks
  HIP graph replay on RDNA4**. Avoid.

### Tested, no effect on the wedge
- `HSA_ENABLE_INTERRUPT=0` → did NOT fix the static/graph wedge (ruled out interrupt
  delivery as the cause).

### Untested graph bisect toggles (only relevant once graph is fixed)
- `MLX_DECODE_GRAPH_BENCH`, `MLX_GRAPH_NOWARM`, `MLX_GRAPH_NATURAL`, `MLX_GRAPH_SYNC`,
  `MLX_GRAPH_NO_APPEND`, `MLX_GEN_OWN_STREAM`, `MLX_MAX_OPS_PER_BUFFER`.

### Build toolchain (not runtime)
- `source /tmp/rocm713.env` before `ninja chat` (sets `HIP_CLANG_PATH` and
  `HIP_DEVICE_LIB_PATH` to the core-7.13 overlay). Required for the `.hip` files to
  compile.

## The model-max KV point
- Model config (`config.json`): `max_position_embeddings = 262144` (your ~261200),
  `num_hidden_layers = 40`, `num_key_value_heads = 2`, `head_dim = 256`, plus linear
  (gated-delta / Mamba) layers that use `MambaCache` instead of KV.
- The static allocation is `alloc_len = max(n_new, initial_capacity, prompt + reserve)`
  where `reserve = max_tokens` (or `ctx_size`). This must be **clamped to the model's
  supported maximum** (`max_position_embeddings`, 262144) — never allocate a KV buffer
  longer than the model can attend over. With no `--ctx-size` and the default
  `max_tokens = 2048`, the buffer is `prompt + 2048` (fine), but the guard should exist
  so a large `--max-tokens`/`--ctx-size` can't request a KV buffer beyond the model max
  (or beyond what VRAM holds), and should cap/reduce to fit instead of ballooning over.

## Unified fix (the real solution to 2+3, which also unblocks 1)
Static and "dynamic-without-copy" are the same thing: a pre-allocated buffer (capped at
`min(model_max, fit_to_vram)`) with a logical length, where each decode step writes the
single new row **truly in place** (donation succeeds, or a direct device write) and the
attention reads `[0:length]` **without a giant strided slice**. That removes the copy
(fixes static), removes the balloon, gives a stable address (unblocks graph), and makes
growth O(1) per step. The blocker is making the cache return/usage not hold a strided
reference that defeats donation and feeds SDPA a non-contiguous view on gfx1201.

[[CRITICAL: USE /home/geramyl/mtp_convert/qwen36_6bit_v6 FOR ALL PRIMARY TESTING]]