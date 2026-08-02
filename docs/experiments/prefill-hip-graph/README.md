# Prefill HIP graphs (gfx1150 / Qwen3.6-35B-A3B)

**Goal:** measure whether HIP graphs help **prefill** (multi-token forward), not pure decode.

**Branch tip at F1:** `09940a8` (`fix/mtp-stream-p0`)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` · gfx1150 ROCm · single process · **no MTP** · pure decode off

---

## First principles (what “prefill HIP graph” means here)

| Surface | What it is | Shape stability | Status on this stack |
|--------|------------|-----------------|----------------------|
| **Manual-node HIP graphs** (`use_hip_graphs`) | Build kernel-node graphs per commit; optional `hipGraphExecUpdate` | Needs recurring topology key | **Hard-off upstream** since mlx `9c5f1d0d` (net loss + historical RDNA3.5 segfault) |
| **Prefill ExecUpdate** (`MLX_GRAPH_PREFILL_REPLAY=1`) | After `prefill_key_` locks (2 consecutive full chunks), refresh cached execs | Full-size `prefill_step_size` chunks only | Dead without `use_hip_graphs()` true |
| **Pure stream capture** (`decode_capture_*`) | One stream capture of whole L=1 forward + arena | Fixed L=1 addresses | **Decode-only**; pure already slight APU regression — leave OFF |
| **Whole prefill-chunk stream capture** | Capture T=`step` forward once, replay | Needs PrefillArena (fixed temps + KV write offsets) | **Not implemented** — same blocker class as pure decode arena |

Prefill is long multi-token work chunked in `llm_default_prepare` (`include/mlx-lm/llm/llm_model.h`) at `prefill_step_size` (default 512, env `MLX_PREFILL_STEP`). Full-size chunks share topology; the last remainder does not. That is the natural capture key.

Friend claim (“graphs better for prefill than decode”) is about **amortizing launch/build over large multi-kernel forwards**. On this APU the prefill path is already compute-heavy (WMMA QMM); launch overhead is a smaller fraction than on launch-bound dGPUs.

---

## F1 result (honest A/B)

### Recipe

```bash
# Requires local mlx patch (see mlx-use_hip_graphs-optin.patch) rebuilt into ./build/chat
# Prompt ≈ 1653 tokens, step=128 → many full-size chunks
PROMPT=... # see F1_*.txt logs
# Baseline
env -u MLX_HIP_GRAPH_PREFILL -u MLX_GRAPH_PREFILL_REPLAY \
  MLX_PREFILL_STEP=128 MLX_PROFILE_PREFILL=1 \
  ./build/chat MODEL --max-tokens 16 --temperature 0 --no-think
# Experiment
MLX_HIP_GRAPH_PREFILL=1 MLX_GRAPH_PREFILL_REPLAY=1 \
  MLX_PREFILL_STEP=128 MLX_PROFILE_PREFILL=1 \
  ./build/chat MODEL --max-tokens 16 --temperature 0 --no-think
```

### Numbers (warmup ON, paired rounds, no MTP)

| Probe | prompt_tokens | prefill_s | pp/s | gen t/s (short) | peak mem |
|-------|---------------|-----------|------|-----------------|----------|
| r1 baseline eager | 1653 | 15.045 | **109.87** | 30.07 | 18.3 GB |
| r1 exp graph | 1653 | 14.610 | **113.15** | 29.15 | 20.8 GB |
| r2 exp graph | 1653 | 14.472 | **114.23** | 29.10 | 20.8 GB |
| r2 baseline eager | 1653 | 14.810 | **111.62** | 30.35 | 18.3 GB |
| r3 baseline turn1 | 1653 | 14.827 | **111.49** | — | 18.3 GB |
| r3 exp turn1 | 1653 | 14.420 | **114.64** | — | 20.8 GB |
| r3 baseline turn2 (history) | 3391 | 31.002 | **109.38** | — | — |
| r3 exp turn2 (history) | 3391 | 29.020 | **116.85** | — | — |

**Mean single-turn (r1+r2):** eager ≈ **110.7 pp/s**, graph ≈ **113.7 pp/s** → **~+2.7%**  
**Longer re-prefill (r3 turn2):** **+6.8%** (109.4 → 116.9)  
**≥10% win bar:** **not met**

Notes:
- No segfault on gfx1150 in these runs (historical RDNA3.5 prefill graph SEGV not reproduced here).
- Graph path peaks **~+2.5 GB** higher.
- Profile’s `gpu_exec` drops on graph path (~0.16s vs ~0.88s) while `host_build` (prepare wall, includes chunk evals) stays similar — total pp/s is the product metric.
- Cold run without warmup (confounded first probe) is discarded for claims.

### Verdict F1

**FAIL** for product enablement (≥10% fixed long-prompt bar).  
**Not blocked** for further fires: opt-in path builds and runs; delta is small positive, not a free win.

---

## Code map

| Path | Role |
|------|------|
| `src/common/generate.cpp` `TokenIterator::prepare` | Forces `gpu_set_graph_decode_mode(false)` for prefill; F1 adds env banner under `MLX_PROFILE_PREFILL` / graph flags |
| `include/mlx-lm/llm/llm_model.h` `llm_default_prepare` | Chunked prefill at `prefill_step_size` |
| `build/_deps/mlx-src/mlx/backend/rocm/device.cpp` `use_hip_graphs` | Gate for manual-node graphs (**default false**) |
| same file `commit()` / `prefill_key_` | Prefill topology lock + `MLX_GRAPH_PREFILL_REPLAY` ExecUpdate |
| same file `decode_capture_*` | Pure L=1 stream capture — **not** prefill |
| `src/common/graph_decode.cpp` | Fixed-address pos/input for pure decode only |

### Upstream mlx history (do not re-litigate decode)

- `5ed2d3d3` prefill replay scaffolding (WIP, default off)  
- `345fe570` prefill uses capture+ExecUpdate; still ~5–8% **under** eager historically  
- `9c5f1d0d` hard-off `use_hip_graphs()` — flags became inert  

F1 patch restores env opt-in only for measurement: `mlx-use_hip_graphs-optin.patch` (apply under `build/_deps/mlx-src`, rebuild `chat`). **Not** product default.

---

## Blockers for a real ≥10% prefill-graph win

1. **No PrefillArena** — stream-capture whole chunk cannot bake pointers when temps reallocate and KV write offsets advance.  
2. **Remainder chunk** — last `T < step` breaks topology; pad-to-bucket needs correct attention mask (not done).  
3. **Rebuild-per-eval tax** — without true build-once (build-skip / fixed addresses), ExecUpdate still pays graph *build* each chunk. Historical note: “true build-once needs build-skip for captured nodes.”  
4. **APU already compute-bound on prefill** — launch batching has little headroom vs WMMA GEMM.  
5. **Patch lives in FetchContent tree** — product path needs mlx PR on `rocm-support`, not only local `_deps`.

---

## F2: whole-chunk one graph (`MLX_PREFILL_ONE_GRAPH`)

Engine opt-in: during `prepare()` only, set `gpu_set_graph_decode_mode(true)` so mlx does **not** mid-forward commit multi-token chunks. Requires `MLX_USE_HIP_GRAPHS=1` (decode-mode uses the decode graph gate). Default remains eager/safe.

| | mean prefill_s | mean pp/s |
|--|----------------|-----------|
| eager | ~14.82 | **~111.5** |
| ONE_GRAPH+USE_HIP_GRAPHS+REPLAY | ~14.31 | **~115.5** (**~+3.6%**) |

**FAIL** bar ≥10%. DECODE-flag-only config regressed. Details: `F2_RESULTS.md`.

---

## F3: absorb tail + step 128/512 (FINAL)

`MLX_PREFILL_ABSORB_TAIL=1` runs all T>1 prefill inside prepare (leave 1 tok for step). Best graph stack still **flat/slightly worse** on long prompt:

| | step128 pp/s | step512 pp/s |
|--|--------------|--------------|
| eager | 112.0 | **132.3** |
| absorb+ONE_GRAPH+USE+REPLAY | 111.3 | 131.2 |

**Schedule STOP:** three consecutive fires fail ≥10% long-prompt bar. Design: `PREFILL_ARENA_DESIGN.md`. Product: **keep prefill eager**.

## Schedule status

| Fire | Δ long pp/s | Outcome |
|------|-------------|---------|
| F1 split+replay | ~+2.7% | FAIL |
| F2 ONE_GRAPH | ~+3.6% | FAIL |
| F3 absorb+ONE_GRAPH | ~−0.7% | FAIL → **stop** |

## Next (only if mlx PrefillArena lands)

1. Implement PrefillArena + stream capture in mlx `rocm-support`.  
2. Re-run F1 long-prompt recipe; require ≥10% before any product opt-in.  
3. Prefer `MLX_PREFILL_STEP=512` for prefill throughput (eager already wins vs 128) — orthogonal to graphs.  


---

## Logs

- `F1_r1_baseline_eager.txt`, `F1_r1_exp_prefill_graph.txt`  
- `F1_r2_baseline_eager.txt`, `F1_r2_exp_prefill_graph.txt`  
- `F1_r3_baseline_2turn.txt`, `F1_r3_exp_2turn.txt`  
- Confounded cold run (discard): `F1_baseline_eager_step128.txt`  
