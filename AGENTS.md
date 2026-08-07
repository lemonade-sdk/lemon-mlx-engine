# AGENTS.md — lemon-mlx-engine

Operational guide for agents (and humans) building and running this engine,
with an emphasis on the AMD ROCm / gfx1151 (Ryzen AI Max / Radeon 8060S) path.

The engine is an OpenAI-compatible C++ inference server (`examples/server.cpp`,
`src/common/server.cpp`) built on a ROCm-enabled fork of MLX
(`NripeshN/mlx`, branch `rocm-support`, fetched via CMake `FetchContent`).

---

## 1. ROCm runtime environment (REQUIRED on gfx1151)

MLX JIT-compiles some kernels at runtime with hipRTC/comgr — notably the
Gated-Delta-Net custom kernels used by the hybrid Qwen3.5/3.6 models
(`custom_kernel_gdn_conv_step_*`). The hipRTC compile is handed exactly one
include path, built as `-I${ROCM_HOME}/include` (see
`mlx/backend/rocm/jit_module.cpp::rocm_home()`), resolved from, in order:

1. `ROCM_HOME`
2. `ROCM_PATH`
3. `/opt/rocm` (Linux fallback)

On this box ROCm 7.2.1 installs its headers under a versioned **core-7.13**
subtree, so the HIP runtime header lives at:

```
/opt/rocm/core-7.13/include/hip/hip_runtime.h
```

The default `/opt/rocm/include` does **not** contain `hip/hip_runtime.h`. If
`ROCM_HOME`/`ROCM_PATH` are unset (or point at `/opt/rocm`), the runtime kernel
compile fails and the server dies during warmup:

```
Error: Failed to compile kernel 'custom_kernel_gdn_conv_step_bfloat16_...':
  fatal error: 'hip/hip_runtime.h' file not found
```

**Fix / requirement:** point ROCm at the 7.13 runtime + build headers before
launching the server (and before building):

```bash
export ROCM_HOME=/opt/rocm/core-7.13
export ROCM_PATH=/opt/rocm/core-7.13
```

With this set, warmup JIT-compiles the GDN kernels and the model loads
(`Model loaded. Memory: active=19.8 GB` for Qwen3.6-35B-A3B q4).

> If your ROCm install is a flat `/opt/rocm` (headers directly under
> `/opt/rocm/include/hip/`), the fallback is fine and no override is needed.
> The override is specifically for the versioned `core-7.13` split layout.

---

## 2. Default environment configuration

The **default / product mode** is: eager decode, thinking enabled, **no MTP**,
**no speculative decode**, no pure-graph. That is simply launching the server
with no special flags. The environment below reflects that default; every
`MLX_*` knob is opt-in and off unless listed as on.

### Required (ROCm gfx1151)

| Variable       | Value                    | Purpose                                            |
| -------------- | ------------------------ | -------------------------------------------------- |
| `ROCM_HOME`    | `/opt/rocm/core-7.13`    | hipRTC/comgr include root (see §1)                 |
| `ROCM_PATH`    | `/opt/rocm/core-7.13`    | fallback for the same, and for tooling             |

### Model cache (HuggingFace)

| Variable        | Purpose                                                      |
| --------------- | ----------------------------------------------------------- |
| `HF_HUB_CACHE`  | HF cache dir (highest priority for model resolution)        |
| `HF_HOME`       | HF home; `$HF_HOME/hub` used as cache if `HF_HUB_CACHE` unset|
| `HF_TOKEN`      | HF API token for private models                             |

### Decode / model tuning knobs — leave UNSET for default mode

These are diagnostics / opt-in optimizations. Default behavior (unset) is the
supported, stable path on gfx1151.

| Variable                   | Default (unset) behavior                                        |
| -------------------------- | -------------------------------------------------------------- |
| `MLX_LOAD_MTP_HEAD`        | MTP head skipped (not needed for eager/no-MTP). Set `1` only for `--use-mtp`. |
| `MLX_ENABLE_QUANT_FUSE`    | Quant projection fuse OFF (unfused matmuls are the eager-safe path on ROCm). Set `1` to opt in. |
| `MLX_DECODE_GRAPH_PURE`    | Pure-relaunch graph decode OFF (eager is faster on gfx1151). Set `1` to opt in. |
| `MLX_DECODE_GRAPH_PURE_OFF`| Forces pure-graph OFF even if the above is set.               |
| `MLX_SYNC_DECODE`          | Async decode. Set `1` to fully retire each forward (debugging). |
| `MLX_GDN_NO_FUSED` / `MLX_GDN_NO_FUSED2` | Fused Gated-Delta-Net decode path used.          |
| `MLX_DECODE_ARENA_MB`      | Phase-scoped decode arena disabled.                           |

---

## 3. Build (ROCm)

MLX is pinned/tracked in `CMakeLists.txt` (`FetchContent` of
`NripeshN/mlx` @ `rocm-support`). To pick up a new MLX `rocm-support` tip:

```bash
# advance the fetched MLX checkout, then rebuild the changed objects
git -C build/_deps/mlx-src fetch origin rocm-support
git -C build/_deps/mlx-src checkout <sha>       # e.g. rocm-support tip

export ROCM_HOME=/opt/rocm/core-7.13 ROCM_PATH=/opt/rocm/core-7.13
cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build build --target server -j 32
```

Configure should report `Setting CMAKE_HIP_ARCHITECTURES to: gfx1151` and
`ROCm WMMA support: ON`.

---

## 4. Run — default mode (no MTP)

```bash
export ROCM_HOME=/opt/rocm/core-7.13 ROCM_PATH=/opt/rocm/core-7.13

./build/server /path/to/Qwen3.6-35B-A3B-4bit \
    --host 127.0.0.1 --port 19200 --ctx-size 40000
```

- No `--use-mtp` (MTP/speculative decode stays off — the default).
- No `--no-think` (thinking enabled — the "normal" default).
- q4 (~20 GB) or q6 (~28 GB); gfx1151 is a unified-memory APU, so the MLX
  server coexists with other GPU users via GTT.

Health / smoke:

```bash
curl -s http://127.0.0.1:19200/health
curl -s http://127.0.0.1:19200/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"...","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32,"temperature":0}'
```

---

## 5. Stability testing

Large-context / long-generation regressions on this path show up as **decode
degeneration** (a token or short phrase repeating, e.g. `SAR/SAR/SAR/...`) or a
warmup/decode SIGSEGV. When validating a build:

- Run repeated long-generation requests (`max_tokens` ≥ 512) and check for any
  token repeated many times consecutively.
- Run repeated ~8–10k-token needle-recall prompts and confirm exact recall.
- Confirm `GET /health` still returns `{"status":"ok"}` after every run.

All of the above must pass with MTP disabled (default mode).
