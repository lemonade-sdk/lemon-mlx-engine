# E3 — MLX HSACO / kernel-object inventory (hot op: quantized matmul)

**Date:** 2026-08-02  
**Branch:** `exp/redline-kernel-launch`  
**Host context:** gfx1150 · MLX FetchContent under `build/_deps/mlx-src` · build tree `build/_deps/mlx-build`  
**Hot op chosen:** **quantized matmul / qmv / gather_qmm** (`qmm.hip`) — dominant decode launch+compute family per [`SUBAGENT_ENGINE_LAUNCH_MAP.md`](SUBAGENT_ENGINE_LAUNCH_MAP.md).

**Verdict (one line):**  
- **Drop-in export of product qmm into Redline `Executable::load`:** **NOT ready / not feasible without major tooling.**  
- **On-disk HSACO for *some* MLX kernels (JIT fused elementwise):** **YES** under `/tmp/mlx/…/hsaco/gfx1150/`.  
- **qmm device ISA exists** inside AOT object as clang offload bundle (not a standalone Redline-friendly CO).

**Not claimed:** gen t/s, successful Redline load of qmm, product wire.

---

## 1. Two MLX ROCm code paths

| Path | What | How launched | On-disk CO? | Redline fit |
|------|------|--------------|-------------|-------------|
| **AOT HIP** (`quantized/qmm.hip`, …) | Template `__global__` kernels compiled into `libmlx` | **`hipLaunchKernel(func*, …)`** via function pointer (`add_kernel_node`) | No standalone `.hsaco`; **embedded offload bundle** in `.o` | Poor as-is (pointer ABI ≠ module symbol) |
| **JIT hiprtc** (`jit_module.cpp` / `compiled.cpp`) | Fused elementwise / compiled graphs | **`hipModuleLoadData` + `hipModuleLaunchKernel`** | **Yes** — `.hsaco` + `.txt` + `.hip` cache | Good *format* match; wrong *op class* for hot qmm |

Product decode default: **eager** launches (`use_hip_graphs() == false`) — each primitive pays host launch ([engine map](SUBAGENT_ENGINE_LAUNCH_MAP.md)).

---

## 2. Hot op: `qmm.hip` (AOT)

### Source / build

| Item | Path / fact |
|------|-------------|
| Source | [`build/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip`](../../../build/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip) (~275KB) |
| CMake list | `mlx/backend/rocm/CMakeLists.txt` includes `quantized/qmm.hip` in `HIP_SOURCES`; compile with `--offload-arch=…`, **`-fno-gpu-rdc`** (no separate device link) |
| Host object | `build/_deps/mlx-build/mlx/backend/rocm/hip_objs/quantized/qmm.o` (~6.5MB) |
| Archive | Linked into `build/_deps/mlx-build/libmlx.a` |

### Launch pattern (file cite)

Dense WMMA path records a **C++ kernel pointer**, not a module symbol:

```3370:3379:build/_deps/mlx-src/mlx/backend/rocm/quantized/qmm.hip
        enc.add_kernel_node(
            &rocm::qmm_wmma_dense_kernel<hip_bfloat16, hip_bfloat16, BITS, 64, HB>,
            grid, block, 0u,
            gpu_ptr<const hip_bfloat16>(x),
            gpu_ptr<const uint32_t>(w),
            gpu_ptr<const hip_bfloat16>(scales),
            ...
            M, N, K);
```

Eager path ends in:

```565:575:build/_deps/mlx-src/mlx/backend/rocm/device.cpp
void CommandEncoder::add_kernel_node_raw(...) {
  if (!use_hip_graphs()) {
    ...
    hipLaunchKernel(func, grid_dim, block_dim, params, smem_bytes, stream_);
```

Many other decode shapes use the same pattern: `qmv_warp_shared_kernel`, `gather_qmv_*`, etc. (strings in `qmm.o` show large template forest: bits × group × dtype × bias).

### Device image presence (inventory fact)

In `qmm.o` binary search (this host, 2026-08-02):

| Magic / string | Offset in `qmm.o` |
|----------------|-------------------|
| `__CLANG_OFFLOAD_BUNDLE__` | present (~byte 659456) |
| `amdgcn-amd-amdhsa` | present |
| `.hip_fatbin` name fragment | present |

So **device code is embedded** in the AOT object (clang offload bundle), but **MLX never writes a Redline-style standalone `floor_k.co` for qmm**. Extracting would need `clang-offload-bundler` (or equivalent) + symbol/kernarg reverse engineering — **not a one-file handoff**.

### Why Redline C ABI is a poor first consumer

Redline’s proven E1/E2 path:

```text
HSACO file → Executable::load → kernel("name.kd") → kernarg pool → SingleQueueBatchGraph
```

MLX qmm product path:

```text
&templated_kernel_fn → hipLaunchKernel → stack kernelParams[]
```

Gaps:

1. **No public module symbol table** in the launch path (host stub + fatbin, not `hipModuleGetFunction`).  
2. **Huge specialization space** (dtype, bits, group size, bias, gather/MoE variants).  
3. **Kernarg layout** is C++ parameter list, not an exported metadata contract.  
4. **MoE `gather_qmm`** is data-dependent expert routing — fights single retained IB unless multipath/re-record (RESEARCH already flags this).

**Feasibility label for product qmm → Redline load:** **NOT FEASIBLE as drop-in.**  
**Feasibility for *research extract* of one template from offload bundle:** **POSSIBLE with tooling** (unproven this fire — no unbundle attempt).

---

## 3. JIT HSACO path (feasible format, not hot qmm)

### Mechanism (file cites)

| Step | Location |
|------|----------|
| Cache dir | `jit_module.cpp` `hsaco_cache_dir()` → `MLX_HSACO_CACHE_DIR/<arch>` or **`/tmp/mlx/<mlx_version>/hsaco/<arch>`** |
| Write | `write_cached_hsaco` → `.hsaco` binary + `.txt` (logical name ↔ mangled) + `.hip` source |
| Compile | `hiprtcCompileProgram` / `hiprtcGetCode` |
| Load | `hipModuleLoadData(&module_, hsaco.data())` then `hipModuleGetFunction` |
| Launch | `launch_module_kernel` → `hipModuleLaunchKernel` when graphs OFF |

### Observed on this host

| Item | Value |
|------|--------|
| Cache | `/tmp/mlx/0.32.0/hsaco/gfx1150/` |
| Count | **10** `*.hsaco` files (elementwise/fused names: Sigmoid/Broadcast/Multiply/… — not qmm) |
| Format sample | ELF **AMD GPU** (`file(1)`: `ELF 64-bit LSB shared object, AMD GPU architecture version 1`) |
| Name map | Companion `.txt` lists demangled + mangled symbols (e.g. `…_contiguous<uint32_t, 1>` → `_ZN3mlx4core4rocm…`) |

These files are **already Redline-shaped** (loadable code objects with known symbols). They are **not** the dominant qmm compute path.

---

## 4. Engine-side custom kernels

| Item | Status |
|------|--------|
| `src/**/*.hip` | Essentially none for decode matmul |
| `src/common/mtp_delta_kernel.cpp` | Present; not a qmm substitute |

Engine does **not** own a separate HSACO pipeline for the hot op; it relies on MLX ROCm backend.

---

## 5. Feasibility matrix (E3 deliverable)

| Approach | Feasible? | Effort | Notes |
|----------|-----------|--------|-------|
| **A.** Feed JIT cache `.hsaco` into Redline for a fused elementwise smoke | **Yes** | Low | Proves MLX CO ↔ Redline load; **not** decode win |
| **B.** Unbundle `qmm.o` offload image → pick one `qmv_*` symbol → Redline | **Maybe** | High | Needs bundler + kernarg ABI work; unproven |
| **C.** Recompile a **slice** of `qmm.hip` / toy qmv with `hipcc --genco --offload-arch=gfx1150` for fixed shape | **Yes** | Med | Duplicates source; full control of symbol/kernarg |
| **D.** Interpose `CommandEncoder::add_kernel_node_raw` / launch (no HSACO export) | **Design** | Med–High | E4 hook class; stays in-process |
| **E.** Assume product qmm drops into Redline without work | **No** | — | Contradicted by pointer launch + templates |

**E3 board answer:** For **one hot op (qmm)**, standalone HSACO export for Redline is **not currently feasible as a clean product path**; **format-feasible** objects exist only on the **JIT** side; **practical next integration** is **C or D**, not “copy qmm.hsaco.”

---

## 6. Implications for E4 (`MLX_REDLINE_DECODE`)

Prefer design that **does not** require full qmm HSACO export:

1. **Fixed T=1 subgraph** of *launch-heavy small ops* (where E1/E2 floor win lives) via Redline retained AQL, leave large qmm on HIP — or  
2. **In-process** record of launches (encoder shim) without external CO — or  
3. **Explicit recompiled** expert/qmv subset (path C) behind env OFF.

Stable buffers (`graph_decode_*`) remain useful for kernarg patching either way.

---

## 7. Inventory checklist (done)

| Check | Result |
|-------|--------|
| Hot op identified | qmm / qmv / gather_qmm |
| JIT vs AOT | Both present; hot path AOT pointer launch |
| Disk HSACO | JIT yes (`/tmp/mlx/0.32.0/hsaco/gfx1150`); qmm no standalone |
| Offload bundle in qmm.o | Yes (`__CLANG_OFFLOAD_BUNDLE__`, `amdgcn-amd-amdhsa`) |
| Symbol names for Redline | JIT `.txt` yes; qmm launch path no module names |
| Feasible/not | **Not drop-in; JIT/recompile/shim are the real doors** |
