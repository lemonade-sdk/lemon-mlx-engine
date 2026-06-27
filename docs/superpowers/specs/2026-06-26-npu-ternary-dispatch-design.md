# NPU Dispatch for 1-bit Ternary Matrix Multiplication

## Status: Design Proposal

## Overview

Add NPU (AI Engine) dispatch for 1-bit ternary matmuls in the lemon-mlx-engine.
Instead of running ternary × fp16 matmuls as full fp16 operations on the GPU
(which is 16× more memory bandwidth than necessary), dispatch them to the
AMD XDNA NPU where the ternary sparsity can be exploited directly.

## System Context

### Hardware

- **APU**: AMD Ryzen AI MAX+ 395 (Strix Halo)
- **GPU**: Radeon 8060S (gfx1151) — unified memory with CPU
- **NPU**: RyzenAI-npu5 at PCI BDF 0000:c6:00.1 — AIE2 array with up to ~31 TFLOPS
- **Memory**: Unified LPDDR5X (CPU/GPU/NPU share via PCIe)

### Available Toolchain

| Component | Status |
|-----------|--------|
| `clang++ --target=aie2-none-unknown-elf` | ✅ Available (Ubuntu clang 21.1.8) |
| `Xilinx/aie_api` headers | ✅ Cloned to /tmp/aie_api |
| XRT C++ headers | ✅ /home/bcloud/torch2aie/toolchain/xrt/include/xrt/ |
| XRT runtime (libxrt++.so) | ✅ Installed |
| Python pyxrt bindings | ✅ Installed |
| `aie.iron` (IRON JIT) | ❌ NOT available |

### Current NPU Backend

The existing `src/npu/npu_backend.cpp` uses a Python subprocess to run
IRON JIT compilation — but `aie.iron` isn't installed. This rewrite
replaces it with a direct C++ XRT path.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  MLX Engine                              │
│  ┌────────────────────────────────────────────────────┐  │
│  │            npu::backend (C++ XRT)                  │  │
│  │  ┌──────────────┐  ┌─────────────┐                │  │
│  │  │ kernel_pool  │  │ buffer_mgr  │                │  │
│  │  └──────┬───────┘  └──────┬──────┘                │  │
│  │         │                 │                        │  │
│  │  ┌──────┴─────────────────┴──────────────────────┐ │  │
│  │  │              XRT Runtime                      │ │  │
│  │  │  load_xclbin → open_context → create_bo → run │ │  │
│  │  └──────────────────────┬────────────────────────┘ │  │
│  └─────────────────────────┼──────────────────────────┘  │
│                            │ PCIe                        │
│  ┌─────────────────────────┼──────────────────────────┐  │
│  │        AMD XDNA NPU    │                          │  │
│  │  ┌──────────────────────┴──────────────────────┐   │  │
│  │  │         AI Engine Array (AIE2)              │   │  │
│  │  │  ┌─────┐ ┌─────┐ ┌─────┐    ┌─────┐        │   │  │
│  │  │  │Tile0│ │Tile1│ │Tile2│ ...│TileN│        │   │  │
│  │  │  │GEMM │ │GEMM │ │GEMM │    │GEMM │        │   │  │
│  │  │  └─────┘ └─────┘ └─────┘    └─────┘        │   │  │
│  │  └─────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: AIE Kernel — ternary_gemv_aie

Write vectorized GEMV kernel consuming ternary weights packed as 2-bit codes
and producing fp16 output. Each AIE tile computes 16 output rows in parallel.

**Kernel signature:**
```
ternary_gemv_aie(
    const uint8_t* packed_weights,  // [ceil(N/4), K] U8 packed ternary
    const bfloat16* activations,    // [K] bf16 activations  
    bfloat16* output,               // [N] bf16 output
    int N,                          // output rows
    int K                           // input dimension
)
```

**AIE math**: For each output row `oc` with packed 2-bit codes `c[i]`:
```
result[oc] = Σ_i (c[i] - 1) * act[i] * scale[oc]
```
where `c[i] ∈ {0,1,2}`, mapped to ternary `{-1,0,+1}`.

**Tile tiling**: With `M=16` output rows per tile and `K=256` elements per vector pass,
each tile processes a 16×256 chunk. Multiple tiles run in parallel across output rows.

### Phase 2: C++ XRT Backend

Replace the Python subprocess in `src/npu/npu_backend.cpp` with:

1. **Kernel compilation at build time** — Use `clang++ --target=aie2` to compile
   the AIE kernel to an ELF during CMake build.

2. **XCLBIN packaging** — Package the compiled ELF into an XCLBIN using
   `pyxrt` or `xclbinutil`.

3. **XRT runtime integration**:
   - `npu::init()` → Load XCLBIN, open AIE context
   - `npu::matmul()` → Create buffer objects, run kernel, sync
   - `npu::ternary_gemv()` → Specialized entry for ternary matmuls

### Phase 3: MLX Integration

Add an NPU compute primitive to the MLX engine:

1. In `linear_forward()` in `quantized_linear.h`, check if NPU is available
   for ternary weights. If so, dispatch to NPU instead of GPU.

2. Add a configuration flag `--npu` to enable NPU dispatch.

3. Fall back to GPU (quantized_matmul) when NPU unavailable or for
   non-ternary weights.

## Build Integration

Add to CMakeLists.txt:

```cmake
option(MLX_LM_BUILD_NPU "Build NPU backend" ON)

if(MLX_LM_BUILD_NPU)
    # Find XRT
    find_library(XRT_LIB xrt++)

    # Compile AIE kernel
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/kernels/ternary_gemv_aie.o
        COMMAND clang++ --target=aie2-none-unknown-elf
            -I/tmp/aie_api/include
            -std=c++2b -O2
            -c ${CMAKE_SOURCE_DIR}/src/npu/kernels/ternary_gemv_aie.cpp
            -o ${CMAKE_BINARY_DIR}/kernels/ternary_gemv_aie.o
    )
endif()
```

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/npu/kernels/ternary_gemv_aie.cpp` | **NEW** — AIE kernel for ternary GEMV |
| `src/npu/npu_backend.cpp` | Rewrite — native C++ XRT path |
| `include/mlx-lm/npu/npu_backend.h` | Add ternary_gemv API |
| `src/npu/npu_jit.py` | Remove (replaced by C++ XRT) |
| `CMakeLists.txt` | Add AIE kernel build step |
| `include/mlx-lm/common/quantized_linear.h` | Add NPU dispatch in linear_forward |

## Success Criteria

1. Qwen3-8B-BitNet runs with NPU dispatch enabled
2. Ternary matmul executes on NPU (verified via XRT profiling)
3. Speedup over full fp16 matmul on GPU
4. All existing tests pass

## Open Questions

1. How many AIE tiles are available on NPU5 in Strix Halo?
2. What is the data transfer latency between unified memory and NPU?
3. Can the NPU share the unified LPDDR5X memory directly (no PCIe copy)?
4. Does the NPU support bfloat16 natively, or only int8?

