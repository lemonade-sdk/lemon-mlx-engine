# NPU Unified Plane Backend Design

## Goal
Replace the current Python-IRON-JIT-subprocess NPU backend in `lemon-mlx-engine` with a direct C++ XRT backend that loads Chess-compiled xclbins and executes zero-copy GEMM via shared memory (Strix Halo UMA), transparently dispatching to format-specific xclbins (Q4NX, FP16, BitNet) based on weight registry metadata.

## Architecture

```
Model inference (50+ architectures)
  → linear_forward(x, w, bias) in quantized_linear.h
    → checks QuantizedWeightRegistry for format metadata
    → calls npu::quantized_matmul_from_mlx() ← NPU path
    → falls back to mx::quantized_matmul() if NPU fails

npu::quantized_matmul_from_mlx()  ← npu_backend.cpp
  → extracts raw pointers from mlx::core::array
  → calls npu::quantized_matmul()

npu::quantized_matmul()  ← npu_backend.cpp
  → auto-selects xclbin from format (bits, group_size, mode)
  → lazy-loads xclbin via npu_xclbin_cache
  → wraps MLX pointers as userptr BOs (SVM, zero-copy)
  → executes kernel
  → result is already in output pointer (no sync needed)

XRT C++ API (xrt::bo, xrt::kernel, xrt::hw_context)
  → amdxdna kernel driver
  → NPU hardware (Strix Halo RyzenAI-npu5)
```

## Key Properties

1. **Zero-copy dispatch**: All pointers wrapped as `xrt::bo(device, userptr, sz, XRT_BO_FLAGS_SVM, group)` — no DMA, NPU reads/writes shared memory directly
2. **Format-agnostic**: Single `npu::quantized_matmul()` dispatches to the right xclbin based on `QuantizationInfo.bits` + `group_size` + `mode`
3. **Lazy xclbin loading**: Each xclbin loaded once on first use, cached forever
4. **Fail-soft**: Returns `false` on error → caller falls back to CPU/GPU `mx::quantized_matmul()`
5. **Minimal changes to existing code**: Only `linear_forward()` in `quantized_linear.h` gets an `#ifdef MLX_BUILD_NPU` branch

## File Layout

### New files (in `src/npu/`)
- `npu_backend_impl.cpp` — main implementation of `npu::quantized_matmul()`, `npu::init()`, format dispatch
- `npu_xclbin_cache.cpp` — load xclbin, cache BOs/hw_context/kernel per format key
- `npu_gemm_q4nx.cpp` — Q4NX weight size calculation, layout preparation
- `npu_gemm_fp16.cpp` — FP16 weight size calculation, layout preparation
- `npu_gemm_bitnet.cpp` — BitNet weight size calculation, layout preparation
- `npu_kernel_runner.cpp` — BO creation, sync, kernel launch, wait

### Modified files
- `include/mlx-lm/npu/npu_backend.h` — add `quantized_matmul()`, `quantized_matmul_from_mlx()` signatures
- `include/mlx-lm/common/quantized_linear.h` — add `#ifdef MLX_BUILD_NPU` branch in `linear_forward()`
- `CMakeLists.txt` — find XRT, link `xrt++` to `mlx-lm-npu`
- `examples/test_npu.cpp` — rewrite to test full quantized_matmul path

### Requirements
- XRT headers: `/opt/xilinx/xrt/include/` or `/usr/include/` or `$XRT_INSTALL_DIR/include/`
- XRT libraries: `libxrt++.so` from system-wide `/usr/lib/x86_64-linux-gnu/` or toolchain
- xclbins directory: `$NPU_XCLBIN_DIR` (default: `${CMAKE_INSTALL_PREFIX}/lib/npu/xclbins/`)
- Format: `q4nx.xclbin` + `q4nx_instr.bin`, `fp16.xclbin` + `fp16_instr.bin`, `bitnet.xclbin` + `bitnet_instr.bin`

## API

```cpp
namespace npu {

// Initialize NPU device (called once at startup)
bool init();

// Main quantized GEMM dispatch — called from linear_forward via from_mlx helper
bool quantized_matmul(
    const void* x,           // BF16 activations (16 * M * K bytes)
    const void* w,           // Packed weights (format-specific)
    const float* scales,     // Per-group scale factors or nullptr
    const float* biases,     // Per-group zero-points or nullptr
    void* out,               // Output (16 * M * N bytes, BF16)
    int M, int K, int N,     // GEMM dimensions
    int group_size,          // 32 for Q4, 0 for FP16/BitNet
    int bits,                // 4 for Q4NX, 16 for FP16, 2 for BitNet
    const std::string& mode  // "affine" for Q4, "none" for FP16, "ternary" for BitNet
);

// Fallback: simple BF16 matmul (for non-quantized FP16 xclbin path)
bool matmul_bf16(
    const void* A, const void* B, void* C,
    int M, int K, int N
);

} // namespace npu
```

## XCLBIN Caching

```cpp
struct XCLBINCache {
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<uint32_t> instr_v;
};

// Key format: "q4nx" | "fp16" | "bitnet"
static std::unordered_map<std::string, std::shared_ptr<XCLBINCache>> s_cache;

shared_ptr<XCLBINCache> get_or_load(const string& fmt) {
    if (s_cache.count(fmt)) return s_cache[fmt];
    auto c = make_shared<XCLBINCache>();
    c->device = xrt::device(0);
    auto xclbin = xrt::xclbin(dir + "/" + fmt + ".xclbin");
    c->device.register_xclbin(xclbin);
    c->context = xrt::hw_context(c->device, xclbin.get_uuid());
    c->kernel = xrt::kernel(c->context, "matmul_vectorized_bfp16");
    c->instr_v = load_instr_binary(dir + "/" + fmt + "_instr.bin");
    s_cache[fmt] = c;
    return c;
}
```

## Integration into linear_forward()

In `quantized_linear.h`, the NPU path is:

```cpp
if (qi) {
#ifdef MLX_BUILD_NPU
    auto result = npu::quantized_matmul_from_mlx(input, w, bias, *qi);
    if (result.has_value()) return *result;
#endif
    // Fallback: existing mx::quantized_matmul path
}
```

The `from_mlx` helper extracts raw pointers from MLX arrays, calls `npu::quantized_matmul()`, and returns the resulting array.

## Weight Format Detection

Format is determined solely from `QuantizationInfo`:
- `bits == 4 && group_size ∈ {32, 128}` → Q4NX xclbin
- `bits == 16` → FP16 xclbin (scales/biases ignored)
- `bits == 2 && mode == "ternary"` → BitNet xclbin
- Anything else → `return false` (fallback to MX)

Weight buffer sizes per format:
- Q4NX: `(K / 2) * N * 4` bytes (2 int4 per byte, NX-layout shuffled)
- FP16: `K * N * 2` bytes (raw BF16, no layout shuffle)
- BitNet: K * N / 4 bytes (2 bits per weight, ternary { -1, 0, +1 } packed)

## CMake Integration

```cmake
if(MLX_LM_BUILD_NPU)
    # Find XRT
    find_library(XRT_LIB xrt++ 
        PATHS /usr/lib/x86_64-linux-gnu $ENV{XRT_INSTALL_DIR}/lib)
    find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
        PATHS /usr/include $ENV{XRT_INSTALL_DIR}/include)
    
    add_library(mlx-lm-npu STATIC
        src/npu/npu_backend_impl.cpp
        src/npu/npu_xclbin_cache.cpp
        src/npu/npu_kernel_runner.cpp
        src/npu/npu_gemm_q4nx.cpp
        src/npu/npu_gemm_fp16.cpp
        src/npu/npu_gemm_bitnet.cpp
    )
    target_include_directories(mlx-lm-npu PUBLIC
        ${XRT_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(mlx-lm-npu PUBLIC ${XRT_LIB})
endif()
```

## Current Files to Keep

- `src/npu/kernels/npu_gemm.cc` — unmodified (Peano fallback reference)
- `src/npu/npu_jit.py` — kept for debug/fallback
- `src/npu/npu_backend.cpp` — replaced by `npu_backend_impl.cpp`; old file can be renamed or removed after migration
