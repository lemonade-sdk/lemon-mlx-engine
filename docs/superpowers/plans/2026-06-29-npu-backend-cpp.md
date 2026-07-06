# NPU Unified Plane Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python-IRON-JIT-subprocess NPU backend with a direct C++ XRT backend that zero-copy dispatches GEMM to format-specific Chess xclbins.

**Architecture:** The NPU backend stays as a static library (`mlx-lm-npu`) linked into server/chat/test_npu. `linear_forward()` in `quantized_linear.h` calls `npu::quantized_matmul_from_mlx()` which extracts raw pointers from MLX arrays, wraps them as XRT userptr BOs (zero-copy via SVM), and executes the correct xclbin. Falls back to `mx::quantized_matmul` if NPU returns false. Three xclbins (Q4NX, FP16, BitNet) are lazy-loaded and cached.

**Tech Stack:** C++20, XRT 2.21.75 (system) / 2.23.0 (toolchain), MLX C++ arrays, Strix Halo NPU (amdxdna kernel driver)

## Global Constraints

- All MLX array data pointers are in shared memory (UMA on Strix Halo) — no DMA copies needed
- `xrt::bo` with `XRT_BO_FLAGS_SVM` wraps any user pointer as zero-copy
- XRT headers at `torch2aie/ (local toolchain)toolchain/xrt/include/` or `/usr/include/xrt/`
- XRT libs at `torch2aie/ (local toolchain)toolchain/xrt/lib64/` or `/usr/lib/x86_64-linux-gnu/`
- `MLX_BUILD_NPU` compile definition gates NPU code paths
- The `quantized_linear.h` changes must be minimal — only an `#ifdef` branch in `linear_forward()`
- Existing `npu_backend.cpp` and `npu_jit.py` are kept as fallback — don't delete them

---

### Task 1: Rewrite `npu_backend.h` with New API

**Files:**
- Create: `include/mlx-lm/npu/npu_backend.h` (overwrite existing)
- Modify: none

**Interfaces:**
- Produces: `npu::init()`, `npu::quantized_matmul()`, `npu::matmul_bf16()`, `npu::device_name()`, `npu::peak_tflops()`, `npu::is_available()`

- [ ] **Step 1: Write the new header**

```cpp
// Copyright © 2025-2026 — NPU Unified Plane Backend
// Direct C++ XRT backend for AMD XDNA NPU — replaces Python IRON JIT.
// Zero-copy via userptr BOs (SVM) on Strix Halo UMA.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace npu {

/// Initialize NPU device. Called once at startup.
/// Returns true if NPU is available.
bool init();

/// Check if NPU is initialized.
bool is_available();

/// Get NPU device name (e.g. "RyzenAI-npu5").
const char* device_name();

/// Get peak TFLOPS of detected NPU.
float peak_tflops();

/// Format-aware quantized GEMM dispatch.
///
/// Auto-selects the correct xclbin (Q4NX, FP16, or BitNet) based on
/// bits/group_size/mode, wraps all pointers as userptr XRT BOs (zero-copy),
/// and executes on NPU. Returns true on success, false on failure (caller
/// falls back to mx::quantized_matmul).
///
/// @param x       BF16 activations, layout [M, K], 2 bytes per element
/// @param w       Packed weights, format-specific:
///                Q4NX:   (K/2) * N * 4 bytes (2 int4 per byte, NX layout)
///                FP16:   K * N * 2 bytes (raw BF16)
///                BitNet: K * N / 4 bytes (2 bits per weight, ternary)
/// @param scales  Per-group scale factors (float), or nullptr for FP16/BitNet
/// @param biases  Per-group zero-points (float), or nullptr for FP16/BitNet
/// @param out     Output buffer, BF16, layout [M, N], 2 bytes per element
/// @param M       Number of rows in A and C
/// @param K       Reduction dimension (columns of A, rows of B)
/// @param N       Number of columns in B and C
/// @param group_size  Group size for quantization (32 for Q4, 0 otherwise)
/// @param bits    Bit width (4 for Q4NX, 16 for FP16, 2 for BitNet)
/// @param mode    Quantization mode ("affine", "none", "ternary")
/// @return true if NPU execution succeeded, false if fallback needed
bool quantized_matmul(
    const void* x,
    const void* w,
    const float* scales,
    const float* biases,
    void* out,
    int M, int K, int N,
    int group_size,
    int bits,
    const std::string& mode);

/// Simple BF16/BF16 matmul (non-quantized path).
/// Wraps pointers as userptr BOs and runs the FP16 xclbin.
bool matmul_bf16(
    const void* A, const void* B, void* C,
    int M, int K, int N);

} // namespace npu
```

- [ ] **Step 2: Verify file written**

Run: `cat include/mlx-lm/npu/npu_backend.h | head -5`
Expected: `#pragma once`, `namespace npu {`

- [ ] **Step 3: Commit**

```bash
git add include/mlx-lm/npu/npu_backend.h
git commit -m "feat: rewrite npu_backend.h with format-aware quantized_matmul API"
```

---

### Task 2: Create XCLBIN Cache Module

**Files:**
- Create: `src/npu/npu_xclbin_cache.h`
- Create: `src/npu/npu_xclbin_cache.cpp`

**Interfaces:**
- Produces: `npu::detail::XCLBINCache`, `npu::detail::get_or_load_xclbin(format_key)`, `npu::detail::instr_cache_path()`, `npu::detail::xclbin_dir()`

- [ ] **Step 1: Write the header**

```cpp
// XCLBIN cache — lazy-load and cache xclbins per format key.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

namespace npu {
namespace detail {

struct XCLBINCache {
    xrt::device device;
    xrt::hw_context context;
    xrt::kernel kernel;
    std::vector<uint32_t> instr_v;
};

/// Return the xclbin directory (default: $NPU_XCLBIN_DIR or
/// /usr/local/lib/npu/xclbins/).
std::string xclbin_dir();

/// Return the instruction binary path for a given format name.
std::string instr_bin_path(const std::string& format);

/// Load an instruction binary from disk into a uint32 vector.
std::vector<uint32_t> load_instr_binary(const std::string& path);

/// Get or load the xclbin for a given format key.
/// Keys: "q4nx", "fp16", "bitnet"
std::shared_ptr<XCLBINCache> get_or_load_xclbin(const std::string& format);

/// Determine format key from quantization parameters.
std::string format_key(int bits, int group_size, const std::string& mode);

} // namespace detail
} // namespace npu
```

- [ ] **Step 2: Write the implementation**

```cpp
#include "npu_xclbin_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

namespace npu {
namespace detail {

static std::unordered_map<std::string, std::shared_ptr<XCLBINCache>> s_cache;

std::string xclbin_dir() {
    const char* env = std::getenv("NPU_XCLBIN_DIR");
    if (env) return env;
    for (const auto& path : {
        "/usr/local/lib/npu/xclbins/",
        "/opt/npu/xclbins/",
        "./xclbins/"
    }) {
        if (std::ifstream(path + "q4nx.xclbin").good())
            return path;
    }
    return "/usr/local/lib/npu/xclbins/";
}

std::string instr_bin_path(const std::string& format) {
    return xclbin_dir() + "/" + format + "_instr.bin";
}

std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[NPU] Failed to open instruction binary: %s\n",
                     path.c_str());
        return {};
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> data(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

std::string format_key(int bits, int group_size, const std::string& mode) {
    if (bits == 4) return "q4nx";
    if (bits == 16) return "fp16";
    if (bits == 2 && mode == "ternary") return "bitnet";
    return "unknown";
}

std::shared_ptr<XCLBINCache> get_or_load_xclbin(const std::string& format) {
    auto it = s_cache.find(format);
    if (it != s_cache.end()) return it->second;

    auto cache = std::make_shared<XCLBINCache>();

    cache->device = xrt::device(0);

    std::string xclbin_path = xclbin_dir() + "/" + format + ".xclbin";
    std::printf("[NPU] Loading xclbin: %s\n", xclbin_path.c_str());

    auto xclbin = xrt::xclbin(xclbin_path);
    cache->device.register_xclbin(xclbin);
    cache->context = xrt::hw_context(cache->device, xclbin.get_uuid());

    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                 [](const xrt::xclbin::kernel& k) {
                                     auto name = k.get_name();
                                     return name.rfind("matmul_vectorized_bfp16", 0) == 0;
                                 });
    auto kernelName = xkernel.get_name();
    cache->kernel = xrt::kernel(cache->context, kernelName);

    cache->instr_v = load_instr_binary(instr_bin_path(format));
    if (cache->instr_v.empty()) {
        std::fprintf(stderr, "[NPU] Warning: empty instruction binary for %s\n",
                     format.c_str());
    }

    s_cache[format] = cache;
    std::printf("[NPU] Loaded xclbin: %s (kernel=%s, instr_size=%zu)\n",
                format.c_str(), kernelName.c_str(), cache->instr_v.size());
    return cache;
}

} // namespace detail
} // namespace npu
```

- [ ] **Step 3: Commit**

```bash
git add src/npu/npu_xclbin_cache.h src/npu/npu_xclbin_cache.cpp
git commit -m "feat: add XCLBIN lazy-loading cache module"
```

---

### Task 3: Create Kernel Runner Module

**Files:**
- Create: `src/npu/npu_kernel_runner.h`
- Create: `src/npu/npu_kernel_runner.cpp`

**Interfaces:**
- Produces: `npu::detail::run_kernel(cache, a_ptr, a_sz, b_ptr, b_sz, c_ptr, c_sz, instr_bo_ret)` — wraps BOs, executes, returns true/false

- [ ] **Step 1: Write the header**

```cpp
// Kernel runner — wraps user pointers as XRT BOs and executes kernel.
#pragma once

#include <cstddef>
#include <memory>

#include "npu_xclbin_cache.h"

namespace npu {
namespace detail {

/// Run the GEMM kernel on NPU with zero-copy userptr BOs.
///
/// All pointers must be in shared memory (UMA) — XRT_BO_FLAGS_SVM enables
/// the NPU to read/write them directly without DMA copies.
///
/// @param cache    Loaded xclbin cache entry
/// @param a_ptr    Input A buffer (activations, BF16)
/// @param a_sz     Size of A buffer in bytes
/// @param b_ptr    Input B buffer (weights, format-specific)
/// @param b_sz     Size of B buffer in bytes
/// @param c_ptr    Output C buffer (results, BF16)
/// @param c_sz     Size of C buffer in bytes
/// @return true if kernel completed successfully
bool run_kernel(
    const std::shared_ptr<XCLBINCache>& cache,
    void* a_ptr, size_t a_sz,
    void* b_ptr, size_t b_sz,
    void* c_ptr, size_t c_sz);

} // namespace detail
} // namespace npu
```

- [ ] **Step 2: Write the implementation**

```cpp
#include "npu_kernel_runner.h"
#include "xrt/xrt_bo.h"
#include <cstdio>

namespace npu {
namespace detail {

bool run_kernel(
    const std::shared_ptr<XCLBINCache>& cache,
    void* a_ptr, size_t a_sz,
    void* b_ptr, size_t b_sz,
    void* c_ptr, size_t c_sz)
{
    auto& device = cache->device;
    auto& kernel = cache->kernel;

    // userptr BOs with SVM flag = NPU accesses memory directly, zero-copy
    auto bo_instr = xrt::bo(device, cache->instr_v.data(),
                            cache->instr_v.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE,
                            kernel.group_id(1));

    auto bo_a = xrt::bo(device, a_ptr, a_sz,
                        XRT_BO_FLAGS_SVM | XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));

    auto bo_b = xrt::bo(device, b_ptr, b_sz,
                        XRT_BO_FLAGS_SVM | XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

    auto bo_c = xrt::bo(device, c_ptr, c_sz,
                        XRT_BO_FLAGS_SVM | XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

    // Only instruction BO needs sync (small, ~4KB)
    // Data BOs (userptr+SVM) are already visible to NPU
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr,
                      static_cast<int>(cache->instr_v.size()),
                      bo_a, bo_b, bo_c);

    auto status = run.wait();
    if (status != ERT_CMD_STATE_COMPLETED) {
        std::fprintf(stderr, "[NPU] Kernel failed: status=%d\n",
                     static_cast<int>(status));
        return false;
    }

    // Result is already in c_ptr — userptr BO writes directly to original pointer
    return true;
}

} // namespace detail
} // namespace npu
```

- [ ] **Step 3: Commit**

```bash
git add src/npu/npu_kernel_runner.h src/npu/npu_kernel_runner.cpp
git commit -m "feat: add zero-copy kernel runner with userptr BOs"
```

---

### Task 4: Create Format-Specific Weight Size Helpers

**Files:**
- Create: `src/npu/npu_gemm_q4nx.h`
- Create: `src/npu/npu_gemm_fp16.h`
- Create: `src/npu/npu_gemm_bitnet.h`

**Interfaces:**
- Produces: `npu::detail::q4nx_weight_size(K,N)`, `npu::detail::fp16_weight_size(K,N)`, `npu::detail::bitnet_weight_size(K,N)`

- [ ] **Step 1: Write Q4NX helpers**

```cpp
// src/npu/npu_gemm_q4nx.h
// Q4NX format weight and buffer size calculations.
#pragma once
#include <cstddef>

namespace npu {
namespace detail {

/// Compute byte size of Q4NX-packed weights.
/// Q4NX packs 2 int4 per byte with NX layout shuffle.
inline size_t q4nx_weight_size(int K, int N) {
    return static_cast<size_t>(K / 2) * static_cast<size_t>(N) * 4;
}

inline size_t q4nx_output_size(int M, int N) {
    return static_cast<size_t>(M) * static_cast<size_t>(N) * 2; // BF16
}

} // namespace detail
} // namespace npu
```

- [ ] **Step 2: Write FP16 helpers**

```cpp
// src/npu/npu_gemm_fp16.h
#pragma once
#include <cstddef>

namespace npu {
namespace detail {

inline size_t fp16_weight_size(int K, int N) {
    return static_cast<size_t>(K) * static_cast<size_t>(N) * 2; // BF16
}

inline size_t fp16_output_size(int M, int N) {
    return static_cast<size_t>(M) * static_cast<size_t>(N) * 2;
}

} // namespace detail
} // namespace npu
```

- [ ] **Step 3: Write BitNet helpers**

```cpp
// src/npu/npu_gemm_bitnet.h
#pragma once
#include <cstddef>

namespace npu {
namespace detail {

/// Byte size of BitNet ternary weights (2 bits per weight, ternary {-1,0,+1}).
inline size_t bitnet_weight_size(int K, int N) {
    return static_cast<size_t>(K) * static_cast<size_t>(N) / 4;
}

inline size_t bitnet_output_size(int M, int N) {
    return static_cast<size_t>(M) * static_cast<size_t>(N) * 2;
}

} // namespace detail
} // namespace npu
```

- [ ] **Step 4: Commit**

```bash
git add src/npu/npu_gemm_q4nx.h src/npu/npu_gemm_fp16.h src/npu/npu_gemm_bitnet.h
git commit -m "feat: add format-specific weight size helpers for Q4NX/FP16/BitNet"
```

---

### Task 5: Rewrite `npu_backend.cpp` as Main Backend Impl + Update CMakeLists.txt

**Files:**
- Modify: `src/npu/npu_backend.cpp` (replace content with new implementation)
- Modify: `CMakeLists.txt` (add new source files and XRT linking)

**Interfaces:**
- Consumes: `npu::detail::get_or_load_xclbin()`, `npu::detail::run_kernel()`, weight size helpers
- Produces: `npu::init()`, `npu::quantized_matmul()`, `npu::matmul_bf16()`

- [ ] **Step 1: Replace `src/npu/npu_backend.cpp` content**

```cpp
// NPU backend — format dispatch, xclbin selection, zero-copy kernel execution.
// Replaces the old Python-IRON-JIT subprocess implementation.
#include "mlx-lm/npu/npu_backend.h"

#include "npu_kernel_runner.h"
#include "npu_xclbin_cache.h"
#include "npu_gemm_q4nx.h"
#include "npu_gemm_fp16.h"
#include "npu_gemm_bitnet.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace npu {

namespace {

struct NPUState {
    bool initialized = false;
    std::string name;
    float peak_tflops = 0.0f;
};

NPUState& state() {
    static NPUState s;
    return s;
}

bool detect_device() {
    try {
        xrt::device dev(0);
        auto bdf = dev.get_info<xrt::info::device::bdf>();
        auto name = dev.get_info<xrt::info::device::name>();
        state().name = name;

        if (name.find("npu5") != std::string::npos)
            state().peak_tflops = 31.2f;
        else if (name.find("npu4") != std::string::npos)
            state().peak_tflops = 23.0f;
        else if (name.find("npu3") != std::string::npos)
            state().peak_tflops = 16.0f;
        else
            state().peak_tflops = 10.0f;

        std::printf("[NPU] Detected: %s (BDF=%s, %.1f TFLOPS peak)\n",
                    name.c_str(), bdf.c_str(), state().peak_tflops);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[NPU] Device detection failed: %s\n", e.what());
        return false;
    }
}

} // anonymous namespace

bool init() {
    if (state().initialized) return true;
    state().initialized = detect_device();
    return state().initialized;
}

bool is_available() { return state().initialized; }
const char* device_name() { return state().name.c_str(); }
float peak_tflops() { return state().peak_tflops; }

bool quantized_matmul(
    const void* x, const void* w,
    const float* scales, const float* biases,
    void* out,
    int M, int K, int N,
    int group_size, int bits,
    const std::string& mode)
{
    if (!state().initialized) {
        std::fprintf(stderr, "[NPU] Not initialized\n");
        return false;
    }

    std::string fmt = detail::format_key(bits, group_size, mode);
    if (fmt == "unknown") {
        std::fprintf(stderr, "[NPU] Unknown format: bits=%d group=%d mode=%s\n",
                     bits, group_size, mode.c_str());
        return false;
    }

    auto cache = detail::get_or_load_xclbin(fmt);
    if (!cache || cache->instr_v.empty()) {
        std::fprintf(stderr, "[NPU] No xclbin loaded for format: %s\n", fmt.c_str());
        return false;
    }

    size_t a_sz = static_cast<size_t>(M) * static_cast<size_t>(K) * 2;
    size_t b_sz = 0;
    size_t c_sz = static_cast<size_t>(M) * static_cast<size_t>(N) * 2;

    if (bits == 4)       b_sz = detail::q4nx_weight_size(K, N);
    else if (bits == 16) b_sz = detail::fp16_weight_size(K, N);
    else if (bits == 2 && mode == "ternary")
                         b_sz = detail::bitnet_weight_size(K, N);
    else return false;

    return detail::run_kernel(
        cache,
        const_cast<void*>(x), a_sz,
        const_cast<void*>(w), b_sz,
        out, c_sz);
}

bool matmul_bf16(const void* A, const void* B, void* C,
                 int M, int K, int N) {
    return quantized_matmul(A, B, nullptr, nullptr, C,
                            M, K, N, 0, 16, "none");
}

} // namespace npu
```

- [ ] **Step 2: Update CMakeLists.txt NPU section**

Replace the NPU backend section in `CMakeLists.txt` (currently lines ~279-304) with:

```cmake
# NPU backend (requires XRT)
if(MLX_LM_BUILD_NPU)
    # Find XRT
    find_library(XRT_LIB xrt++
        PATHS /usr/lib/x86_64-linux-gnu
              torch2aie/ (local toolchain)toolchain/xrt/lib64
              $ENV{XRT_INSTALL_DIR}/lib
        NO_DEFAULT_PATH)
    if(NOT XRT_LIB)
        find_library(XRT_LIB xrt++ /usr/lib/x86_64-linux-gnu)
    endif()

    find_path(XRT_INCLUDE_DIR xrt/xrt_device.h
        PATHS /usr/include
              torch2aie/ (local toolchain)toolchain/xrt/include
              $ENV{XRT_INSTALL_DIR}/include)

    if(NOT XRT_LIB OR NOT XRT_INCLUDE_DIR)
        message(FATAL_ERROR "XRT not found — set XRT_INSTALL_DIR or install XRT")
    endif()

    message(STATUS "XRT: lib=${XRT_LIB} include=${XRT_INCLUDE_DIR}")

    add_library(mlx-lm-npu STATIC
        src/npu/npu_backend.cpp
        src/npu/npu_xclbin_cache.cpp
        src/npu/npu_kernel_runner.cpp
    )
    target_include_directories(mlx-lm-npu PUBLIC
        ${XRT_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(mlx-lm-npu PUBLIC ${XRT_LIB})
    target_compile_definitions(mlx-lm-npu PUBLIC MLX_BUILD_NPU)
    message(STATUS "NPU backend enabled (C++ XRT)")
endif()
```

- [ ] **Step 3: Commit**

```bash
git add src/npu/npu_backend.cpp CMakeLists.txt
git commit -m "feat: implement NPU unified plane backend with XRT direct API"
```

---

### Task 6: Integrate into `quantized_linear.h` with `from_mlx` Helper

**Files:**
- Modify: `include/mlx-lm/common/quantized_linear.h`

**Interfaces:**
- Produces: `mlx_lm::quantized_matmul_from_mlx()` bridge from MLX arrays to NPU

- [ ] **Step 1: Add NPU include guard block**

After the last `#include` in `quantized_linear.h`, add:

```cpp
#ifdef MLX_BUILD_NPU
#include "mlx-lm/npu/npu_backend.h"
#endif
```

- [ ] **Step 2: Add the bridge helper before `linear_forward()`**

```cpp
#ifdef MLX_BUILD_NPU
/// Bridge: call NPU quantized_matmul from MLX arrays.
inline std::optional<mlx::core::array> quantized_matmul_from_mlx(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array* bias,
    const QuantizationInfo& qi)
{
    if (x.dtype() != mlx::core::bfloat16) return std::nullopt;

    auto& x_shape = x.shape();
    auto& w_shape = w.shape();
    if (x_shape.size() != 2 || w_shape.size() != 2) return std::nullopt;

    int M = static_cast<int>(x_shape[0]);
    int K = static_cast<int>(x_shape[1]);
    int N = static_cast<int>(w_shape[0]);  // MLX: w is [out_features, in_features]

    const void* x_ptr = x.data<uint16_t>();
    const void* w_ptr = w.data<uint32_t>();
    const float* s_ptr = qi.scales.data<float>();
    const float* b_ptr = qi.biases.has_value() ? qi.biases->data<float>() : nullptr;

    auto out = mlx::core::array::zeros({M, N}, mlx::core::bfloat16);
    void* out_ptr = out.data<uint16_t>();

    if (!npu::quantized_matmul(x_ptr, w_ptr, s_ptr, b_ptr,
                               out_ptr, M, K, N,
                               qi.group_size, qi.bits, qi.mode)) {
        return std::nullopt;
    }

    if (bias) out = mlx::core::add(out, *bias);
    return out;
}
#endif
```

- [ ] **Step 3: Update `linear_forward()` to add NPU dispatch**

Inside the `if (qi)` block in `linear_forward()`, add the NPU try-before-fallback:

```cpp
    if (qi) {
#ifdef MLX_BUILD_NPU
        auto npu_result = quantized_matmul_from_mlx(input, w, bias, *qi);
        if (npu_result.has_value()) return *npu_result;
#endif
        auto result = mlx::core::quantized_matmul(
              input, w, qi->scales, qi->biases,
              /*transpose=*/true, qi->group_size, qi->bits,
              /*mode=*/qi->mode);
        if (bias) result = mlx::core::add(result, *bias);
        return result;
    }
```

- [ ] **Step 4: Commit**

```bash
git add include/mlx-lm/common/quantized_linear.h
git commit -m "feat: integrate NPU unified plane into linear_forward() dispatch"
```

---

### Task 7: Rewrite test_npu.cpp for the New API

**Files:**
- Modify: `examples/test_npu.cpp`

- [ ] **Step 1: Write the new test**

```cpp
// NPU backend test — verifies NPU detection and quantized_matmul dispatch
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cstring>

#include "mlx-lm/npu/npu_backend.h"

int main() {
    printf("=== NPU Unified Plane Backend Test ===\n\n");

    if (!npu::init()) {
        printf("  ❌ NPU not available\n");
        return 1;
    }

    printf("  ✅ NPU initialized: %s\n", npu::device_name());
    printf("  📊 Peak TFLOPS: %.1f\n\n", npu::peak_tflops());

    // Test 1: BF16 matmul
    printf("--- Test 1: BF16 matmul 32x64x128 ---\n");
    const int M = 32, K = 64, N = 128;

    // Allocate BF16 data (uint16_t = bfloat16)
    std::vector<uint16_t> A(M * K, 0x3F80); // 1.0 in BF16
    std::vector<uint16_t> B(K * N, 0x3F80); // 1.0 in BF16
    std::vector<uint16_t> C(M * N, 0);

    bool ok = npu::matmul_bf16(A.data(), B.data(), C.data(), M, K, N);
    printf("  %s (matmul_bf16)\n", ok ? "✅ PASS" : "❌ FAIL");

    // Test 2: Init device info
    printf("\n--- Test 2: Device Info ---\n");
    printf("  Device: %s\n", npu::device_name());
    printf("  TFLOPS: %.1f\n", npu::peak_tflops());
    printf("  Available: %s\n", npu::is_available() ? "yes" : "no");

    printf("\n=== Test %s ===\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
```

- [ ] **Step 2: Commit**

```bash
git add examples/test_npu.cpp
git commit -m "test: rewrite test_npu for new NPU unified plane API"
```

---

### Task 8: Build and Verify

**Files:**
- Modify: none (build the project)

- [ ] **Step 1: Build with NPU enabled**

```bash
cd lemon-mlx-engine
mkdir -p build && cd build
cmake .. -DMLX_LM_BUILD_NPU=ON -DMLX_LM_BUILD_EXAMPLES=ON \
  -DXRT_INSTALL_DIR=torch2aie/ (local toolchain)toolchain/xrt \
  -DCMAKE_PREFIX_PATH=torch2aie/ (local toolchain)toolchain/xrt
make -j$(nproc) test_npu
```

Expected: Compiles without errors. Links against `libxrt++.so`.

- [ ] **Step 2: Run the test**

```bash
cd lemon-mlx-engine/build
./bin/test_npu
```

Expected: Prints NPU info, runs BF16 matmul test, says PASSED.

- [ ] **Step 3: Copy q4nx.xclbin to expected location**

```bash
mkdir -p /usr/local/lib/npu/xclbins/
cp torch2aie/ (local toolchain)examples/gemm_asymmetric_tile_buffering/config2/final_3072x4096x1536_192x128x96.xclbin \
   /usr/local/lib/npu/xclbins/q4nx.xclbin
cp torch2aie/ (local toolchain)examples/config2/build/final_3072x4096x1536_192x128x96/instr.bin \
   /usr/local/lib/npu/xclbins/q4nx_instr.bin
```

- [ ] **Step 4: Build the full server with NPU enabled**

```bash
cd lemon-mlx-engine/build
cmake .. -DMLX_LM_BUILD_NPU=ON -DMLX_LM_BUILD_EXAMPLES=ON
make -j$(nproc) server
```

Expected: server binary links with `mlx-lm-npu`.

- [ ] **Step 5: Commit final build config**

```bash
git add CMakeLists.txt
git commit -m "build: enable NPU backend with XRT integration"
```

---

### Task 9: Smoke Test with an Actual Model Load

**Files:**
- Modify: none (test the integration)

- [ ] **Step 1: Run the NPU-enabled server with a small model**

```bash
cd lemon-mlx-engine/build
./bin/server --mlx-model mlx-community/Qwen2.5-0.5B-4bit
```

Expected: Server starts, loads model, detects NPU, calls `npu::quantized_matmul()` for weight projections.

- [ ] **Step 2: Verify NPU calls in logs**

Check that server emits `[NPU]` log lines showing xclbin loading and format detection.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: full NPU unified plane backend integration"
```
