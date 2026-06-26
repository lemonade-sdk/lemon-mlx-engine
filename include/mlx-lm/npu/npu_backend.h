#pragma once

#include <cstdint>

// Path to NPU JIT helper (set at build time)
#ifndef NPU_INSTALL_DIR
#define NPU_INSTALL_DIR "/usr/local"
#endif

namespace npu {

/// Initialize the NPU device. Returns true if NPU is available.
bool init();

/// Check if NPU is initialized and available.
bool is_available();

/// Get NPU device name (e.g. "RyzenAI-npu5").
const char* device_name();

/// Perform GEMM: C[M][N] = A[M][K] * B[K][N]
/// All matrices are row-major int32.
/// Returns true on success, false on failure (falls back to CPU/GPU).
bool matmul(
    const int32_t* A, const int32_t* B, int32_t* C,
    int M, int K, int N);

/// Get total NPU compute in TFLOPS (peak theoretical).
float peak_tflops();

} // namespace npu
