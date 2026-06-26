// NPU backend — AMD XDNA NPU acceleration
#pragma once

#include <cstdint>

namespace npu {

// Initialize NPU. Returns true if NPU is available.
bool init();

// Check if NPU is initialized and accessible
bool is_available();

// Get NPU device name
const char* device_name();

// Get peak TFLOPS of the NPU
float peak_tflops();

// Run ternary GEMV on NPU:
//   result[oc] = scale * Σ_k ternary(weights[oc,k]) * activations[k]
// where weights are packed U8 with 4 ternary codes per byte (lane-major).
// Returns true on success.
bool ternary_gemv(
    const uint8_t* packed_weights,  // [ceil(N/4), K] packed ternary codes
    const float* activations,       // [K] float32 activations
    float* result,                  // [N] output (float32, will be scaled)
    float weight_scale,             // scale factor
    bool invert_scale,              // use 1/weight_scale if true
    int N,                          // number of output rows
    int K                           // input dimension
);

// Run int32 GEMM on NPU (legacy, for basic testing)
bool matmul(const int32_t* A, const int32_t* B, int32_t* C,
            int M, int K, int N);

} // namespace npu
