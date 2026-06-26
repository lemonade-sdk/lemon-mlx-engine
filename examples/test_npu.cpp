// NPU backend test — verifies NPU detection and GEMM
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlx-lm/npu/npu_backend.h"

int main() {
    printf("=== NPU Backend Test ===\n\n");

    // Initialize NPU
    printf("Initializing NPU...\n");
    if (!npu::init()) {
        printf("  ❌ NPU not available\n");
        printf("  ℹ️  Set NPU_XCLBIN_PATH or build with -DMLX_LM_BUILD_NPU=ON\n");
        return 1;
    }

    printf("  ✅ NPU initialized: %s\n", npu::device_name());
    printf("  📊 Peak TFLOPS: %.1f\n\n", npu::peak_tflops());

    // Run GEMM test
    const int M = 16, K = 32, N = 32;
    printf("Running GEMM %dx%dx%d on NPU...\n", M, K, N);

    std::vector<int32_t> A(M * K, 2);
    std::vector<int32_t> B(K * N, 3);
    std::vector<int32_t> C(M * N, 0);

    if (!npu::matmul(A.data(), B.data(), C.data(), M, K, N)) {
        printf("  ❌ GEMM failed on NPU\n");
        return 1;
    }

    // Verify results
    int32_t expected = 2 * 3 * K;  // 192
    bool pass = true;
    for (int i = 0; i < M * N; i++) {
        if (C[i] != expected) {
            printf("  ❌ Mismatch at [%d]: got %d, expected %d\n", i, C[i], expected);
            pass = false;
            break;
        }
    }

    if (pass) {
        printf("  ✅ GEMM result: %d (expected %d)\n", C[0], expected);
        printf("  ✅ All %d elements match!\n\n", M * N);
    }

    printf("=== Test %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
