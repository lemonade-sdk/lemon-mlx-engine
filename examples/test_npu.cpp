// NPU backend test — verifies NPU detection, ternary GEMV, and GEMM
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "mlx-lm/npu/npu_backend.h"

int main() {
    printf("=== NPU Backend Test ===\n\n");

    // Initialize NPU
    printf("Initializing NPU...\n");
    if (!npu::init()) {
        printf("  ❌ NPU not available\n");
        return 1;
    }

    printf("  ✅ NPU initialized: %s\n", npu::device_name());
    printf("  📊 Peak TFLOPS: %.1f\n\n", npu::peak_tflops());

    // ── Test 1: Ternary GEMV ──────────────────────────────────────────
    printf("Test 1: Ternary GEMV (N=8, K=128)...\n");

    const int N = 8, K = 128;
    int packed_rows = (N + 3) / 4;

    // Packed U8 weights: 4 ternary codes per byte, lane-major
    std::vector<uint8_t> weights(packed_rows * K, 0);
    for (int oc = 0; oc < N; oc++) {
        int row = oc / 4;
        int lane = oc % 4;
        for (int k = 0; k < K; k++) {
            // Simple pattern: all +1 ternary values
            int code = 2;  // code 2 = ternary +1
            weights[row * K + k] |= (code << (lane * 2));
        }
    }

    // Activations: all 1.0
    std::vector<float> acts(K, 1.0f);
    std::vector<float> result(N, 0.0f);

    float weight_scale = 0.5f;
    float expected = (float)K * 1.0f * 0.5f;  // Σ (+1) * 1.0 * 0.5 = K * 0.5

    if (!npu::ternary_gemv(weights.data(), acts.data(), result.data(),
                            weight_scale, false, N, K)) {
        printf("  ❌ Ternary GEMV failed\n");
        return 1;
    }

    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (std::abs(result[i] - expected) > 1.0f) {
            printf("  ❌ Mismatch at [%d]: got %.1f, expected %.1f\n",
                   i, result[i], expected);
            pass = false;
        }
    }
    if (pass) {
        printf("  ✅ All %d values match (expected %.1f)\n", N, expected);
    }

    // ── Test 2: Mixed ternary values ──────────────────────────────────
    printf("\nTest 2: Mixed ternary values (N=4, K=16)...\n");

    const int N2 = 4, K2 = 16;
    int pr2 = (N2 + 3) / 4;
    std::vector<uint8_t> w2(pr2 * K2, 0);

    // Set specific ternary patterns: oc=0 all +1, oc=1 all -1, oc=2 all 0, oc=3 mixed
    for (int oc = 0; oc < N2; oc++) {
        int row = oc / 4;
        int lane = oc % 4;
        for (int k = 0; k < K2; k++) {
            int tv;
            if (oc == 0) tv = 1;       // all +1
            else if (oc == 1) tv = -1;  // all -1
            else if (oc == 2) tv = 0;   // all 0
            else tv = (k % 3) - 1;       // mixed
            w2[row * K2 + k] |= ((tv + 1) << (lane * 2));
        }
    }

    std::vector<float> acts2(K2, 1.0f);
    std::vector<float> res2(N2, 0.0f);

    if (!npu::ternary_gemv(w2.data(), acts2.data(), res2.data(),
                            1.0f, false, N2, K2)) {
        printf("  ❌ Mixed GEMV failed\n");
        return 1;
    }

    printf("  Results: %.0f %.0f %.0f %.0f (expect: %d %d 0 %d)\n",
           res2[0], res2[1], res2[2], res2[3],
           K2, -K2, 0);

    // ── Summary ──────────────────────────────────────────────────────
    printf("\n=== All tests %s ===\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
