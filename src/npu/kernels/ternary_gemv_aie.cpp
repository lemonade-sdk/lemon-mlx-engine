// SPDX-License-Identifier: MIT
// AIE2 kernel for ternary GEMV — self-contained for IRON JIT compilation.
// Compiles with: clang++ --target=aie2-none-unknown-elf -O2 -std=c++20

typedef unsigned char uint8_t;
typedef int int32_t;

extern "C" void
ternary_gemv_aie(
    const uint8_t* __restrict packed_weights,
    const float* __restrict activations,
    float* __restrict result,
    int N, int K)
{
    for (int oc = 0; oc < N; oc++) {
        int row = oc / 4;
        int lane = oc % 4;
        int bit_shift = lane * 2;
        float acc = 0.0f;

        for (int k = 0; k < K; k++) {
            uint8_t byte_val = packed_weights[row * K + k];
            int code = (byte_val >> bit_shift) & 0x03;
            int tv = code - 1;
            acc += (float)tv * activations[k];
        }
        result[oc] = acc;
    }
}
