#include <stdint.h>
#include <aie2pintrin.h>

extern "C" void gemm_16x32x32(int32_t *a, int32_t *b, int32_t *c,
                               int32_t M, int32_t K, int32_t N) {
    for (int i = 0; i < M; i++) {
        int32_t *row_a = &a[i * K];
        for (int j = 0; j < N; j++) {
            int32_t sum = 0;
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                v16int32 va = *(v16int32 *)&row_a[k];
                for (int v = 0; v < 16; v++) {
                    sum += ((int32_t *)&va)[v] * b[(k + v) * N + j];
                }
            }
            for (; k < K; k++) sum += row_a[k] * b[k * N + j];
            c[i * N + j] = sum;
        }
    }
}
