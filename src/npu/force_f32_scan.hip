// Force the float32 strided_scan<Sum> kernel to be available at runtime.
#include <hip/hip_runtime.h>
#include "mlx/backend/rocm/reduce/reduce_ops.hpp"

namespace mlx::core::rocm {

template <typename T, typename U, typename Op,
          int N_READS, int BM, int BN,
          bool inclusive, bool reverse>
__global__ void strided_scan(
    const T* in, U* out,
    int32_t axis_size, int64_t stride, int64_t stride_blocks);

// Force instantiation via static constructor taking function address.
struct ForceF32Scan {
  ForceF32Scan() {
    volatile auto p = &strided_scan<float, float, Sum, 4, 64, 64, true, false>;
    (void)p;
  }
};
[[maybe_unused]] static ForceF32Scan force;

} // namespace mlx::core::rocm
