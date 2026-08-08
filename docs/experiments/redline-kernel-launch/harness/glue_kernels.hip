// SPDX-License-Identifier: Apache-2.0
// Product-equivalent decode glue for Redline ownership (P9).
// Mirrors mlx indexing.hip: _kv_pos_set / _kv_pos_inc / _scalar_copy_i32
// Compile:
//   hipcc --genco --offload-arch=gfx1150 glue_kernels.hip -o glue_kernels-gfx1150.co

#include <hip/hip_runtime.h>

// kernarg: [p:u64 @0][v:i32 @8]
extern "C" __global__ void glue_pos_set(int* p, int v) {
    p[0] = v;
}

// kernarg: [p:u64 @0][delta:i32 @8]
extern "C" __global__ void glue_pos_inc(int* p, int delta) {
    p[0] += delta;
}

// kernarg: [dst:u64 @0][src:u64 @8]
extern "C" __global__ void glue_scalar_copy_i32(int* dst, const int* src) {
    dst[0] = src[0];
}
