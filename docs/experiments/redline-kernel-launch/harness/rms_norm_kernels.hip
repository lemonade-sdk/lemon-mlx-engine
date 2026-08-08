// SPDX-License-Identifier: Apache-2.0
// Product-equivalent packed RMSNorm for Redline ownership (P12).
// Mirrors mlx/backend/rocm/rms_norm.hip packed path algorithm:
//   normalizer = 1/sqrt(mean(x^2)+eps); out = w * T(x * normalizer)
// Compile:
//   hipcc --genco --offload-arch=gfx1150 rms_norm_kernels.hip \
//     -o rms_norm_kernels-gfx1150.co

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__device__ float warp_reduce_sum_rms(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_xor(val, offset);
  }
  return val;
}

template <typename T, int BLOCK_DIM, int N_READS = 4>
__device__ void rms_norm_row(
    const T* x,
    const T* w,
    T* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  float normalizer = 0;
  for (int i = threadIdx.x * N_READS; i < (int)axis_size; i += BLOCK_DIM * N_READS) {
#pragma unroll
    for (int j = 0; j < N_READS && i + j < (int)axis_size; ++j) {
      float t = static_cast<float>(x[i + j]);
      normalizer += t * t;
    }
  }

  __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE + 1];

  float warp_sum = warp_reduce_sum_rms(normalizer);
  int lane = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  if (lane == 0) {
    shared_sum[warp_id] = warp_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    normalizer =
        (lane < (BLOCK_DIM + WARP_SIZE - 1) / WARP_SIZE) ? shared_sum[lane] : 0.f;
    normalizer = warp_reduce_sum_rms(normalizer);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    shared_sum[0] = normalizer;
  }
  __syncthreads();
  normalizer = 1.0f / sqrtf(shared_sum[0] / (float)axis_size + eps);

  for (int i = threadIdx.x * N_READS; i < (int)axis_size; i += BLOCK_DIM * N_READS) {
#pragma unroll
    for (int j = 0; j < N_READS && i + j < (int)axis_size; ++j) {
      int idx = i + j;
      T normalized = static_cast<T>(static_cast<float>(x[idx]) * normalizer);
      T wi = (w_stride == 0) ? w[0] : w[idx * w_stride];
      out[idx] = wi * normalized;
    }
  }
}

// kernarg (natural HIP pack):
//   x:u64@0  w:u64@8  out:u64@16  eps:f32@24  axis_size:u32@28  w_stride:i64@32
// grid.x = n_rows, block.x = 256
extern "C" __global__ void rms_norm_f32(
    const float* x,
    const float* w,
    float* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  size_t row = blockIdx.x;
  rms_norm_row<float, 256, 4>(
      x + (int64_t)row * axis_size,
      w,
      out + (int64_t)row * axis_size,
      eps,
      axis_size,
      w_stride);
}

extern "C" __global__ void rms_norm_f16(
    const __half* x,
    const __half* w,
    __half* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  size_t row = blockIdx.x;
  rms_norm_row<__half, 256, 4>(
      x + (int64_t)row * axis_size,
      w,
      out + (int64_t)row * axis_size,
      eps,
      axis_size,
      w_stride);
}

extern "C" __global__ void rms_norm_bf16(
    const __hip_bfloat16* x,
    const __hip_bfloat16* w,
    __hip_bfloat16* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
  size_t row = blockIdx.x;
  rms_norm_row<__hip_bfloat16, 256, 4>(
      x + (int64_t)row * axis_size,
      w,
      out + (int64_t)row * axis_size,
      eps,
      axis_size,
      w_stride);
}
