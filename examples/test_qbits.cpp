// Numerical-correctness test for affine-quantized matmul across bit widths.
// Mirrors REAL model usage: bf16 activations + bf16 scales/biases (the tiled
// ROCm QMV kernel only has bf16/fp16 instantiations — float32 would itself hit
// an unimplemented path and is not representative). Compares quantized_matmul
// against a dequantize-then-matmul reference using RELATIVE error, since bf16
// accumulation alone produces ~0.3% abs error at these magnitudes.
#include <mlx/mlx.h>
#include <iostream>
#include <cmath>
namespace mx = mlx::core;

// returns mean(|a-b|) / mean(|ref|) — ~0 for correct, ~1 for uncorrelated garbage
static float rel_err(const mx::array& a, const mx::array& b) {
  auto af = mx::astype(a, mx::float32);
  auto bf = mx::astype(b, mx::float32);
  auto diff = mx::mean(mx::abs(mx::subtract(af, bf)));
  auto scale = mx::mean(mx::abs(af));
  mx::eval({diff, scale});
  return diff.item<float>() / std::max(1e-6f, scale.item<float>());
}

int main() {
  int gs = 64;

  std::cout << "=== 2D quantized_matmul, bf16 (attention/projection path) ===\n";
  std::cout << "W=[512,1024], x=[4,1024]\n";
  for (int bits : {4, 6, 8}) {
    auto W = mx::astype(mx::random::normal({512, 1024}, mx::float32), mx::bfloat16);
    auto x = mx::astype(mx::random::normal({4, 1024}, mx::float32), mx::bfloat16);
    mx::eval({W, x});
    auto q = mx::quantize(W, gs, bits);  // scales/biases inherit bf16
    auto ref = mx::matmul(x, mx::transpose(mx::dequantize(q[0], q[1], q[2], gs, bits)));
    auto gpu = mx::quantized_matmul(x, q[0], q[1], q[2], /*transpose=*/true, gs, bits);
    mx::eval({ref, gpu});
    float r = rel_err(ref, gpu);
    std::cout << "  bits=" << bits << "  rel_err=" << std::scientific << r
              << (r > 0.05f ? "   *** GARBAGE ***" : "   ok") << "\n";
  }

  // 3D gather_qmm (MoE decode path, M=1). Reference via take + batched matmul
  // (unambiguous). Set MLX_ROCM_GATHER_QMV_USE_TILED=1 to route 4/8-bit through
  // the tiled gather kernel; this test guards its correctness vs warp-shared.
  std::cout << "\n=== gather_qmm, bf16, M=1 decode (MoE expert path) ===\n";
  std::cout << "w=[E=8,N=512,K=2048], B=16 (token,expert) pairs, M=1\n";
  for (int bits : {4, 6, 8}) {
    int E = 8, B = 16, N = 512, K = 2048;
    auto W = mx::astype(mx::random::normal({E, N, K}, mx::float32), mx::bfloat16);
    auto x = mx::astype(mx::random::normal({B, 1, K}, mx::float32), mx::bfloat16);
    mx::eval({W, x});
    auto q = mx::quantize(W, gs, bits);
    auto lhs = mx::arange(0, B, mx::uint32);              // identity (each batch its own x)
    auto rhs = mx::astype(mx::remainder(mx::arange(0, B, mx::int32),
                                        mx::array(E)), mx::uint32);  // expert per batch
    mx::eval({lhs, rhs});
    auto wdeq = mx::dequantize(q[0], q[1], q[2], gs, bits);          // [E,N,K]
    auto wg = mx::take(wdeq, rhs, 0);                                // [B,N,K]
    auto ref = mx::matmul(x, mx::swapaxes(wg, -1, -2));              // [B,1,N]
    auto gpu = mx::gather_qmm(x, q[0], q[1], q[2], lhs, rhs,
                              /*transpose=*/true, gs, bits);         // [B,1,N]
    mx::eval({ref, gpu});
    float r = rel_err(ref, gpu);
    std::cout << "  bits=" << bits << "  ref=" << ref.shape(0) << "x" << ref.shape(2)
              << "  gpu=" << gpu.shape(0) << "x" << gpu.shape(2)
              << "  rel_err=" << std::scientific << r
              << (r > 0.05f ? "   *** GARBAGE ***" : "   ok") << "\n";
  }

  std::cout << "\n=== Done ===\n";
  return 0;
}
