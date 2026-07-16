// SDPA correctness harness: compare the ROCm GPU flash kernel against an
// explicit CPU reference (matmul + softmax + matmul) across head dims and
// sequence lengths. A divergence that grows with L points at the flash
// online-softmax accumulation; a flat large divergence at one D points at an
// LDS/launch problem for that D. MHA (H==Hkv) to keep the reference simple.
#include <mlx/mlx.h>
#include <cmath>
#include <cstdio>
namespace mx = mlx::core;

static mx::array reference(const mx::array& q, const mx::array& k,
                           const mx::array& v, float scale, int L, bool causal) {
  auto qf = mx::astype(q, mx::float32, mx::Device::cpu);
  auto kf = mx::astype(k, mx::float32, mx::Device::cpu);
  auto vf = mx::astype(v, mx::float32, mx::Device::cpu);
  auto kt = mx::swapaxes(kf, -1, -2, mx::Device::cpu);
  auto scores = mx::multiply(mx::matmul(qf, kt, mx::Device::cpu),
                             mx::array(scale), mx::Device::cpu);
  if (causal) {
    auto r = mx::arange(L, mx::int32, mx::Device::cpu);
    auto col = mx::reshape(r, {1, L}, mx::Device::cpu);
    auto row = mx::reshape(r, {L, 1}, mx::Device::cpu);
    auto allowed = mx::less_equal(col, row, mx::Device::cpu);
    auto mask = mx::where(allowed, mx::array(0.0f), mx::array(-1e9f),
                          mx::Device::cpu);
    scores = mx::add(scores, mask, mx::Device::cpu);
  }
  auto p = mx::softmax(scores, -1, true, mx::Device::cpu);
  return mx::matmul(p, vf, mx::Device::cpu);
}

static void run(int H, int L, int D, bool causal) {
  mx::random::seed(1234);
  auto q = mx::astype(mx::random::normal({1, H, L, D}, mx::float32), mx::bfloat16);
  auto k = mx::astype(mx::random::normal({1, H, L, D}, mx::float32), mx::bfloat16);
  auto v = mx::astype(mx::random::normal({1, H, L, D}, mx::float32), mx::bfloat16);
  mx::eval({q, k, v});

  float scale = 1.0f / std::sqrt((float)D);
  std::string mode = causal ? "causal" : "";

  auto og = mx::fast::scaled_dot_product_attention(
      q, k, v, scale, mode, {}, {}, mx::Device(mx::Device::gpu, 0));
  mx::eval(og);
  auto gf = mx::astype(og, mx::float32, mx::Device::cpu);

  auto rf = reference(q, k, v, scale, L, causal);
  mx::eval(rf);

  auto d = mx::abs(mx::subtract(gf, rf, mx::Device::cpu), mx::Device::cpu);
  auto md = mx::max(d, mx::Device::cpu);
  auto mean = mx::mean(d, mx::Device::cpu);
  mx::eval({md, mean});

  float maxd = md.item<float>(), meand = mean.item<float>();
  printf("D=%3d L=%5d %-7s max|g-ref|=%.4f  mean=%.5f  %s\n",
         D, L, causal ? "causal" : "full", maxd, meand,
         maxd < 0.06f ? "OK" : "*** DIVERGE ***");
}

int main() {
  // Bisect the tile boundary. BLOCK_M=BLOCK_N=64, so:
  //   L<=16  : one 16x16 WMMA tile, one wave writes output (rows 0..L-1).
  //   L=32   : two query waves written, two key tiles -> first multi-tile case.
  //   L=64   : full single K/V block (still NO online-softmax rescale).
  //   L>=128 : first time the k_start loop runs >1 iteration (rescale active).
  // If L=16 passes but L=32 fails, the bug is the multi-tile full-block WMMA
  // path, not the online softmax. Run both causal and non-causal to factor the
  // causal mask out of the picture.
  for (int causal = 1; causal >= 0; causal--) {
    printf("=== %s ===\n", causal ? "causal" : "non-causal");
    for (int D : {64, 128, 256}) {
      for (int L : {8, 16, 17, 32, 48, 64, 128, 256, 1024}) {
        run(8, L, D, causal != 0);
      }
      printf("\n");
    }
  }
  return 0;
}
