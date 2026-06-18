// Validate a DEVICE-offset RoPE (position comes from a [1] device array, not a
// host int) against mx::fast::rope. This is the keystone for graph-replayable
// decode: one captured graph must apply RoPE at a position that advances per
// replay, which requires the position to live on-device.
//
// NeoX / non-traditional convention (traditional=false), matching mx::fast::rope:
//   rotate the first `dims` channels, pass the rest through; pair channel i with
//   i+dims/2; inv_freq[i] = base^(-2i/dims); theta = pos * inv_freq.
#include <mlx/mlx.h>
#include <iostream>
#include <cmath>
namespace mx = mlx::core;

// Device-offset NeoX RoPE for x [B, H, L, D]. `pos` is a scalar device array
// (the absolute position of the FIRST of the L tokens). For decode L==1.
static mx::array device_rope(const mx::array& x, const mx::array& pos, int dims,
                             float base, float scale) {
  int D = x.shape(-1);
  int half = dims / 2;
  // inv_freq[i] = base^(-2i/dims), i in [0, half)
  auto i = mx::arange(0, half, mx::float32);
  auto inv_freq = mx::exp(mx::multiply(i, mx::array(-std::log(base) * 2.0f / dims)));
  // positions for the L tokens: pos + arange(L)
  int L = x.shape(-2);
  auto p = mx::add(mx::astype(pos, mx::float32),
                   mx::astype(mx::arange(0, L, mx::int32), mx::float32));  // [L]
  p = mx::multiply(p, mx::array(scale));
  // theta[L, half] = outer(p, inv_freq)
  auto theta = mx::multiply(mx::expand_dims(p, 1), mx::expand_dims(inv_freq, 0));
  auto cos = mx::cos(theta);  // [L, half]
  auto sin = mx::sin(theta);
  // broadcast to [1,1,L,half]
  cos = mx::reshape(cos, {1, 1, L, half});
  sin = mx::reshape(sin, {1, 1, L, half});

  auto rot = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), L, dims});
  auto pass = (dims < D)
      ? std::optional<mx::array>(mx::slice(x, {0, 0, 0, dims},
                                           {x.shape(0), x.shape(1), L, D}))
      : std::nullopt;
  auto x1 = mx::slice(rot, {0, 0, 0, 0}, {rot.shape(0), rot.shape(1), L, half});
  auto x2 = mx::slice(rot, {0, 0, 0, half}, {rot.shape(0), rot.shape(1), L, dims});
  auto rx1 = mx::subtract(mx::multiply(x1, cos), mx::multiply(x2, sin));
  auto rx2 = mx::add(mx::multiply(x2, cos), mx::multiply(x1, sin));
  auto out = mx::concatenate({rx1, rx2}, -1);
  if (pass.has_value()) out = mx::concatenate({out, pass.value()}, -1);
  return out;
}

int main() {
  int B = 1, H = 4, D = 64, dims = 64;  // full rotary first
  float base = 1.0e6f, scale = 1.0f;
  for (int offset : {0, 1, 7, 123}) {
    auto x = mx::random::normal({B, H, 1, D}, mx::float32);
    mx::eval(x);
    auto ref = mx::fast::rope(x, dims, /*traditional=*/false, base, scale, offset);
    auto pos = mx::array({offset}, {1}, mx::int32);   // position ON DEVICE
    auto got = device_rope(x, pos, dims, base, scale);
    mx::eval({ref, got});
    auto err = mx::max(mx::abs(mx::subtract(ref, got)));
    mx::eval(err);
    float e = err.item<float>();
    std::cout << "offset=" << offset << "  max_abs_err=" << std::scientific << e
              << (e < 1e-4f ? "   ok" : "   *** MISMATCH ***") << "\n";
  }
  // Partial rotary (dims < D): Qwen3-Next uses partial_rotary_factor=0.25.
  {
    int pdims = 16;
    auto x = mx::random::normal({B, H, 1, D}, mx::float32);
    mx::eval(x);
    auto ref = mx::fast::rope(x, pdims, false, base, scale, 7);
    auto got = device_rope(x, mx::array({7}, {1}, mx::int32), pdims, base, scale);
    mx::eval({ref, got});
    auto err = mx::max(mx::abs(mx::subtract(ref, got))); mx::eval(err);
    float e = err.item<float>();
    std::cout << "partial dims=" << pdims << "  max_abs_err=" << std::scientific
              << e << (e < 1e-4f ? "   ok" : "   *** MISMATCH ***") << "\n";
  }
  std::cout << "Done\n";
  return 0;
}
