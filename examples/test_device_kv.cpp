// Validate the two remaining device-driven decode primitives against host refs:
//  (1) device-indexed KV WRITE: put_along_axis at a position held in a device
//      array, vs the host slice_update at offset.
//  (2) device-length attention MASK: arange(CAP) <= pos, so a single decode
//      query over the full fixed-size KV attends only to written positions.
#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;

int main() {
  int B = 1, H = 2, CAP = 8, D = 4;

  // --- (1) KV write: device put_along_axis vs host slice_update ---
  auto KV_ref = mx::zeros({B, H, CAP, D});
  auto KV_dev = mx::zeros({B, H, CAP, D});
  for (int t = 0; t < 5; ++t) {
    auto nk = mx::full({B, H, 1, D}, float(t + 1));        // distinct per step
    KV_ref = mx::slice_update(KV_ref, nk, mx::Shape{0, 0, t, 0},
                              mx::Shape{B, H, t + 1, D});
    // Device write: dynamic-start slice_update with the time position on the
    // device (axis 2). This is exactly the cache write, but graph-replayable.
    auto posd = mx::array({t}, {1}, mx::int32);            // position ON DEVICE
    KV_dev = mx::slice_update(KV_dev, nk, posd, std::vector<int>{2});
  }
  mx::eval({KV_ref, KV_dev});
  auto werr = mx::max(mx::abs(mx::subtract(KV_ref, KV_dev)));
  mx::eval(werr);
  float we = werr.item<float>();
  std::cout << "KV write (put_along_axis @ device pos) max_abs_err=" << we
            << (we == 0.0f ? "   ok" : "   *** MISMATCH ***") << "\n";

  // --- (2) device-length mask: arange(CAP) <= pos ---
  // For a decode query at absolute position p, valid keys are 0..p. Build an
  // additive mask (0 where valid, -inf where not) from a DEVICE position.
  for (int p : {0, 3, 7}) {
    auto pos = mx::array({p}, {1}, mx::int32);
    auto cols = mx::arange(0, CAP, mx::int32);             // [CAP]
    auto valid = mx::less_equal(cols, mx::reshape(pos, {1}));  // [CAP] bool
    auto addmask = mx::where(valid, mx::array(0.0f),
                             mx::array(-std::numeric_limits<float>::infinity()));
    mx::eval(addmask);
    // expected: first p+1 entries are 0, rest -inf
    int n_valid = 0;
    for (int j = 0; j < CAP; ++j)
      if (addmask.data<float>()[j] == 0.0f) n_valid++;
    std::cout << "mask @pos=" << p << "  valid_cols=" << n_valid
              << (n_valid == p + 1 ? "   ok" : "   *** WRONG ***") << "\n";
  }

  std::cout << "Done\n";
  return 0;
}
