// Isolated repro for dynamic slice_update (device-array start index) hang.
#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;

static bool run(mx::Dtype dt, const char* name) {
  int B = 1, H = 2, CAP = 64, D = 256;
  int slot = 5;
  auto buf = mx::zeros({B, H, CAP, D}, dt);
  auto update = mx::full({B, H, 1, D}, 3.0f, dt);
  auto pos = mx::array({slot}, {1}, mx::int32);

  buf = mx::slice_update(buf, update, pos, std::vector<int>{2});
  mx::eval(buf);

  auto buf_f = mx::astype(buf, mx::float32);
  auto at_slot = mx::slice(buf_f, {0, 0, slot, 0}, {B, H, slot + 1, D});
  auto at_zero = mx::slice(buf_f, {0, 0, 0, 0}, {B, H, 1, D});
  auto at_prev = mx::slice(buf_f, {0, 0, slot - 1, 0}, {B, H, slot, D});
  auto slot_min = mx::min(at_slot);
  auto slot_max = mx::max(at_slot);
  auto zero_max = mx::max(mx::abs(at_zero));
  auto prev_max = mx::max(mx::abs(at_prev));
  mx::eval({slot_min, slot_max, zero_max, prev_max});

  float smin = slot_min.item<float>(), smax = slot_max.item<float>();
  float zmax = zero_max.item<float>(), pmax = prev_max.item<float>();
  std::cout << name << ": slot[" << slot << "] min=" << smin << " max=" << smax
            << " (expect ~3,~3); slot[0] absmax=" << zmax
            << " slot[4] absmax=" << pmax << " (expect 0,0)\n";
  bool ok = (std::abs(smin - 3.0f) < 0.05f) && (std::abs(smax - 3.0f) < 0.05f) &&
            (zmax == 0.0f) && (pmax == 0.0f);
  std::cout << "  " << (ok ? "PASS" : "*** FAIL ***") << "\n";
  return ok;
}

int main() {
  bool ok = true;
  ok &= run(mx::float32, "f32");
  ok &= run(mx::bfloat16, "bf16");
  std::cout << (ok ? "ALL PASS" : "*** SOME FAIL ***") << "\n";
  return ok ? 0 : 1;
}
