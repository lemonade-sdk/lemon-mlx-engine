// Standalone check: the decode arena hands out identical device addresses for
// an identical allocation sequence across token resets. This determinism is the
// precondition for build-once HIP-graph relaunch.
#include <cstdio>
#include <cstdlib>
#include <mlx/mlx.h>
#include <mlx-lm/common/gpu_stubs.h>

namespace mx = mlx::core;

int main() {
  fprintf(stderr, "[arena] start\n");
  mx::set_default_device(mx::Device::gpu);
  fprintf(stderr, "[arena] device set; warmup...\n");
  { auto w = mx::add(mx::ones({4}), mx::ones({4})); mx::eval(w); }
  fprintf(stderr, "[arena] warmup done; begin arena\n");

  const size_t cap = size_t(256) * 1024 * 1024;
  if (!mx::decode_arena_begin(cap, 0, nullptr)) {
    printf("decode_arena_begin failed\n");
    return 1;
  }
  fprintf(stderr, "[arena] begin ok\n");

  // A small but multi-op "token": several allocations in a fixed order.
  auto run = []() -> void* {
    auto a = mx::ones({512, 512}, mx::float32);
    auto b = mx::matmul(a, a);
    auto c = mx::add(b, a);
    auto d = mx::multiply(c, mx::array(2.0f));
    mx::eval(d);
    return static_cast<void*>(d.data<float>());
  };

  void* p1 = run();
  size_t hw1 = mx::decode_arena_high_water();

  mx::decode_arena_reset();
  void* p2 = run();

  mx::decode_arena_reset();
  void* p3 = run();

  size_t hw3 = mx::decode_arena_high_water();
  bool overflow = mx::decode_arena_overflowed();
  mx::decode_arena_end();

  printf("p1=%p p2=%p p3=%p\n", p1, p2, p3);
  printf("high_water token1=%zu token3=%zu overflow=%d\n", hw1, hw3,
         (int)overflow);
  bool ok = (p1 == p2) && (p2 == p3) && !overflow && (hw1 == hw3);
  printf(ok ? "DETERMINISTIC OK\n" : "NONDETERMINISTIC FAIL\n");
  return ok ? 0 : 2;
}
