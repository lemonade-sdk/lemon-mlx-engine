// Smoke test: phase transitions + generation-stamped freelist drop.
// Build under MLX_BUILD_ROCM (examples/test_arena pattern).
#include <cstdio>
#include <mlx/mlx.h>

#if defined(MLX_BUILD_ROCM)
#include <mlx/backend/rocm/rocm.h>
#endif

namespace mx = mlx::core;

int main() {
#if !defined(MLX_BUILD_ROCM)
  printf("SKIP (no ROCm)\n");
  return 0;
#else
  if (!mx::rocm::is_available()) {
    printf("SKIP (ROCm unavailable)\n");
    return 0;
  }
  mx::set_default_device(mx::Device::gpu);
  { auto w = mx::add(mx::ones({4}), mx::ones({4})); mx::eval(w); }

  using mx::rocm::MemoryPhase;

  mx::rocm::set_memory_phase(MemoryPhase::Prefill);
  if (mx::rocm::memory_phase() != MemoryPhase::Prefill) {
    printf("FAIL: phase not Prefill\n");
    return 1;
  }
  uint32_t g0 = mx::rocm::memory_generation();
  printf("prefill gen=%u\n", g0);

  // Allocate / free some workspace so freelist has prefill-stamped blocks.
  {
    auto a = mx::ones({1024, 1024}, mx::float32);
    auto b = mx::matmul(a, a);
    mx::eval(b);
  }
  // Drop live refs so buffers recycle into freelist.
  mx::clear_cache();

  size_t dropped = mx::rocm::memory_end_prefill();
  printf("end_prefill dropped_bytes=%zu phase=%d gen=%u\n",
         dropped,
         static_cast<int>(mx::rocm::memory_phase()),
         mx::rocm::memory_generation());

  if (mx::rocm::memory_phase() != MemoryPhase::Decode) {
    printf("FAIL: expected Decode after end_prefill\n");
    return 2;
  }

  mx::rocm::set_memory_phase(MemoryPhase::Train);
  if (mx::rocm::memory_phase() != MemoryPhase::Train) {
    printf("FAIL: phase not Train\n");
    return 3;
  }
  mx::rocm::set_memory_phase(MemoryPhase::Idle);

  printf("OK phase-memory smoke\n");
  return 0;
#endif
}
