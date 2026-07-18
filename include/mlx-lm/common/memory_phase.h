// Phase-scoped ROCm memory API for lemon-mlx-engine.
//
// Wraps mlx::core::rocm::MemoryPhase so the generate path can mark Prefill vs
// Decode and drop prefill freelist blocks before the token loop. On non-ROCm
// builds these are no-ops.
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(MLX_BUILD_ROCM)
#include <mlx/backend/rocm/rocm.h>
#endif

namespace mlx_lm {

enum class MemoryPhase : int {
  Idle = 0,
  Load = 1,
  Prefill = 2,
  Decode = 3,
  Train = 4,
};

inline void set_memory_phase(MemoryPhase phase) {
#if defined(MLX_BUILD_ROCM)
  mlx::core::rocm::set_memory_phase(
      static_cast<mlx::core::rocm::MemoryPhase>(static_cast<int>(phase)));
#else
  (void)phase;
#endif
}

inline MemoryPhase memory_phase() {
#if defined(MLX_BUILD_ROCM)
  return static_cast<MemoryPhase>(
      static_cast<int>(mlx::core::rocm::memory_phase()));
#else
  return MemoryPhase::Idle;
#endif
}

// Prefill → Decode handoff: free freelist slabs stamped during Prefill, then
// switch freelist policy to decode (0.5 util). Returns bytes dropped.
inline size_t memory_end_prefill() {
#if defined(MLX_BUILD_ROCM)
  return mlx::core::rocm::memory_end_prefill();
#else
  return 0;
#endif
}

inline size_t memory_drop_generation(uint32_t gen = 0) {
#if defined(MLX_BUILD_ROCM)
  return mlx::core::rocm::memory_drop_generation(gen);
#else
  (void)gen;
  return 0;
#endif
}

inline uint32_t memory_generation() {
#if defined(MLX_BUILD_ROCM)
  return mlx::core::rocm::memory_generation();
#else
  return 0;
#endif
}

// RAII: set phase on construction, Idle (or previous) on destruction.
class MemoryPhaseGuard {
 public:
  explicit MemoryPhaseGuard(MemoryPhase phase)
      : prev_(memory_phase()) {
    set_memory_phase(phase);
  }
  ~MemoryPhaseGuard() {
    set_memory_phase(prev_);
  }
  MemoryPhaseGuard(const MemoryPhaseGuard&) = delete;
  MemoryPhaseGuard& operator=(const MemoryPhaseGuard&) = delete;

 private:
  MemoryPhase prev_;
};

} // namespace mlx_lm
