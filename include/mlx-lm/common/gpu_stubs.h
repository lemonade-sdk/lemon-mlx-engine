// Forward declarations for GPU primitives not (yet) exported by the
// NripeshN/mlx fork. Stub implementations live in src/common/gpu_stubs.cpp.
//
// Include this header anywhere these symbols are called. When the upstream
// MLX fork catches up, delete this header and gpu_stubs.cpp.
#pragma once

#include <cstddef>
#include <cstdint>

namespace mlx::core {
class array;
}

namespace mlx::core {

// KV-cache position helpers
void gpu_kv_pos_set(array& pos, int v);
void gpu_kv_pos_increment(array& pos, int delta);

// GPU scalar/buffer copy
void gpu_scalar_copy_i32(array& dst, array& src);
void gpu_buffer_copy(array& dst, array& src);

// Decode arena lifecycle
bool decode_arena_begin(size_t capacity, int device, void* stream);
void decode_arena_reset();
void decode_arena_end();
size_t decode_arena_high_water();
bool decode_arena_overflowed();

// Pure decode recording
void decode_pure_record(int slot);
void decode_pure_replay(int slot);
void decode_pure_off();
size_t decode_pure_chain_len(int slot);

} // namespace mlx::core
