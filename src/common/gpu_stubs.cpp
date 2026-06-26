// GPU stub implementations for ROCm/GPU primitives not yet exposed by the
// NripeshN/mlx fork at the pinned commit. These are forward-declared in
// graph_decode.cpp and generate.cpp but the underlying MLX library does not
// (yet) export them. The stubs let the engine link cleanly on ROCm.
//
// When the upstream fork catches up, delete this file.

#include "mlx/mlx.h"

namespace mx = mlx::core;

namespace mlx::core {

// ── KV-cache position helpers (graph_decode.cpp) ──

void gpu_kv_pos_set(array& pos, int v) {
    // Fallback: CPU-style assignment
    pos = mx::array(v, mx::int32);
}

void gpu_kv_pos_increment(array& pos, int delta) {
    // Fallback: CPU-style increment
    pos = mx::add(pos, mx::array(delta, mx::int32));
}

void gpu_scalar_copy_i32(array& dst, array& src) {
    // Fallback: element-wise copy via eval
    mx::eval(src);
    dst = mx::astype(src, mx::int32);
}

void gpu_buffer_copy(array& dst, array& src) {
    // Fallback: copy via eval
    mx::eval(src);
    dst = mx::astype(src, dst.dtype());
}

// ── Decode arena (graph_decode.cpp, generate.cpp, test_arena.cpp) ──

bool decode_arena_begin(size_t capacity, int device, void* stream) {
    // Stub: always succeed
    return true;
}

void decode_arena_reset() {
    // Stub: no-op
}

void decode_arena_end() {
    // Stub: no-op
}

size_t decode_arena_high_water() {
    // Stub: return minimal
    return 0;
}

bool decode_arena_overflowed() {
    // Stub: never overflow
    return false;
}

// ── Pure decode recording (generate.cpp) ──

void decode_pure_record(int slot) {
    // Stub: no-op
    (void)slot;
}

void decode_pure_replay(int slot) {
    // Stub: no-op
    (void)slot;
}

void decode_pure_off() {
    // Stub: no-op
}

size_t decode_pure_chain_len(int slot) {
    // Stub: return 0 (no chain)
    (void)slot;
    return 0;
}

} // namespace mlx::core
