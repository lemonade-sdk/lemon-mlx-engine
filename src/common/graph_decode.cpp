// Copyright © 2025
#include "mlx-lm/common/graph_decode.h"
#include "mlx-lm/common/redline_decode_session.h"
#include <cstdlib>

namespace mx = mlx::core;

// In-place device-scalar kernels (ROCm backend): mutate the pos buffer contents
// without reallocating, keeping the captured graph's baked address valid.
namespace mlx::core {
void gpu_kv_pos_set(array& pos, int v);
void gpu_kv_pos_increment(array& pos, int delta);
void gpu_scalar_copy_i32(array& dst, array& src);
}

namespace mlx_lm {

namespace {
// Must match mlx::core::rocm::RocmBuffer field order (backend/rocm/allocator.h).
// Used only to read the GPU VRAM pointer for research kernarg bake.
struct RocmBufferLayout {
    void* data;
    size_t size;
    bool is_managed;
    int device;
    void* host_shadow;
    bool host_dirty;
    void* alloc_stream;
};
} // namespace

static bool g_external = false;
static bool g_capturing = false;

// Constructed lazily on first use (not at static-init time, before --device
// selection).
mx::array& graph_decode_pos() {
    static mx::array* g_pos = nullptr;
    if (g_pos == nullptr) {
        g_pos = new mx::array(mx::zeros({1}, mx::int32));
        mx::eval(*g_pos);
    }
    return *g_pos;
}

void set_graph_decode_pos(int offset) {
    // Mutate the pos buffer in place via a raw kernel.
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    auto& p = graph_decode_pos();
    // P9: optional Redline ownership of this product glue launch.
    if (redline_try_own_pos_set(p, offset)) {
        return;
    }
    mx::gpu_kv_pos_set(p, offset);
    mx::synchronize(mx::default_stream(mx::default_device()));
#else
    auto& p = graph_decode_pos();
    p = mx::slice_update(p, mx::broadcast_to(mx::array(offset, mx::int32), p.shape()),
                         mx::Shape(p.ndim(), 0), p.shape());
    mx::eval(p);
#endif
}

// Advance the device position in place by delta (loop-owned, between replays).
void advance_graph_decode_pos(int delta) {
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    auto& p = graph_decode_pos();
    if (redline_try_own_pos_inc(p, delta)) {
        return;
    }
    mx::gpu_kv_pos_increment(p, delta);
#else
    set_graph_decode_pos(0);  // non-ROCm has no graph path
#endif
}

bool graph_external_pos() { return g_external; }
void set_graph_external_pos(bool on) { g_external = on; }

// Fixed-address [1,1] int32 input-token buffer. Constructed lazily (after device
// selection) and kept resident so its device address never changes.
mx::array& graph_decode_input() {
    static mx::array* g_input = nullptr;
    if (g_input == nullptr) {
        g_input = new mx::array(mx::zeros({1, 1}, mx::int32));
        mx::eval(*g_input);
    }
    return *g_input;
}

void set_graph_decode_input_from(mx::array& token) {
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    auto& dst = graph_decode_input();
    // token may be [1] or [1,1]; the kernel copies element 0 either way.
    // P9: optional Redline ownership of product scalar-copy glue.
    if (redline_try_own_scalar_copy_i32(dst, token)) {
        return;
    }
    mx::gpu_scalar_copy_i32(dst, token);
#else
    (void)token;
#endif
}

bool graph_capturing() { return g_capturing; }
void set_graph_capturing(bool on) { g_capturing = on; }

bool graph_decode_enabled() {
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    // Opt-in during bring-up.
    static const bool on = std::getenv("MLX_DECODE_GRAPH") != nullptr;
    return on;
#else
    return false;
#endif
}

void* graph_decode_device_data_ptr(mx::array& a) {
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    // Ensure buffer is allocated / resident on the default device.
    mx::eval(a);
    if (!a.buffer().ptr()) {
        return nullptr;
    }
    auto* rb = static_cast<RocmBufferLayout*>(a.buffer().ptr());
    if (!rb || !rb->data) {
        return nullptr;
    }
    return static_cast<char*>(rb->data) + a.offset();
#else
    (void)a;
    return nullptr;
#endif
}

} // namespace mlx_lm
