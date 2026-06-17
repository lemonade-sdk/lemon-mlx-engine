// Copyright © 2025
#include "mlx-lm/common/graph_decode.h"

namespace mx = mlx::core;

namespace mlx_lm {

static bool g_external = false;

// Constructed lazily on first use, NOT at static-init time. Building it at static
// init (a global mx::array) forces a HIP stream to be created on the default GPU
// before main() runs — i.e. before --device selection — which both strands it on
// device 0 and, on a discrete GPU over TB5, intermittently hangs during process
// startup. A function-local static defers construction to runtime on the selected
// device.
mx::array& graph_decode_pos() {
    static mx::array* g_pos = nullptr;
    if (g_pos == nullptr) {
        g_pos = new mx::array(mx::zeros({1}, mx::int32));
        mx::eval(*g_pos);
    }
    return *g_pos;
}

void set_graph_decode_pos(int offset) {
    auto& p = graph_decode_pos();
    p = mx::slice_update(p, mx::broadcast_to(mx::array(offset, mx::int32), p.shape()),
                         mx::Shape(p.ndim(), 0), p.shape());
    mx::eval(p);
}

bool graph_external_pos() { return g_external; }
void set_graph_external_pos(bool on) { g_external = on; }

} // namespace mlx_lm
