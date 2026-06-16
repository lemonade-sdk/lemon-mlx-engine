// Copyright © 2025
#include "mlx-lm/common/graph_decode.h"

namespace mx = mlx::core;

namespace mlx_lm {

static mx::array g_pos = mx::zeros({1}, mx::int32);
static bool g_pos_init = false;
static bool g_external = false;

mx::array& graph_decode_pos() {
    if (!g_pos_init) {
        mx::eval(g_pos);
        g_pos_init = true;
    }
    return g_pos;
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
