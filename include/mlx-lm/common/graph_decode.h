// Copyright © 2025
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// Persistent device-position scalar for HIP-graph decode. The model's
// device-position RoPE and attention mask read this fixed-address [1] int32
// array; the decode loop updates its contents in place between graph replays so
// one captured graph advances through positions without re-capture.
mlx::core::array& graph_decode_pos();

// In-place device write of the absolute position (slice_update donates the
// buffer, so the address the captured graph baked stays valid).
void set_graph_decode_pos(int offset);

// When true, the model does NOT set the position itself — the decode loop owns
// it (set once before capture, bumped between replays). False on the plain
// eager path, where the model keeps it in sync each forward.
bool graph_external_pos();
void set_graph_external_pos(bool on);

} // namespace mlx_lm
