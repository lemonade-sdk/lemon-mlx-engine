// Copyright © 2025
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// Persistent device-position scalar for HIP-graph decode (fixed-address [1] int32).
mlx::core::array& graph_decode_pos();

// In-place device write of the absolute position via a raw kernel.
void set_graph_decode_pos(int offset);

// Advance the device position in place by delta (loop-owned, between replays).
void advance_graph_decode_pos(int delta);

// When true, the decode loop owns the position; false on the eager path.
bool graph_external_pos();
void set_graph_external_pos(bool on);

// Ping-pong parity for the GDN recurrent state under build-once replay.
//   0 -> read state slots [0]/[1], write scratch slots [2]/[3]
//   1 -> read scratch slots [2]/[3], write state slots [0]/[1]
// Single-graph mode pins parity 0 (and the loop copies [2]->[0]); two-parity
// mode alternates 0/1 per token so the write of one relaunch is the read of the
// next — no copy. Default 0.
int graph_decode_parity();
void set_graph_decode_parity(int parity);

// Persistent device input-token buffer for build-once graph decode (fixed
// address [1,1] int32). The recorded graph's embedding gather reads this buffer;
// the loop feeds each new token into it (device copy) so the buffer is never
// reallocated between relaunches.
mlx::core::array& graph_decode_input();

// Device-copy the freshly-sampled token (a [1]/[1,1] int32 array) into the
// fixed-address input buffer. Runs on-stream, ordered after the producing step.
void set_graph_decode_input_from(mlx::core::array& token);

// Whether HIP-graph decode is active (ROCm only; false elsewhere).
bool graph_decode_enabled();

// True only while the single decode step is being captured (and during replay).
bool graph_capturing();
void set_graph_capturing(bool on);

} // namespace mlx_lm
