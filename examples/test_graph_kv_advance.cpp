// CRITICAL de-risk for graph-replayable decode: capture a "decode step" that
// (a) writes a marker into a fixed KV buffer at a DEVICE position, and
// (b) bumps that position by 1 — then REPLAY it N times. If the dynamic
// slice_update reads the device position at REPLAY time (not capture time), the
// writes advance and KV[0:N] become marked while KV[N:] stay zero. If the
// position is baked, every replay overwrites the same slot. This decides whether
// one static graph can follow a growing KV.
#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;

namespace mlx::core {
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
void gpu_graph_reset();
bool gpu_graph_available();
}

int main() {
#if defined(MLX_BUILD_ROCM)
  int CAP = 16, D = 4;
  auto KV = mx::zeros({1, 1, CAP, D});
  auto pos = mx::array({0}, {1}, mx::int32);
  mx::eval({KV, pos});

  auto one_row = mx::ones({1, 1, 1, D});
  mx::eval(one_row);

  // One decode step. Write the marker at the device position with a ONE-HOT
  // where (arange==pos ? new : KV) — this uses only standard elementwise kernels
  // that capture cleanly, AVOIDING the JIT compute_dynamic_offset that dynamic
  // slice_update needs (that JIT module launch hangs under stream capture). Bump
  // pos with a BAKED slice_update (Shape args, not the dynamic array-start form).
  auto step = [&]() {
    auto cols = mx::arange(0, CAP, mx::int32);                 // [CAP]
    auto sel = mx::reshape(mx::equal(cols, pos), {1, 1, CAP, 1});  // true at pos
    KV = mx::where(sel, mx::broadcast_to(one_row, {1, 1, CAP, D}), KV);
    auto next = mx::add(pos, mx::array(1, mx::int32));
    pos = mx::slice_update(pos, next, mx::Shape{0}, mx::Shape{1});
  };

  // Warm the allocator pools (capture can't hipMalloc mid-stream).
  for (int w = 0; w < 4; ++w) { step(); mx::eval({KV, pos}); }
  // Reset so the count starts clean for the replay check.
  KV = mx::zeros({1, 1, CAP, D});
  pos = mx::array({0}, {1}, mx::int32);
  mx::eval({KV, pos});

  std::cout << "begin_capture\n";
  mx::gpu_graph_begin_capture();
  step();
  mx::eval({KV, pos});
  bool ok = mx::gpu_graph_end_capture();
  std::cout << "end_capture ok=" << ok << "\n";
  if (!ok) { mx::gpu_graph_reset(); std::cout << "capture FAILED\n"; return 1; }

  const int N = 6;
  for (int i = 1; i < N; ++i) mx::gpu_graph_replay();  // step 0 captured + N-1 replays
  mx::eval({KV, pos});
  mx::gpu_graph_reset();

  // Count marked rows: KV[0,0,t,0] == 1 means written.
  int marked = 0;
  for (int t = 0; t < CAP; ++t)
    if (KV.data<float>()[t * D] == 1.0f) marked++;
  int final_pos = pos.data<int>()[0];
  std::cout << "after " << N << " steps: marked_rows=" << marked
            << "  final_pos=" << final_pos
            << ((marked == N && final_pos == N)
                    ? "   *** ADVANCES — 2x graph decode is viable ***"
                    : "   baked (position did not advance on replay)")
            << "\n";
#else
  std::cout << "ROCm build required\n";
#endif
  return 0;
}
