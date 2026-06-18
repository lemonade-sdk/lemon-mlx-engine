// Minimal isolation: does an in-place device counter accumulate across HIP-graph
// replays? pos = slice_update(pos, pos+1) — slice_update donates in place when
// refcount==1 (the same mechanism the KV cache + MLX_DECODE_GRAPH harness use).
// If this prints final_pos==N, the donation loop survives replay and option A
// (no new kernel) is viable for the device KV position.
#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;
namespace mlx::core {
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
void gpu_graph_reset();
}

int main() {
#if defined(MLX_BUILD_ROCM)
  auto pos = mx::array({0}, {1}, mx::int32);
  mx::eval(pos);
  auto one = mx::array(1, mx::int32);
  mx::eval(one);

  auto bump = [&]() {
    pos = mx::slice_update(pos, mx::add(pos, one), mx::Shape{0}, mx::Shape{1});
  };
  // warm
  for (int w = 0; w < 4; ++w) { bump(); mx::eval(pos); }
  pos = mx::array({0}, {1}, mx::int32); mx::eval(pos);

  mx::gpu_graph_begin_capture();
  bump();
  mx::async_eval(pos);
  bool ok = mx::gpu_graph_end_capture();
  std::cout << "counter capture ok=" << ok << "\n";
  if (!ok) { mx::gpu_graph_reset(); return 1; }

  const int N = 5;
  for (int i = 1; i < N; ++i) mx::gpu_graph_replay();
  mx::eval(pos);
  mx::gpu_graph_reset();
  int fp = pos.data<int>()[0];
  std::cout << "after " << N << " steps: final_pos=" << fp
            << (fp == N ? "   *** ACCUMULATES (donation loop survives replay) ***"
                        : "   does NOT accumulate (need a real in-place kernel)")
            << "\n";
#else
  std::cout << "ROCm build required\n";
#endif
  return 0;
}
