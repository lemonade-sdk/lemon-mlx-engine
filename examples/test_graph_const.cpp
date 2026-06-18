// Isolate the empty-graph cause: does creating a HOST-derived constant array
// mid-graph (mx::array(scalar), mx::arange) record under HIP-graph capture, or
// does it force eager execution (empty graph)? The device-pos decode path builds
// its mask from such constants every step; the normal forward never does.
#include <mlx/mlx.h>
#include <iostream>
#include <chrono>
namespace mx = mlx::core;
namespace mlx::core {
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
void gpu_graph_reset();
}

static double time_replays(int M) {
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < M; i++) mx::gpu_graph_replay();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / M;
}

int main() {
#if defined(MLX_BUILD_ROCM)
  int N = 4096;
  auto a = mx::random::normal({N}); mx::eval(a);

  auto run = [&](const char* name, auto build) {
    for (int w = 0; w < 4; ++w) { auto c = build(); mx::eval(c); }  // warm
    mx::gpu_graph_begin_capture();
    auto c = build();
    mx::eval(c);
    bool ok = mx::gpu_graph_end_capture();
    double ms = ok ? time_replays(30) : -1;
    mx::gpu_graph_reset();
    std::cout << name << ": capture ok=" << ok << "  replay=" << ms << " ms"
              << (ms > 0.01 ? "  RECORDS" : "  EMPTY (eager)") << "\n";
  };

  // Baseline: pure leaf-only compute (like the normal forward) — should record.
  run("leaf-only (a*a+a)", [&]() { return mx::add(mx::multiply(a, a), a); });
  // Host scalar created mid-graph (like mask's mx::array(0.0f)/(-inf)).
  run("host-scalar add ", [&]() { return mx::add(a, mx::array(1.0f)); });
  // arange created mid-graph (like the mask's arange(CAP)).
  run("arange+compare  ", [&]() {
    auto cols = mx::arange(0, N, mx::int32);
    auto m = mx::astype(mx::less(cols, mx::array(N / 2, mx::int32)), mx::float32);
    return mx::add(a, m);
  });
#else
  std::cout << "ROCm build required\n";
#endif
  return 0;
}
