// Decisive de-risk: does a REAL in-place device kernel accumulate across HIP-graph
// replays? MLX functional ops (slice_update donation) do NOT — end_capture freezes
// their host-constant uploads, so pos+1 bakes at capture. A raw kernel (pos[0]+=1)
// has no host upload. If final_pos==N, in-place kernels unblock graph-replay decode.
#include <mlx/mlx.h>
#include <iostream>
namespace mx = mlx::core;

namespace mlx::core {
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
void gpu_graph_reset();
void gpu_kv_pos_increment(array& pos, int delta);
void gpu_kv_pos_set(array& pos, int v);
void gpu_kv_row_write(array& kv, const array& row, const array& pos);
}

int main() {
#if defined(MLX_BUILD_ROCM)
  auto strm = mx::default_stream(mx::default_device());
  auto pos = mx::array({0}, {1}, mx::int32);
  mx::eval(pos);
  std::cerr << "[1] pos init = " << pos.data<int>()[0] << std::endl;

  // --- Stage 1: does the in-place kernel work EAGERLY? ---
  mx::gpu_kv_pos_increment(pos, 1);
  std::cerr << "[2] after 1 eager inc (launched)" << std::endl;
  mx::synchronize(strm);
  std::cerr << "[3] eager inc value = " << pos.data<int>()[0]
            << " (expect 1)" << std::endl;

  mx::gpu_kv_pos_increment(pos, 1);
  mx::gpu_kv_pos_increment(pos, 1);
  mx::synchronize(strm);
  std::cerr << "[4] after 2 more eager inc = " << pos.data<int>()[0]
            << " (expect 3)" << std::endl;

  mx::gpu_kv_pos_set(pos, 0);
  mx::synchronize(strm);
  std::cerr << "[5] after set 0 = " << pos.data<int>()[0] << std::endl;

  // --- Stage 2: capture + replay ---
  std::cerr << "[6] begin_capture" << std::endl;
  mx::gpu_graph_begin_capture();
  mx::gpu_kv_pos_increment(pos, 1);
  bool ok = mx::gpu_graph_end_capture();
  std::cerr << "[7] end_capture ok=" << ok << std::endl;
  if (!ok) { mx::gpu_graph_reset(); std::cerr << "capture FAILED" << std::endl; return 1; }

  // Capture RECORDS (does not execute), so pos is still 0 here. Each replay
  // executes the recorded increment exactly once: N replays -> pos == N.
  const int N = 8;
  for (int i = 0; i < N; ++i) mx::gpu_graph_replay();
  mx::synchronize(strm);
  mx::gpu_graph_reset();

  int fp = pos.data<int>()[0];
  std::cerr << "after " << N << " replays: final_pos=" << fp
            << (fp == N ? "   *** COUNTER ACCUMULATES ***" : "   counter FAIL")
            << std::endl;
  bool pass = (fp == N);

  // --- Stage 3: in-place KV row-write advances across replays ---
  int B = 1, H = 2, CAP = 16, D = 4;
  auto KV = mx::zeros({B, H, CAP, D}, mx::float32);
  auto row = mx::ones({B, H, 1, D}, mx::float32);
  auto kpos = mx::array({0}, {1}, mx::int32);
  mx::eval({KV, row, kpos});
  // warm (stream/alloc), then reset to a clean fixed buffer.
  mx::gpu_kv_row_write(KV, row, kpos);
  mx::gpu_kv_pos_increment(kpos, 1);
  mx::synchronize(strm);
  KV = mx::zeros({B, H, CAP, D}, mx::float32);
  mx::eval(KV);
  mx::gpu_kv_pos_set(kpos, 0);
  mx::synchronize(strm);

  std::cerr << "[kv] begin_capture (write row @ pos, then pos++)" << std::endl;
  mx::gpu_graph_begin_capture();
  mx::gpu_kv_row_write(KV, row, kpos);
  mx::gpu_kv_pos_increment(kpos, 1);
  bool kok = mx::gpu_graph_end_capture();
  std::cerr << "[kv] end_capture ok=" << kok << std::endl;
  if (kok) {
    const int M = 6;
    for (int i = 0; i < M; ++i) mx::gpu_graph_replay();
    mx::synchronize(strm);
    mx::gpu_graph_reset();
    int marked = 0;
    for (int t = 0; t < CAP; ++t)
      if (KV.data<float>()[t * D] == 1.0f) marked++;  // row h=0,d=0 of slot t
    int kp = kpos.data<int>()[0];
    bool kv_ok = (marked == M && kp == M);
    std::cerr << "after " << M << " replays: marked_rows=" << marked
              << " kpos=" << kp
              << (kv_ok ? "   *** KV WRITE ADVANCES across replays ***"
                        : "   KV write does NOT advance")
              << std::endl;
    pass = pass && kv_ok;
  } else {
    mx::gpu_graph_reset();
    pass = false;
  }
  return pass ? 0 : 2;
#else
  std::cout << "ROCm build required\n";
  return 0;
#endif
}
