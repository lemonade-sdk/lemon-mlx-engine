// Phase C: does a captured ATTENTION decode step replay correctly with an
// ADVANCING device position? Exercises the exact op sequence the model uses:
//   device-pos RoPE  +  in-place KV row write  +  device-pos masked SDPA.
// Inputs are fixed; only pos advances (driven by the in-place increment kernel),
// so RoPE phase varies the output per step. We compare captured-replay outputs
// against an eager reference. If they match, the attention graph is viable.
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <vector>
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
  const int Hq = 4, Hkv = 2, D = 64, CAP = 32, N = 8;
  const float scale = 1.0f / std::sqrt((float)D);
  const float kInf = std::numeric_limits<float>::infinity();
  auto strm = mx::default_stream(mx::default_device());

  // Fixed pre-RoPE inputs (deterministic).
  auto q0 = mx::random::normal({1, Hq, 1, D});
  auto k0 = mx::random::normal({1, Hkv, 1, D});
  auto v0 = mx::random::normal({1, Hkv, 1, D});
  mx::eval({q0, k0, v0});
  auto cols = mx::arange(0, CAP, mx::int32);
  mx::eval(cols);

  // One attention decode step into (KVk,KVv) at device position pos.
  // KV write uses standard mx::slice_update (DynamicSliceUpdate): a real MLX
  // primitive (so lazy inputs are scheduled correctly) whose write offset is
  // computed ON-DEVICE from pos, so it advances across replays. Only the pos++
  // accumulator needs the custom in-place kernel (slice_update freezes for that).
  auto step = [&](mx::array& KVk, mx::array& KVv, mx::array& pos) -> mx::array {
    auto q = mx::fast::rope(q0, D, false, 10000.0f, 1.0f, pos);
    auto k = mx::fast::rope(k0, D, false, 10000.0f, 1.0f, pos);
    KVk = mx::slice_update(KVk, k, pos, std::vector<int>{2});
    KVv = mx::slice_update(KVv, v0, pos, std::vector<int>{2});
    auto mask = mx::astype(
        mx::reshape(mx::where(mx::less_equal(cols, pos), mx::array(0.0f),
                              mx::array(-kInf)),
                    {1, 1, 1, CAP}),
        mx::float32);
    return mx::fast::scaled_dot_product_attention(q, KVk, KVv, scale, "", mask);
  };

  // --- Eager reference: pos = 0..N-1 ---
  std::vector<std::vector<float>> eager(N, std::vector<float>(Hq * D));
  {
    auto KVk = mx::zeros({1, Hkv, CAP, D});
    auto KVv = mx::zeros({1, Hkv, CAP, D});
    auto pos = mx::array({0}, {1}, mx::int32);
    mx::eval({KVk, KVv, pos});
    for (int t = 0; t < N; ++t) {
      auto out = step(KVk, KVv, pos);
      mx::eval(out);
      std::copy(out.data<float>(), out.data<float>() + Hq * D, eager[t].begin());
      mx::gpu_kv_pos_increment(pos, 1);
      mx::synchronize(strm);
    }
  }

  // --- Captured replay ---
  auto KVk = mx::zeros({1, Hkv, CAP, D});
  auto KVv = mx::zeros({1, Hkv, CAP, D});
  auto pos = mx::array({0}, {1}, mx::int32);
  mx::eval({KVk, KVv, pos});
  // Warm: compile all kernels (RoPE/SDPA JIT) before capture, then reset state.
  for (int w = 0; w < 3; ++w) {
    auto out = step(KVk, KVv, pos);
    mx::eval(out);
    mx::gpu_kv_pos_increment(pos, 1);
    mx::synchronize(strm);
  }
  KVk = mx::zeros({1, Hkv, CAP, D});
  KVv = mx::zeros({1, Hkv, CAP, D});
  mx::eval({KVk, KVv});
  mx::gpu_kv_pos_set(pos, 0);
  mx::synchronize(strm);

  // Capture the decode step ONLY (read pos). The loop owns pos: it is advanced
  // BETWEEN replays, not inside the graph — otherwise the in-graph increment
  // races the pos readers (mask/rope) as a write-after-read hazard.
  std::cerr << "begin_capture (rope + kv_write + masked sdpa)" << std::endl;
  mx::gpu_graph_begin_capture();
  auto out = step(KVk, KVv, pos);
  mx::eval(out);
  bool ok = mx::gpu_graph_end_capture();
  std::cerr << "end_capture ok=" << ok << std::endl;
  if (!ok) { mx::gpu_graph_reset(); std::cerr << "capture FAILED" << std::endl; return 1; }

  double max_err = 0.0;
  mx::gpu_kv_pos_set(pos, 0);  // start at position 0
  mx::synchronize(strm);
  for (int t = 0; t < N; ++t) {
    mx::gpu_graph_replay();      // executes the step at the current pos
    mx::gpu_kv_pos_increment(pos, 1);  // loop advances pos AFTER the replay
    mx::synchronize(strm);
    const float* g = out.data<float>();
    double e = 0.0;
    for (int i = 0; i < Hq * D; ++i)
      e = std::max(e, (double)std::abs(g[i] - eager[t][i]));
    max_err = std::max(max_err, e);
    // Diagnostics: how many KV rows are non-zero, and pos, after this replay.
    int kpos = pos.data<int>()[0];
    int nz = 0;
    const float* kk = KVk.data<float>();
    for (int r = 0; r < CAP; ++r) {
      bool any = false;
      for (int d = 0; d < D; ++d) if (kk[r * D + d] != 0.0f) { any = true; break; }
      if (any) nz++;
    }
    std::cerr << "  replay t=" << t << " err=" << e << " pos=" << kpos
              << " kv_nonzero_rows=" << nz
              << "  g[0..2]=" << g[0] << "," << g[1] << "," << g[2]
              << "  eager[0..2]=" << eager[t][0] << "," << eager[t][1] << ","
              << eager[t][2] << std::endl;
  }
  mx::gpu_graph_reset();

  bool pass = max_err < 1e-3;
  std::cerr << "overall max_err=" << max_err
            << (pass ? "   *** ATTENTION GRAPH REPLAY MATCHES EAGER ***"
                     : "   MISMATCH — a device-pos op did not advance on replay")
            << std::endl;
  return pass ? 0 : 2;
#else
  std::cout << "ROCm build required\n";
  return 0;
#endif
}
