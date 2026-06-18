// HIP graph capture/replay mechanism probe (ROCm).
//
// Validates the capture-aware CommandEncoder path end to end:
//   1. begin_capture + mx::eval must NOT deadlock (the hostfunc/sync guards),
//   2. capture records kernels WITHOUT executing them (output VRAM stays stale),
//   3. replay (one hipGraphLaunch) executes them and writes the correct result.
//
// This is the foundation for graph-replayed decode. It does not yet exercise
// input mutation across replays (discrete-GPU host->VRAM coherence) or the
// model caches — those are the next milestones.

#include <mlx/mlx.h>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <tuple>
#include <unistd.h>

namespace mlx::core {
bool gpu_arena_begin(size_t capacity);
void gpu_arena_end();
bool gpu_graph_begin_capture();
bool gpu_graph_end_capture();
bool gpu_graph_replay();
bool gpu_graph_available();
void gpu_graph_reset();
} // namespace mlx::core

namespace mx = mlx::core;

int main() {
  setbuf(stdout, nullptr); // unbuffered so partial output survives a hang
  printf("start\n");
  mx::set_default_device(mx::Device::gpu);
  printf("device set\n");

  // The real decode loop captures on a non-default created stream
  // (generation_stream). Mimic that when PROBE_STREAM is set.
  if (getenv("PROBE_STREAM")) {
    auto s = mx::new_stream(mx::default_device());
    mx::set_default_stream(s);
    printf("using non-default stream (index %d)\n", s.index);
  }

  bool use_arena = getenv("PROBE_ARENA") != nullptr;
  if (use_arena) {
    if (!mx::gpu_arena_begin(4 * 1024 * 1024)) {
      printf("FAIL: arena_begin returned false\n");
      return 1;
    }
    printf("arena begun\n");
  } else {
    printf("arena SKIPPED (set PROBE_ARENA=1 to enable)\n");
  }

  const char* op = getenv("PROBE_OP");
  std::string opname = op ? op : "elem";

  // Fixed input.
  auto x = mx::array({1.0f, 2.0f, 3.0f, 4.0f});
  mx::eval(x);
  printf("x eval'd\n");

  int nops = 1;
  if (const char* e = getenv("PROBE_OPS")) nops = atoi(e);

  // Test op selection. Each `step` builds a small graph exercising one model op
  // type, to find which one breaks HIP graph replay on RDNA4.
  std::function<mlx::core::array()> step;
  if (opname == "qmv") {
    // quantized_matmul (the bulk of the model's decode kernels). PROBE_OPS sets
    // the matrix dim, so PROBE_OPS=4096 makes an ~8 MB weight => an individual
    // fine-grained allocation (>1 MB, like real model weights) instead of a slab
    // page, isolating the "large fine-grained buffer in a graph" case.
    int K = (nops > 1 ? nops : 512), N = K;
    auto xq = mx::ones({1, K});
    auto W = mx::ones({N, K});
    auto q = mx::quantize(W, 64, 4); // {wq, scales, biases}
    auto wq = q[0], sc = q[1], bs = q[2];
    mx::eval(xq, wq, sc, bs);
    step = [=]() {
      return mx::quantized_matmul(xq, wq, sc, bs, /*transpose=*/true, 64, 4);
    };
    printf("op=qmv (quantized_matmul %dx%d)\n", N, K);
  } else if (opname == "sort") {
    auto v = mx::random::uniform({1, 256});
    mx::eval(v);
    step = [=]() { return mx::argpartition(v, 248, -1); };
    printf("op=sort (argpartition over 256)\n");
  } else if (opname == "softmax") {
    auto v = mx::random::uniform({1, 256});
    mx::eval(v);
    step = [=]() { return mx::softmax(v, -1); };
    printf("op=softmax (256)\n");
  } else if (opname == "rmsnorm") {
    auto v = mx::ones({1, 256}), w = mx::ones({256});
    mx::eval(v, w);
    step = [=]() { return mx::fast::rms_norm(v, w, 1e-6f); };
    printf("op=rmsnorm\n");
  } else if (opname == "sdpa") {
    auto qd = mx::ones({1, 4, 1, 64}), kd = mx::ones({1, 4, 16, 64}),
         vd = mx::ones({1, 4, 16, 64});
    mx::eval(qd, kd, vd);
    step = [=]() {
      return mx::fast::scaled_dot_product_attention(qd, kd, vd, 0.125f, std::string());
    };
    printf("op=sdpa\n");
  } else if (opname == "parallel") {
    // Independent (parallel) branches in the graph DAG, then combine — the model
    // has many (q/k/v projections, 8 MoE experts) that the linear chains don't.
    auto xb = mx::ones({1, 256});
    auto q = mx::quantize(mx::ones({256, 256}), 64, 4);
    auto wq = q[0], sc = q[1], bs = q[2];
    mx::eval(xb, wq, sc, bs);
    step = [=]() {
      std::vector<mlx::core::array> outs;
      for (int i = 0; i < nops; i++)
        outs.push_back(mx::quantized_matmul(xb, wq, sc, bs, true, 64, 4));
      auto acc = outs[0];
      for (size_t i = 1; i < outs.size(); i++) acc = mx::add(acc, outs[i]);
      return acc;
    };
    printf("op=parallel (%d independent qmv branches)\n", nops);
  } else if (opname == "gather") {
    // gather_qmm — the MoE expert matmul (data-dependent expert selection).
    int E = 8, K = 256, N = 256;
    auto xg = mx::ones({1, 1, K});
    auto q = mx::quantize(mx::ones({E, N, K}), 64, 4);
    auto wq = q[0], sc = q[1], bs = q[2];
    auto rhs = mx::reshape(mx::arange(E, mx::uint32), {1, 1, E});
    mx::eval(xg, wq, sc, bs, rhs);
    step = [=]() {
      return mx::gather_qmm(xg, wq, sc, bs, std::nullopt, rhs, true, 64, 4,
                            "affine", false);
    };
    printf("op=gather (gather_qmm MoE, %d experts)\n", E);
  } else if (opname == "mixed") {
    // Big graph mixing op types at scale (like the model): qmv -> rmsnorm ->
    // silu, repeated. nops iterations ~ several hundred mixed nodes.
    auto v = mx::ones({1, 256}), w = mx::ones({256});
    auto q = mx::quantize(mx::ones({256, 256}), 64, 4);
    auto wq = q[0], sc = q[1], bs = q[2];
    mx::eval(v, w, wq, sc, bs);
    step = [=]() {
      auto y = v;
      for (int i = 0; i < nops; i++) {
        y = mx::quantized_matmul(y, wq, sc, bs, true, 64, 4);
        y = mx::fast::rms_norm(y, w, 1e-6f);
        y = mx::multiply(y, mx::sigmoid(y));
      }
      return y;
    };
    printf("op=mixed (%d x [qmv,rmsnorm,silu] ~ %d nodes)\n", nops, nops * 3);
  } else if (opname == "sliceupd") {
    // External persistent buffer modified in-graph (like the KV cache write).
    auto kv = mx::zeros({1, 64, 256});
    mx::eval(kv);
    step = [=]() {
      auto y = kv;
      for (int i = 0; i < nops; i++) {
        auto nk = mx::ones({1, 1, 256});
        y = mx::slice_update(y, nk, mx::Shape{0, i % 63, 0},
                             mx::Shape{1, (i % 63) + 1, 256});
      }
      return mx::sum(y);
    };
    printf("op=sliceupd (%d slice_updates into external buffer)\n", nops);
  } else if (opname == "rope") {
    auto qd = mx::ones({1, 4, 1, 128});
    mx::eval(qd);
    step = [=]() { return mx::fast::rope(qd, 128, false, 10000.0f, 1.0f, 0); };
    printf("op=rope\n");
  } else if (opname == "take") {
    int rows = (nops > 1 ? nops : 100000);
    auto table = mx::ones({rows, 256}); // embedding-like table
    auto idx = mx::zeros({1}, mx::uint32);
    mx::eval(table, idx);
    step = [=]() { return mx::take(table, idx, 0); };
    printf("op=take (gather from %d-row table)\n", rows);
  } else if (opname == "concat") {
    auto a = mx::ones({1, 3, 256}), b = mx::ones({1, 1, 256});
    mx::eval(a, b);
    step = [=]() { return mx::concatenate({a, b}, 1); };
    printf("op=concat\n");
  } else if (opname == "takealong") {
    auto v = mx::random::uniform({1, 1, 256});
    auto idx = mx::zeros({1, 1, 8}, mx::uint32);
    mx::eval(v, idx);
    step = [=]() { return mx::take_along_axis(v, idx, -1); };
    printf("op=takealong\n");
  } else if (opname == "strided") {
    // Strided (non-contiguous) elementwise -> the "general" binary kernel that
    // passes shape/stride metadata by device pointer.
    auto base = mx::ones({32, 64});
    mx::eval(base);
    step = [=]() {
      auto a = mx::transpose(base); // [64,32] strided view
      return mx::add(a, a);
    };
    printf("op=strided (transpose + add -> strided binary kernel)\n");
  } else if (opname == "embedta") {
    // Exact model embedding pattern: broadcast-index take_along_axis.
    auto table = mx::ones({1000, 256});
    auto tok = mx::zeros({1, 1}, mx::uint32);
    mx::eval(table, tok);
    step = [=]() {
      auto idx = mx::broadcast_to(mx::reshape(tok, {1, 1}), {1, 256});
      return mx::take_along_axis(table, idx, 0);
    };
    printf("op=embedta (broadcast-index take_along_axis, like the model)\n");
  } else if (opname == "compiled") {
    static auto comp = mx::compile(
        [](const std::vector<mlx::core::array>& in) -> std::vector<mlx::core::array> {
          return {mx::multiply(in[0], mx::sigmoid(in[0]))}; // silu
        },
        /*shapeless=*/true);
    auto v = mx::ones({1, 256});
    mx::eval(v);
    step = [=]() { return comp({v})[0]; };
    printf("op=compiled (silu via mx::compile)\n");
  } else {
    step = [=]() {
      auto y = x;
      for (int i = 0; i < nops; i++)
        y = mx::add(mx::multiply(y, mx::array(1.0f)), mx::array(0.0f)); // identity
      return y;
    };
    printf("op=elem nops=%d (~%d kernel nodes)\n", nops, 2 * nops);
  }

  // Warmup so any one-time compilation/allocation happens before capture.
  auto warm = step();
  mx::eval(warm);
  printf("warmup: %.1f %.1f %.1f %.1f (expect 4 7 10 13)\n",
         warm.data<float>()[0], warm.data<float>()[1],
         warm.data<float>()[2], warm.data<float>()[3]);

  // --- Capture ---
  printf("begin_capture...\n");
  fflush(stdout);
  mx::gpu_graph_begin_capture();
  auto yc = step();
  mx::eval(yc); // records kernels; must not hang with the capture-aware guards
  bool ok = mx::gpu_graph_end_capture();
  printf("end_capture ok=%d available=%d\n", ok, (int)mx::gpu_graph_available());
  if (!ok) {
    printf("FAIL: capture did not produce a graph\n");
    mx::gpu_arena_end();
    return 1;
  }

  // Output VRAM was only RECORDED, not executed — read it before replay.
  float before = yc.data<float>()[0];
  printf("yc[0] before replay = %.3f (recorded-not-executed; not yet 4)\n", before);

  // --- Replay: one hipGraphLaunch runs all recorded kernels ---
  bool rok = mx::gpu_graph_replay();
  printf("replay ok=%d\n", (int)rok);

  // data() forces a device sync + VRAM readback.
  float a0 = yc.data<float>()[0], a1 = yc.data<float>()[1],
        a2 = yc.data<float>()[2], a3 = yc.data<float>()[3];
  // identity chain => expect x = [1,2,3,4]
  printf("yc after replay = %.4f %.4f %.4f %.4f (expect 1 2 3 4)\n", a0, a1, a2, a3);

  bool correct = rok && a0 == 1 && a1 == 2 && a2 == 3 && a3 == 4;
  bool stale = rok && a0 == before; // never updated by replay
  printf("%s\n", correct ? "PASS: replay correct"
                         : (stale ? "STALE: replay ran but pointer values stale (#3887)"
                                  : "WRONG: replay produced unexpected values"));

  mx::gpu_graph_reset();
  mx::gpu_arena_end();
  // Skip C++/HIP exit handlers: the HIP runtime's fat-binary teardown
  // (__hipUnregisterFatBinary) hangs at process exit on this eGPU, unrelated to
  // the probe. The result is already validated above.
  fflush(stdout);
  _exit(correct ? 0 : 1);
}
