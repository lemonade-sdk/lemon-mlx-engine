// Microbench: decode-shaped lm_head cost (quantized_matmul) on gfx1150.
// Field geometry: LemonMLXE Qwen3.6-35B-A3B-MTP-mlx-4bit
//   vocab=248320, hidden=2048, bits=4, group_size=64
//
// Usage:
//   ./bench_lm_head [path/to/lm_head_only.safetensors|model_dir]
//   Optional env: BENCH_ITERS=10  BENCH_WARM=3
//
// Measures wall time of mx::quantized_matmul only (eval-to-eval). Does NOT
// claim full T₁ or gen t/s — those need a separate decode run.

#include <mlx/mlx.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace mx = mlx::core;
namespace fs = std::filesystem;

static int env_int(const char* k, int def) {
    if (const char* v = std::getenv(k)) return std::atoi(v);
    return def;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    const int vocab = 248320;
    const int hidden = 2048;
    const int bits = 4;
    const int group_size = 64;
    const int warm = env_int("BENCH_WARM", 3);
    const int iters = env_int("BENCH_ITERS", 10);

    std::cout << "=== bench_lm_head ===\n";
    std::cout << "device_default note: MLX default device\n";
    std::cout << "geometry vocab=" << vocab << " hidden=" << hidden
              << " bits=" << bits << " group_size=" << group_size << "\n";
    std::cout << "warm=" << warm << " timed_iters=" << iters << "\n";

    mx::array w_u32 = mx::zeros({1}, mx::uint32);
    mx::array scales = mx::zeros({1}, mx::bfloat16);
    mx::array biases = mx::zeros({1}, mx::bfloat16);
    std::string source = "synthetic_quantize";

    if (argc >= 2) {
        fs::path p(argv[1]);
        fs::path st = p;
        if (fs::is_directory(p)) {
            // Prefer lm_head_only if present, else model.safetensors
            if (fs::exists(p / "lm_head_only.safetensors"))
                st = p / "lm_head_only.safetensors";
            else if (fs::exists(p / "model.safetensors"))
                st = p / "model.safetensors";
            else {
                std::cerr << "No safetensors in " << p << "\n";
                return 1;
            }
        }
        std::cout << "loading " << st << " ...\n";
        auto t_load0 = std::chrono::steady_clock::now();
        auto [weights, meta] = mx::load_safetensors(st.string());
        std::cout << "load wall_ms=" << ms_since(t_load0)
                  << " n_tensors=" << weights.size() << "\n";
        auto it_w = weights.find("lm_head.weight");
        auto it_s = weights.find("lm_head.scales");
        auto it_b = weights.find("lm_head.biases");
        if (it_w == weights.end() || it_s == weights.end()) {
            std::cerr << "missing lm_head.weight/scales in " << st << "\n";
            return 1;
        }
        w_u32 = it_w->second;
        scales = it_s->second;
        if (it_b != weights.end())
            biases = it_b->second;
        else
            biases = mx::zeros(scales.shape(), scales.dtype());
        source = st.string();
        mx::eval({w_u32, scales, biases});
        std::cout << "lm_head.weight shape=" << w_u32.shape()
                  << " dtype=" << w_u32.dtype() << "\n";
        std::cout << "lm_head.scales shape=" << scales.shape()
                  << " dtype=" << scales.dtype() << "\n";
        std::cout << "lm_head.biases shape=" << biases.shape()
                  << " dtype=" << biases.dtype() << "\n";
    } else {
        // Same geometry as field package; values random — kernel traffic class only.
        std::cout << "no path: synthesizing BF16 W then quantize (shape-matched)\n";
        auto W = mx::astype(mx::random::normal({vocab, hidden}, mx::float32), mx::bfloat16);
        mx::eval(W);
        auto q = mx::quantize(W, group_size, bits);
        w_u32 = q[0];
        scales = q[1];
        biases = q[2];
        mx::eval({w_u32, scales, biases});
        std::cout << "synthetic weight shape=" << w_u32.shape() << "\n";
    }

    // Decode T=1 activation: [1, hidden] BF16 (matches linear_fwd path).
    auto x = mx::astype(mx::random::normal({1, hidden}, mx::float32), mx::bfloat16);
    mx::eval(x);

    auto run_once = [&]() {
        auto y = mx::quantized_matmul(
            x, w_u32, scales, biases, /*transpose=*/true, group_size, bits);
        mx::eval(y);
        return y;
    };

    // Warm
    for (int i = 0; i < warm; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto y = run_once();
        double ms = ms_since(t0);
        std::cout << "warm[" << i << "] wall_ms=" << ms
                  << " out_shape=" << y.shape() << "\n";
    }

    // Timed
    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto y = run_once();
        double ms = ms_since(t0);
        samples.push_back(ms);
        // touch result so optimizer cannot DCE (already eval'd)
        (void)y.size();
        std::cout << "iter[" << i << "] wall_ms=" << ms << "\n";
    }

    double sum = 0, mn = samples[0], mxv = samples[0];
    for (double v : samples) {
        sum += v;
        if (v < mn) mn = v;
        if (v > mxv) mxv = v;
    }
    double mean = sum / samples.size();

    // Store bytes for traffic class note (not a bandwidth claim without time).
    const double store_mb =
        (double)(vocab * (hidden / 8) * 4 +  // u32 pack: hidden/8 u32 * 4 bytes
                 vocab * (hidden / group_size) * 2 * 2) /  // scales+biases bf16
        (1024.0 * 1024.0);

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "source=" << source << "\n";
    std::cout << "qmm_mean_ms=" << mean << "\n";
    std::cout << "qmm_min_ms=" << mn << "\n";
    std::cout << "qmm_max_ms=" << mxv << "\n";
    std::cout << "n_timed=" << samples.size() << "\n";
    std::cout << "approx_store_MiB_formula=" << store_mb << "\n";
    std::cout << "NOTE: isolated qmm only; T1 fraction requires decode gen t/s log.\n";
    return 0;
}
