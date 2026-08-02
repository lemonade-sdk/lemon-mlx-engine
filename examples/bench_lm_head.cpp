// Microbench: decode-shaped lm_head cost (quantized_matmul) on gfx1150.
// Field geometry: LemonMLXE Qwen3.6-35B-A3B-MTP-mlx-4bit
//   vocab=248320, hidden=2048, bits=4, group_size=64
//
// Usage:
//   ./bench_lm_head [path/to/lm_head_only.safetensors|model_dir]
//   Optional env: BENCH_ITERS=10  BENCH_WARM=3  BENCH_STAGE2=1
//
// Measures wall time of mx::quantized_matmul only (eval-to-eval). Does NOT
// claim full T₁ or gen t/s — those need a separate decode run.
//
// With BENCH_STAGE2=1 (default on): also times Design-C stage-2 for K-row
// subsets (take packed rows + qmm). Stage-1 algorithm cost is NOT included.

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

struct Stats {
    double mean = 0, mn = 0, mx = 0;
    int n = 0;
};

static Stats summarize(const std::vector<double>& samples) {
    Stats s;
    if (samples.empty()) return s;
    s.n = static_cast<int>(samples.size());
    s.mn = samples[0];
    s.mx = samples[0];
    double sum = 0;
    for (double v : samples) {
        sum += v;
        if (v < s.mn) s.mn = v;
        if (v > s.mx) s.mx = v;
    }
    s.mean = sum / samples.size();
    return s;
}

int main(int argc, char** argv) {
    const int vocab = 248320;
    const int hidden = 2048;
    const int bits = 4;
    const int group_size = 64;
    const int warm = env_int("BENCH_WARM", 3);
    const int iters = env_int("BENCH_ITERS", 10);
    // Default ON: stage-2 K sweep is the Design C fund gate this fire.
    const int do_stage2 = env_int("BENCH_STAGE2", 1);

    std::cout << "=== bench_lm_head ===\n";
    std::cout << "device_default note: MLX default device\n";
    std::cout << "geometry vocab=" << vocab << " hidden=" << hidden
              << " bits=" << bits << " group_size=" << group_size << "\n";
    std::cout << "warm=" << warm << " timed_iters=" << iters
              << " BENCH_STAGE2=" << do_stage2 << "\n";

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

    auto run_full = [&]() {
        auto y = mx::quantized_matmul(
            x, w_u32, scales, biases, /*transpose=*/true, group_size, bits);
        mx::eval(y);
        return y;
    };

    // Warm full
    for (int i = 0; i < warm; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto y = run_full();
        double ms = ms_since(t0);
        std::cout << "full_warm[" << i << "] wall_ms=" << ms
                  << " out_shape=" << y.shape() << "\n";
    }

    // Timed full
    std::vector<double> full_samples;
    full_samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto y = run_full();
        double ms = ms_since(t0);
        full_samples.push_back(ms);
        (void)y.size();
        std::cout << "full_iter[" << i << "] wall_ms=" << ms << "\n";
    }

    Stats full = summarize(full_samples);

    // Store bytes for traffic class note (not a bandwidth claim without time).
    const double store_mb =
        (double)(vocab * (hidden / 8) * 4 +  // u32 pack: hidden/8 u32 * 4 bytes
                 vocab * (hidden / group_size) * 2 * 2) /  // scales+biases bf16
        (1024.0 * 1024.0);

    std::cout << "\n=== SUMMARY FULL ===\n";
    std::cout << "source=" << source << "\n";
    std::cout << "qmm_mean_ms=" << full.mean << "\n";
    std::cout << "qmm_min_ms=" << full.mn << "\n";
    std::cout << "qmm_max_ms=" << full.mx << "\n";
    std::cout << "n_timed=" << full.n << "\n";
    std::cout << "approx_store_MiB_formula=" << store_mb << "\n";
    std::cout << "NOTE: isolated qmm only; T1 fraction requires decode gen t/s log.\n";

    // Design C stage-2: take K packed rows then qmm → [1,K] logits.
    // Indices are fixed arange(0,K) so we measure gather+qmm traffic, not
    // quality of shortlist. Stage-1 cost is NOT included.
    if (do_stage2) {
        // Product-plausible K values + extremes for scaling curve.
        const std::vector<int> ks = {256, 1024, 4096, 8192, 16384};
        const double fund_half = 0.5 * full.mean;  // Design C ≤0.5× full bar

        std::cout << "\n=== STAGE2 take+qmm (Design C gate; stage1 NOT included) ===\n";
        std::cout << "fund_half_ms=" << fund_half
                  << " (0.5 * full_mean; total stage1+stage2 must beat this)\n";

        for (int K : ks) {
            if (K > vocab) continue;
            // Contiguous first-K rows: best-case gather locality.
            auto idx = mx::arange(0, K, mx::int32);
            mx::eval(idx);

            auto run_s2 = [&]() {
                auto w_k = mx::take(w_u32, idx, /*axis=*/0);
                auto s_k = mx::take(scales, idx, /*axis=*/0);
                auto b_k = mx::take(biases, idx, /*axis=*/0);
                auto y = mx::quantized_matmul(
                    x, w_k, s_k, b_k, /*transpose=*/true, group_size, bits);
                mx::eval(y);
                return y;
            };

            // gather-only (no qmm) — host/device take cost
            auto run_gather = [&]() {
                auto w_k = mx::take(w_u32, idx, /*axis=*/0);
                auto s_k = mx::take(scales, idx, /*axis=*/0);
                auto b_k = mx::take(biases, idx, /*axis=*/0);
                mx::eval({w_k, s_k, b_k});
            };

            // Pre-warmed resident subset for qmm-only (after one gather)
            auto w_res = mx::take(w_u32, idx, 0);
            auto s_res = mx::take(scales, idx, 0);
            auto b_res = mx::take(biases, idx, 0);
            mx::eval({w_res, s_res, b_res});
            auto run_qmm_only = [&]() {
                auto y = mx::quantized_matmul(
                    x, w_res, s_res, b_res, /*transpose=*/true, group_size, bits);
                mx::eval(y);
                return y;
            };

            for (int i = 0; i < warm; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                auto y = run_s2();
                std::cout << "s2_K" << K << "_warm[" << i << "] wall_ms="
                          << ms_since(t0) << " out_shape=" << y.shape() << "\n";
            }

            std::vector<double> s2_all, s2_g, s2_q;
            s2_all.reserve(iters);
            s2_g.reserve(iters);
            s2_q.reserve(iters);
            for (int i = 0; i < iters; ++i) {
                auto t0 = std::chrono::steady_clock::now();
                auto y = run_s2();
                double ms = ms_since(t0);
                s2_all.push_back(ms);
                (void)y.size();
                std::cout << "s2_K" << K << "_iter[" << i << "] take_qmm_ms="
                          << ms << "\n";

                t0 = std::chrono::steady_clock::now();
                run_gather();
                double gms = ms_since(t0);
                s2_g.push_back(gms);

                t0 = std::chrono::steady_clock::now();
                auto yq = run_qmm_only();
                double qms = ms_since(t0);
                s2_q.push_back(qms);
                (void)yq.size();
                std::cout << "s2_K" << K << "_iter[" << i << "] gather_only_ms="
                          << gms << " qmm_only_ms=" << qms << "\n";
            }

            Stats a = summarize(s2_all);
            Stats g = summarize(s2_g);
            Stats q = summarize(s2_q);
            double vs_full = (full.mean > 0) ? (100.0 * a.mean / full.mean) : 0;
            double stage1_budget = fund_half - a.mean;

            std::cout << "s2_K" << K << "_SUMMARY take_qmm_mean_ms=" << a.mean
                      << " min=" << a.mn << " max=" << a.mx
                      << " pct_of_full=" << vs_full
                      << " gather_mean_ms=" << g.mean
                      << " qmm_only_mean_ms=" << q.mean
                      << " stage1_budget_to_half_ms=" << stage1_budget
                      << (stage1_budget > 0 ? " BUDGET_OK" : " BUDGET_NEG")
                      << "\n";
        }

        std::cout << "\nNOTE: stage2 only; if take+qmm already ≥ fund_half, "
                     "two-stage cannot fund without faster gather/qmm.\n";
        std::cout << "NOTE: stage1 (shortlist) cost is unmeasured; real total > stage2.\n";
        std::cout << "NOTE: no gen t/s claimed; no quality claim.\n";
    }

    return 0;
}
