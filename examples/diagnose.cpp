// Diagnostic: Isolate SDPA GQA non-determinism + matmul broadcasting
// Tests the exact shapes used in Qwen3-8B attention with GQA

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

namespace mx = mlx::core;

static float max_diff(const mx::array& a, const mx::array& b) {
    auto diff = mx::abs(mx::subtract(
        mx::astype(mx::reshape(a, {-1}), mx::float32),
        mx::astype(mx::reshape(b, {-1}), mx::float32)));
    auto md = mx::max(diff);
    mx::eval(md);
    return md.item<float>();
}

static void test_det(const std::string& label, std::function<mx::array()> fn) {
    mx::synchronize();
    auto r0 = fn();
    mx::eval(r0);
    mx::synchronize();
    auto r1 = fn();
    mx::eval(r1);
    mx::synchronize();

    float md = max_diff(r0, r1);
    std::cerr << label << ": max_diff=" << std::scientific << md;
    if (md == 0.0f) std::cerr << " OK";
    else if (md < 1e-3f) std::cerr << " (minor)";
    else std::cerr << " *** NON-DETERMINISTIC ***";
    std::cerr << std::endl;
}

int main(int argc, char* argv[]) {
    // Qwen3-8B config
    int B = 1, n_q_heads = 32, n_kv_heads = 8, head_dim = 128;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::cerr << "========== SDPA GQA DETERMINISM TESTS ==========" << std::endl;

    // Create fixed random Q, K, V arrays matching Qwen3-8B attention shapes
    for (int L : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
        // Standard shapes (before GQA reshape)
        auto q = mx::random::normal({B, n_q_heads, L, head_dim}, mx::bfloat16);
        auto k = mx::random::normal({B, n_kv_heads, L, head_dim}, mx::bfloat16);
        auto v = mx::random::normal({B, n_kv_heads, L, head_dim}, mx::bfloat16);
        mx::eval({q, k, v});

        // Test 1: SDPA with causal mask
        test_det("  L=" + std::to_string(L) + " SDPA causal", [&]() {
            return mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal");
        });

        // Test 2: SDPA with no mask (for comparison)
        test_det("  L=" + std::to_string(L) + " SDPA none  ", [&]() {
            return mx::fast::scaled_dot_product_attention(q, k, v, scale, "");
        });
    }

    std::cerr << "\n========== MATMUL BROADCASTING (GQA pattern) ==========" << std::endl;

    for (int L : {1, 2, 3, 4, 5, 8}) {
        int n_repeats = n_q_heads / n_kv_heads; // = 4

        // Simulate SDPA GQA decomposition manually
        auto q = mx::random::normal({B, n_q_heads, L, head_dim}, mx::bfloat16);
        auto k = mx::random::normal({B, n_kv_heads, L, head_dim}, mx::bfloat16);
        auto v = mx::random::normal({B, n_kv_heads, L, head_dim}, mx::bfloat16);
        mx::eval({q, k, v});

        // GQA reshape (exactly what the SDPA fallback does)
        auto q_reshaped = mx::reshape(q, {B, n_kv_heads, n_repeats, L, head_dim});
        auto k_expanded = mx::expand_dims(k, 2); // [B, 8, 1, L, D]
        auto v_expanded = mx::expand_dims(v, 2); // [B, 8, 1, L, D]

        // Test: matmul with broadcasting (scores = Q @ K.T)
        auto k_transposed = mx::swapaxes(k_expanded, -1, -2); // [B, 8, 1, D, L]
        test_det("  L=" + std::to_string(L) + " GQA Q@K.T  ", [&]() {
            return mx::matmul(q_reshaped, k_transposed);
        });

        // Test: full GQA attention manually
        test_det("  L=" + std::to_string(L) + " GQA full   ", [&]() {
            auto scores = mx::matmul(q_reshaped, k_transposed);
            scores = mx::multiply(scores, mx::array(scale));
            // Simple causal mask
            if (L > 1) {
                auto mask = mx::triu(mx::full({L, L}, -1e9f), 1);
                scores = mx::add(scores, mask);
            }
            scores = mx::softmax(scores, -1);
            auto out = mx::matmul(scores, v_expanded);
            return mx::reshape(out, {B, n_q_heads, L, head_dim});
        });
    }

    std::cerr << "\n========== MATMUL NON-CONTIGUOUS INPUTS ==========" << std::endl;

    for (int L : {1, 5, 9}) {
        // Test matmul with contiguous inputs
        auto a = mx::random::normal({32, L, 128}, mx::bfloat16);
        auto b = mx::random::normal({32, 128, L}, mx::bfloat16);
        mx::eval({a, b});

        test_det("  L=" + std::to_string(L) + " contiguous ", [&]() {
            return mx::matmul(a, b);
        });

        // Test matmul with non-contiguous input (via transpose)
        auto c = mx::random::normal({32, 128, L}, mx::bfloat16);
        mx::eval(c);
        auto c_t = mx::transpose(c, {0, 2, 1}); // [32, L, 128] but non-contiguous

        test_det("  L=" + std::to_string(L) + " non-contig ", [&]() {
            return mx::matmul(c_t, b);
        });

        // Test matmul with broadcast dimension (GQA pattern)
        auto d = mx::random::normal({8, 4, L, 128}, mx::bfloat16);
        auto e = mx::expand_dims(mx::random::normal({8, L, 128}, mx::bfloat16), 1); // [8, 1, L, 128]
        mx::eval({d, e});
        auto e_t = mx::swapaxes(e, -1, -2); // [8, 1, 128, L]

        test_det("  L=" + std::to_string(L) + " broadcast  ", [&]() {
            return mx::matmul(d, e_t);
        });
    }

    std::cerr << "\n========== JIT COMPILED OP (SwiGLU pattern) ==========" << std::endl;

    for (int L : {1, 5, 9}) {
        auto gate = mx::random::normal({1, L, 12288}, mx::bfloat16);
        auto up = mx::random::normal({1, L, 12288}, mx::bfloat16);
        mx::eval({gate, up});

        // Test compiled SwiGLU
        static auto compiled_swiglu = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto silu_gate = mx::multiply(inputs[0], mx::sigmoid(inputs[0]));
                return {mx::multiply(silu_gate, inputs[1])};
            }, true);

        test_det("  L=" + std::to_string(L) + " SwiGLU     ", [&]() {
            return compiled_swiglu({gate, up})[0];
        });
    }

    // =====================================================
    // FULL MODEL TEST if model path provided
    // =====================================================
    if (argc >= 2) {
        std::cerr << "\n========== FULL MODEL FORWARD PASS ==========" << std::endl;
        auto ctx = mlx_lm::load_llm(argv[1]);
        std::cerr << "Model loaded." << std::endl;
        mlx_lm::GenerateParameters params;

        for (int L : {1, 2, 3, 4, 5, 6}) {
            std::vector<int> tokens(L, 785); // L copies of "The"
            test_det("  L=" + std::to_string(L) + " model fwd  ", [&]() {
                auto tok_arr = mx::array(tokens.data(), {1, L}, mx::int32);
                auto cache = ctx.new_cache_fn(params);
                return ctx.forward_fn(tok_arr, &cache);
            });
        }
    }

    std::cerr << "\n=== Diagnostic complete ===" << std::endl;
    return 0;
}
