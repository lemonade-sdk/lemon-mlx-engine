// Minimal test to isolate GDN non-determinism for Qwen3-Next
#include <mlx/mlx.h>
#include <iostream>
#include <cmath>
namespace mx = mlx::core;

static float max_diff(const mx::array& a, const mx::array& b) {
    auto d = mx::max(mx::abs(mx::subtract(
        mx::astype(mx::reshape(a, {-1}), mx::float32),
        mx::astype(mx::reshape(b, {-1}), mx::float32))));
    mx::eval(d);
    return d.item<float>();
}

static void test(const std::string& label, std::function<mx::array()> fn) {
    mx::synchronize();
    auto r0 = fn(); mx::eval(r0); mx::synchronize();
    auto r1 = fn(); mx::eval(r1); mx::synchronize();
    float md = max_diff(r0, r1);
    std::cerr << "  " << label << ": " << std::scientific << md
              << (md == 0.0f ? " OK" : " *** DIVERGES ***") << std::endl;
}

int main() {
    // Qwen3-Coder-Next: 16 key heads, 32 value heads, head_dim 256
    int B=1, Hk=16, Hv=32, Dk=256, Dv=256;

    for (int T : {1, 2, 3, 4}) {
        std::cerr << "T=" << T << ":" << std::endl;

        // Test 1: SSM step (core of gated delta)
        {
            auto state = mx::random::normal({B, Hv, Dv, Dk}, mx::bfloat16);
            auto q = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto k = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto v = mx::random::normal({B, Hv, Dv}, mx::bfloat16);
            auto g = mx::random::normal({B, Hv}, mx::bfloat16);
            auto beta = mx::random::normal({B, Hv}, mx::bfloat16);
            mx::eval({state, q, k, v, g, beta});
            test("ssm_step", [&]() {
                auto decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
                auto s = mx::multiply(state, decay);
                auto kv = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
                auto delta = mx::multiply(mx::subtract(v, kv), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
                return mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);
            });
        }

        // Test 2: Two SSM steps (sequential state dependency)
        if (T >= 2) {
            auto state = mx::random::normal({B, Hv, Dv, Dk}, mx::bfloat16);
            auto q0 = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto k0 = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto v0 = mx::random::normal({B, Hv, Dv}, mx::bfloat16);
            auto q1 = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto k1 = mx::random::normal({B, Hv, Dk}, mx::bfloat16);
            auto v1 = mx::random::normal({B, Hv, Dv}, mx::bfloat16);
            auto g = mx::random::normal({B, Hv}, mx::bfloat16);
            auto beta = mx::random::normal({B, Hv}, mx::bfloat16);
            mx::eval({state, q0, k0, v0, q1, k1, v1, g, beta});
            test("ssm_2step", [&]() {
                auto decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
                // Step 0
                auto s = mx::multiply(state, decay);
                auto kv = mx::sum(mx::multiply(s, mx::expand_dims(k0, -2)), -1);
                auto delta = mx::multiply(mx::subtract(v0, kv), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k0, -2), mx::expand_dims(delta, -1)));
                auto y0 = mx::sum(mx::multiply(s, mx::expand_dims(q0, -2)), -1);
                // Step 1 (depends on state from step 0)
                s = mx::multiply(s, decay);
                kv = mx::sum(mx::multiply(s, mx::expand_dims(k1, -2)), -1);
                delta = mx::multiply(mx::subtract(v1, kv), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k1, -2), mx::expand_dims(delta, -1)));
                auto y1 = mx::sum(mx::multiply(s, mx::expand_dims(q1, -2)), -1);
                return mx::concatenate({mx::reshape(y0, {-1}), mx::reshape(y1, {-1}), mx::reshape(s, {-1})});
            });
        }

        // Test 3: Large matmul (QKV projection size)
        {
            auto x = mx::random::normal({B, T, 2048}, mx::bfloat16);
            auto w = mx::random::normal({24576, 2048}, mx::bfloat16);
            mx::eval({x, w});
            test("qkvz_proj", [&]() { return mx::matmul(x, mx::transpose(w)); });
        }

        // Test 4: SDPA with this model's GQA config (16 Q, 2 KV, head_dim=256)
        {
            auto q = mx::random::normal({B, 16, T, 256}, mx::bfloat16);
            auto k = mx::random::normal({B, 2, T, 256}, mx::bfloat16);
            auto v = mx::random::normal({B, 2, T, 256}, mx::bfloat16);
            mx::eval({q, k, v});
            float scale = 1.0f / std::sqrt(256.0f);
            test("sdpa_gqa", [&]() {
                return mx::fast::scaled_dot_product_attention(q, k, v, scale, "causal");
            });
        }

        // Test 5: MoE softmax + argpartition over 512 experts
        {
            auto logits = mx::random::normal({B, T, 512}, mx::bfloat16);
            mx::eval(logits);
            test("moe_topk", [&]() {
                auto gates = mx::softmax(logits, -1);
                auto inds = mx::argpartition(gates, 502, -1);
                return mx::slice(inds, {0, 0, 502}, {B, T, 512});
            });
        }
    }

    std::cerr << "\n=== Done ===" << std::endl;
    return 0;
}
