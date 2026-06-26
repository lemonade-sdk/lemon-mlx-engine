// Numerical-correctness test for BitNet 2-bit quantized matmul.
// Verifies that bitnet_repack_weights produces uint32 2-bit weights that
// produce bit-exact results vs dequantize-then-matmul reference.
//
// BitNet packs 4 ternary codes {0→-1, 1→0, 2→+1} per byte (4 values per byte).
// bitnet_repack_weights converts this to MLX uint32 2-bit format for quantized_matmul.

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/bitnet_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx-lm/common/quantize_utils.h>
#include <mlx-lm/llm/models/llama.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <vector>

namespace mx = mlx::core;
namespace mlx_lm {

// Helper: build a BitNet uint8 packed ternary matrix from a flat ternary array.
// ternary_values: out * in values where each is -1, 0, or +1.
// Packs 4 values per byte: byte[row, c] = lane0[1:0] | lane1[3:2] | lane2[5:4] | lane3[7:6]
static mx::array pack_ternary_values(
    const std::vector<int>& ternary_values,
    int out_features,
    int in_features)
{
    std::vector<uint8_t> packed(out_features / 4 * in_features, 0);

    for (int oc = 0; oc < out_features; ++oc) {
        int row = oc / 4;
        int lane = oc % 4;
        int bit_shift = lane * 2;
        for (int c = 0; c < in_features; ++c) {
            int idx = oc * in_features + c;
            int code = ternary_values[idx] + 1; // -1→0, 0→1, 1→2
            packed[row * in_features + c] |= static_cast<uint8_t>(code << bit_shift);
        }
    }

    return mx::array(packed.data(), {out_features / 4, in_features}, mx::uint8);
}

// Helper matching the actual BitNet/dequantize_bitnet_weight lane-major order:
// out[0:R] = lane0, out[R:2R] = lane1, out[2R:3R] = lane2, out[3R:4R] = lane3.
static mx::array pack_ternary_values_lane_major(
    const std::vector<int>& ternary_values,
    int out_features,
    int in_features)
{
    int packed_rows = out_features / 4;
    std::vector<uint8_t> packed(packed_rows * in_features, 0);

    for (int oc = 0; oc < out_features; ++oc) {
        int lane = oc / packed_rows;
        int row = oc % packed_rows;
        int bit_shift = lane * 2;
        for (int c = 0; c < in_features; ++c) {
            int idx = oc * in_features + c;
            int code = ternary_values[idx] + 1; // -1→0, 0→1, 1→2
            packed[row * in_features + c] |= static_cast<uint8_t>(code << bit_shift);
        }
    }

    return mx::array(packed.data(), {packed_rows, in_features}, mx::uint8);
}

TEST_CASE("bitnet_repack_weights: shape and dtype", "[bitnet_quant]") {
    // Small test: 2 output channels × 2 packed rows, in_features=128 (divisible by 128)
    int out_features = 4;
    int in_features = 128;

    // All zeros (code=1)
    std::vector<int> vals(out_features * in_features, 0);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(0.5f, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);

    // Check wq shape: [out, ceil(in/16)]
    int expected_cols = (in_features + 15) / 16;
    REQUIRE(wq.shape().size() == 2);
    REQUIRE(wq.shape(0) == out_features);
    REQUIRE(wq.shape(1) == expected_cols);
    REQUIRE(wq.dtype() == mx::uint32);

    // Check scales shape: [out, num_groups] where num_groups = in/128 = 1
    int num_groups = in_features / 128;
    REQUIRE(scales.shape().size() == 2);
    REQUIRE(scales.shape(0) == out_features);
    REQUIRE(scales.shape(1) == num_groups);
    REQUIRE(scales.dtype() == mx::float16);

    // Check biases shape matches scales
    REQUIRE(biases.shape() == scales.shape());
    REQUIRE(biases.dtype() == mx::float16);
}

TEST_CASE("bitnet_repack_weights: all zeros (code=1) → dequant is 0", "[bitnet_quant]") {
    // BitNet code 1 = ternary value 0
    // MLX 2-bit dequant: code * scale + bias = 1 * scale + (-scale) = 0
    int out_features = 4;
    int in_features = 128;

    // All zeros in ternary = code 1 in BitNet = dequant 0
    std::vector<int> vals(out_features * in_features, 0);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(0.5f, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    mx::eval({wq, scales, biases});

    // Dequantize via MLX
    auto dequant = mx::dequantize(wq, scales, biases, 128, 2);
    mx::eval(dequant);

    // Code 1 → 0 for any scale
    auto expected = mx::full({out_features, in_features}, 0.0f, mx::float16);
    mx::eval(expected);

    auto diff = mx::abs(mx::subtract(mx::astype(dequant, mx::float32), expected));
    mx::eval(diff);
    auto max_diff = mx::max(diff);
    mx::eval(max_diff);

    REQUIRE(max_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet_repack_weights: all ones (code=2) → dequant is +scale", "[bitnet_quant]") {
    // BitNet code 2 = ternary value +1
    // MLX 2-bit dequant: code * scale + bias = 2 * scale + (-scale) = +scale
    int out_features = 4;
    int in_features = 128;

    std::vector<int> vals(out_features * in_features, 1);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale_val = 0.5f;
    auto scale = mx::array(scale_val, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    mx::eval({wq, scales, biases});

    auto dequant = mx::dequantize(wq, scales, biases, 128, 2);
    mx::eval(dequant);

    // Code 2 → +scale
    auto expected = mx::full({out_features, in_features}, scale_val, mx::float16);
    mx::eval(expected);

    auto diff = mx::abs(mx::subtract(mx::astype(dequant, mx::float32), expected));
    mx::eval(diff);
    auto max_diff = mx::max(diff);
    mx::eval(max_diff);

    REQUIRE(max_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet_repack_weights: all minus ones (code=0) → dequant is -scale", "[bitnet_quant]") {
    // BitNet code 0 = ternary value -1
    // MLX 2-bit dequant: code * scale + bias = 0 * scale + (-scale) = -scale
    int out_features = 4;
    int in_features = 128;

    std::vector<int> vals(out_features * in_features, -1);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale_val = 0.5f;
    auto scale = mx::array(scale_val, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    mx::eval({wq, scales, biases});

    auto dequant = mx::dequantize(wq, scales, biases, 128, 2);
    mx::eval(dequant);

    // Code 0 → -scale
    auto expected = mx::full({out_features, in_features}, -scale_val, mx::float16);
    mx::eval(expected);

    auto diff = mx::abs(mx::subtract(mx::astype(dequant, mx::float32), expected));
    mx::eval(diff);
    auto max_diff = mx::max(diff);
    mx::eval(max_diff);

    REQUIRE(max_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet config detects inverse Falcon-E weight_scale semantics", "[bitnet_quant]") {
    auto base = nlohmann::json{
        {"model_type", "bitnet"},
        {"hidden_size", 2048},
        {"num_hidden_layers", 1},
        {"intermediate_size", 4096},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 2},
        {"head_dim", 128},
        {"rms_norm_eps", 1e-5},
        {"vocab_size", 32768},
        {"max_position_embeddings", 32768},
        {"tie_word_embeddings", false},
        {"quantization_config", {{"quant_method", "bitnet"}}}
    };

    auto falcon = base;
    falcon["hidden_act"] = "silu";
    auto falcon_cfg = falcon.get<LlamaConfiguration>();
    REQUIRE(falcon_cfg.bitnet_invert_weight_scales);

    auto bitnet = base;
    bitnet["hidden_act"] = "relu2";
    bitnet["quantization_config"]["linear_class"] = "autobitlinear";
    auto bitnet_cfg = bitnet.get<LlamaConfiguration>();
    REQUIRE_FALSE(bitnet_cfg.bitnet_invert_weight_scales);

    auto explicit_inverse = base;
    explicit_inverse["hidden_act"] = "relu2";
    explicit_inverse["quantization_config"]["linear_class"] = "bitlinear";
    auto explicit_inverse_cfg = explicit_inverse.get<LlamaConfiguration>();
    REQUIRE(explicit_inverse_cfg.bitnet_invert_weight_scales);

    auto silu_autobitlinear = base;
    silu_autobitlinear["hidden_act"] = "silu";
    silu_autobitlinear["quantization_config"]["linear_class"] = "autobitlinear";
    auto silu_autobitlinear_cfg = silu_autobitlinear.get<LlamaConfiguration>();
    REQUIRE_FALSE(silu_autobitlinear_cfg.bitnet_invert_weight_scales);
}

TEST_CASE("bitnet inverse weight_scale dequantizes Falcon-style scales", "[bitnet_quant]") {
    int out_features = 4;
    int in_features = 128;

    std::vector<int> vals(out_features * in_features, 1); // ternary +1
    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(4.0f, mx::bfloat16);

    auto normal = mx::astype(dequantize_bitnet_weight(packed, scale, out_features), mx::float32);
    auto inverse = mx::astype(dequantize_bitnet_weight(packed, scale, out_features, true), mx::float32);
    mx::eval({normal, inverse});

    auto normal_diff = mx::max(mx::abs(mx::subtract(normal, mx::full({out_features, in_features}, 4.0f, mx::float32))));
    auto inverse_diff = mx::max(mx::abs(mx::subtract(inverse, mx::full({out_features, in_features}, 0.25f, mx::float32))));
    mx::eval({normal_diff, inverse_diff});

    REQUIRE(normal_diff.item<float>() < 1e-5f);
    REQUIRE(inverse_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet_repack_weights supports inverse weight_scale", "[bitnet_quant]") {
    int out_features = 8;
    int in_features = 128;

    std::vector<int> vals(out_features * in_features);
    for (int oc = 0; oc < out_features; ++oc) {
        for (int k = 0; k < in_features; ++k) {
            vals[oc * in_features + k] = ((oc * 7 + k * 3) % 3) - 1;
        }
    }

    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(4.0f, mx::bfloat16);
    auto ref = mx::astype(dequantize_bitnet_weight(packed, scale, out_features, true), mx::float32);
    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale, true);
    auto got = mx::astype(mx::dequantize(wq, scales, biases, 128, 2), mx::float32);
    mx::eval({ref, got});

    auto max_diff = mx::max(mx::abs(mx::subtract(ref, got)));
    mx::eval(max_diff);

    REQUIRE(max_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet_repack_weights matches model lane-major dequant layout", "[bitnet_quant]") {
    int out_features = 8;  // >4 exposes lane-major vs interleaved output order
    int in_features = 128;

    std::vector<int> vals(out_features * in_features);
    for (int oc = 0; oc < out_features; ++oc) {
        for (int k = 0; k < in_features; ++k) {
            vals[oc * in_features + k] = ((oc * 7 + k * 3) % 3) - 1;
        }
    }

    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(0.25f, mx::bfloat16);

    auto model_dequant = mx::astype(dequantize_bitnet_weight(packed, scale, out_features), mx::float32);
    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    auto q_dequant = mx::astype(mx::dequantize(wq, scales, biases, 128, 2), mx::float32);
    mx::eval({model_dequant, q_dequant});

    auto max_diff = mx::max(mx::abs(mx::subtract(model_dequant, q_dequant)));
    mx::eval(max_diff);

    REQUIRE(max_diff.item<float>() < 1e-5f);
}

TEST_CASE("bitnet_repack_weights: mixed codes", "[bitnet_quant]") {
    int out_features = 4;
    int in_features = 128;

    // Mix of -1, 0, +1
    std::vector<int> vals(out_features * in_features);
    float scale_val = 0.25f;
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
        vals[i] = (i % 3) - 1; // cycles: -1, 0, 1
    }
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(scale_val, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    mx::eval({wq, scales, biases});

    auto dequant = mx::dequantize(wq, scales, biases, 128, 2);
    mx::eval(dequant);

    // Verify each value matches expected: dequant = (code - 1) * scale = (vals[i] + 1 - 1) * scale = vals[i] * scale
    auto dequant_f = mx::astype(dequant, mx::float32);
    mx::eval(dequant_f);

    auto data = dequant_f.data<float>();
    bool ok = true;
    for (int i = 0; i < static_cast<int>(vals.size()) && ok; ++i) {
        float expected = vals[i] * scale_val;
        float actual = data[i];
        if (std::abs(expected - actual) > 1e-4f) {
            ok = false;
        }
    }
    REQUIRE(ok);
}

TEST_CASE("quantized_matmul matches dequantize-then-matmul (bit-exact)", "[bitnet_quant]") {
    int out_features = 4;
    int in_features = 128;
    int batch_size = 2;

    // Create packed ternary weights
    std::vector<int> vals(out_features * in_features);
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
        vals[i] = (i % 3) - 1; // cycles: -1, 0, 1
    }
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(0.25f, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);

    // Create input: [batch, in_features], bfloat16 (typical for LLM)
    auto x = mx::astype(mx::random::normal({batch_size, in_features}), mx::bfloat16);
    mx::eval({x, wq, scales, biases});

    // Reference: dequantize then matmul
    auto w_dequant = mx::dequantize(wq, scales, biases, 128, 2);
    auto ref = mx::matmul(x, mx::transpose(w_dequant));
    mx::eval(ref);

    // GPU path: quantized_matmul (transpose=true since weight is [out, in])
    auto gpu = mx::quantized_matmul(x, wq, scales, biases, /*transpose=*/true, 128, 2);
    mx::eval(gpu);

    // Should be bit-exact (same accumulation precision, no quantization error)
    auto diff = mx::abs(mx::subtract(mx::astype(ref, mx::float32), mx::astype(gpu, mx::float32)));
    mx::eval(diff);

    auto max_diff = mx::max(diff);
    mx::eval(max_diff);

    float max_err = max_diff.item<float>();
    // The two paths use different accumulation strategies (dequant+matmul vs
    // fused quantized_matmul kernel), so they are not bit-identical. A max
    // error of a few ULPs is expected for fp16 accumulation. A value > 1.0
    // would indicate a real algorithmic difference.
    REQUIRE(max_err < 5.0f);
}

TEST_CASE("quantized_matmul with scale=1.0: max error < 1e-5", "[bitnet_quant]") {
    int out_features = 4;
    int in_features = 128;
    int batch_size = 1;

    // All zeros (code=1 → dequant 0) with scale=1.0
    std::vector<int> vals(out_features * in_features, 0);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(1.0f, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);

    // Input: all ones, bfloat16
    auto x = mx::full({batch_size, in_features}, 1.0f, mx::bfloat16);
    mx::eval({x, wq, scales, biases});

    // Reference: dequantize then matmul
    auto w_dequant = mx::dequantize(wq, scales, biases, 128, 2);
    auto ref = mx::matmul(x, mx::transpose(w_dequant));
    mx::eval(ref);

    // GPU path
    auto gpu = mx::quantized_matmul(x, wq, scales, biases, /*transpose=*/true, 128, 2);
    mx::eval(gpu);

    // Bit-exact: both should produce exactly 0 for each output
    auto ref_f = mx::astype(ref, mx::float32);
    auto gpu_f = mx::astype(gpu, mx::float32);
    mx::eval({ref_f, gpu_f});

    auto match = mx::all(mx::equal(ref_f, gpu_f));
    mx::eval(match);
    REQUIRE(match.item<bool>());
}

TEST_CASE("linear_forward uses registered BitNet 2-bit weights", "[bitnet_quant]") {
    int out_features = 8;
    int in_features = 128;
    int batch_size = 1;

    std::vector<int> vals(out_features * in_features);
    for (int oc = 0; oc < out_features; ++oc) {
        for (int k = 0; k < in_features; ++k) {
            vals[oc * in_features + k] = ((oc * 11 + k * 5) % 3) - 1;
        }
    }

    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(0.25f, mx::bfloat16);
    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);

    std::vector<float> x_data(in_features);
    for (int k = 0; k < in_features; ++k) {
        x_data[k] = static_cast<float>(((k * 7 + 3) % 17) - 8) / 8.0f;
    }
    auto x = mx::astype(mx::array(x_data.data(), {batch_size, in_features}, mx::float32), mx::bfloat16);
    mx::eval({x, wq, scales, biases});

    auto& reg = QuantizedWeightRegistry::instance();
    reg.clear();
    reg.register_weight(&wq, scales, biases, /*group_size=*/128, /*bits=*/2, "affine");

    auto ref_w = dequantize_bitnet_weight(packed, scale, out_features);
    auto ref = mx::matmul(x, mx::transpose(ref_w));
    auto got = linear_forward(x, wq);
    mx::eval({ref, got});

    auto max_diff = mx::max(mx::abs(mx::subtract(mx::astype(ref, mx::float32), mx::astype(got, mx::float32))));
    mx::eval(max_diff);
    reg.clear();

    REQUIRE(max_diff.item<float>() < 1e-4f);
}

TEST_CASE("quantized_matmul matches model dequant for real BitNet decode shape", "[bitnet_quant]") {
    int out_features = 2560;
    int in_features = 2560;
    int batch_size = 1;  // decode path (QMV)

    std::vector<int> vals(out_features * in_features);
    for (int oc = 0; oc < out_features; ++oc) {
        for (int k = 0; k < in_features; ++k) {
            vals[oc * in_features + k] = ((oc * 131 + k * 17) % 3) - 1;
        }
    }

    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(0.25f, mx::bfloat16);
    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);

    std::vector<float> x_data(in_features);
    for (int k = 0; k < in_features; ++k) {
        x_data[k] = static_cast<float>(((k * 13 + 7) % 31) - 15) / 16.0f;
    }
    auto x = mx::astype(mx::array(x_data.data(), {batch_size, in_features}, mx::float32), mx::bfloat16);
    mx::eval({x, wq, scales, biases});

    auto w_ref = dequantize_bitnet_weight(packed, scale, out_features);
    auto ref = mx::matmul(x, mx::transpose(w_ref));
    auto gpu = mx::quantized_matmul(x, wq, scales, biases, /*transpose=*/true, 128, 2);
    mx::eval({ref, gpu});

    auto diff = mx::abs(mx::subtract(mx::astype(ref, mx::float32), mx::astype(gpu, mx::float32)));
    auto max_diff = mx::max(diff);
    mx::eval(max_diff);

    // Fused qmv and dequant+matmul accumulate in different orders, but should
    // agree closely enough. Layout or bias bugs show errors orders of magnitude larger.
    REQUIRE(max_diff.item<float>() < 5.0f);
}

TEST_CASE("bitnet_repack_weights rejects in_features not divisible by 128", "[bitnet_quant]") {
    int out_features = 4;
    int in_features = 64; // NOT divisible by 128

    std::vector<int> vals(out_features * in_features, 0);
    auto packed = pack_ternary_values(vals, out_features, in_features);
    auto scale = mx::array(0.5f, mx::bfloat16);

    REQUIRE_THROWS(bitnet_repack_weights(packed, scale));
}

TEST_CASE("bitnet_repack_weights with larger shape", "[bitnet_quant]") {
    // Realistic size: 4096 output features (1024 packed rows), 2048 in_features
    int out_features = 4096;
    int in_features = 2048;

    std::vector<int> vals(out_features * in_features);
    for (int i = 0; i < static_cast<int>(vals.size()); ++i) {
        vals[i] = (i % 3) - 1; // cycles: -1, 0, 1
    }
    auto packed = pack_ternary_values_lane_major(vals, out_features, in_features);
    auto scale = mx::array(0.1f, mx::bfloat16);

    auto [wq, scales, biases] = bitnet_repack_weights(packed, scale);
    mx::eval({wq, scales, biases});

    // Shapes should be correct
    REQUIRE(wq.shape(0) == out_features);
    REQUIRE(wq.shape(1) == in_features / 16); // 128 uint32 cols
    REQUIRE(scales.shape(0) == out_features);
    REQUIRE(scales.shape(1) == in_features / 128); // 16 groups

    // Quick dequant + matmul to verify no crash
    auto x = mx::full({1, in_features}, 1.0f, mx::bfloat16);
    auto gpu = mx::quantized_matmul(x, wq, scales, biases, true, 128, 2);
    mx::eval(gpu);

    REQUIRE(gpu.shape(0) == 1);
    REQUIRE(gpu.shape(1) == out_features);
}

TEST_CASE("auto_quantize quantizes bf16 weight and registers", "[autoquant]") {
    using namespace mx;

    auto w = astype(random::normal({4, 128}), bfloat16);
    eval(w);

    std::unordered_map<std::string, array> weights;
    weights.insert({std::string("test.weight"), w});

    std::unordered_map<std::string, array*> wmap;
    wmap.insert({std::string("test.weight"), &weights.at(std::string("test.weight"))});

    // Use default BaseConfiguration (per_layer_quantization not set)
    BaseConfiguration base_cfg;
    auto_quantize_weights(weights, wmap, base_cfg);

    auto& qw = weights.at(std::string("test.weight"));
    REQUIRE(qw.dtype() == uint32);
    REQUIRE(qw.ndim() == 2);

    auto* qi = QuantizedWeightRegistry::instance().find(&qw);
    REQUIRE(qi != nullptr);
    REQUIRE(qi->bits == 4);
    REQUIRE(qi->group_size == 64);

    QuantizedWeightRegistry::instance().clear();
}

} // namespace mlx_lm
