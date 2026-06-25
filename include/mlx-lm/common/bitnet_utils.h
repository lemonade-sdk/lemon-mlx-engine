// BitNet ternary quantization utilities.
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// BitNet b1.58 packs ternary values {-1, 0, +1} as 2-bit codes {0, 1, 2}
// four-per-byte in uint8 arrays. The packed shape is [out_features/4, in_features].
// After unpacking, the result is [out_features, in_features], scaled by weight_scale.
inline mlx::core::array dequantize_bitnet_weight(
    const mlx::core::array& packed_weight,
    const mlx::core::array& weight_scale,
    int /*out_features*/)
{
    namespace mx = mlx::core;

    // Cast to int32 for bitwise operations.
    auto packed = mx::astype(packed_weight, mx::int32);

    // Extract 4 ternary values from each byte: bits [1:0], [3:2], [5:4], [7:6].
    // Concatenate along axis 0 (not stack+reshape) to match the reference
    // unpacking: out[0:R]=lane0, out[R:2R]=lane1, out[2R:3R]=lane2, out[3R:4R]=lane3.
    auto v0 = mx::bitwise_and(packed, mx::array(0x03));
    auto v1 = mx::bitwise_and(mx::right_shift(packed, mx::array(2)), mx::array(0x03));
    auto v2 = mx::bitwise_and(mx::right_shift(packed, mx::array(4)), mx::array(0x03));
    auto v3 = mx::bitwise_and(mx::right_shift(packed, mx::array(6)), mx::array(0x03));

    // [packed_rows, in] × 4 → concatenate to [out_features, in].
    auto flat = mx::concatenate({v0, v1, v2, v3}, 0);

    // Map 2-bit codes: 0→-1, 1→0, 2→+1, then scale.
    auto ternary = mx::astype(mx::subtract(flat, mx::array(1)), mx::float16);
    auto scale = mx::astype(weight_scale, mx::float16);
    return mx::multiply(ternary, scale);
}

// Repack BitNet uint8 packed ternary weights into standard MLX uint32 2-bit
// quantized format. Returns {wq_uint32, scales_fp16, biases_fp16}.
//
// BitNet packs 4 ternary codes {0→-1, 1→0, 2→+1} per byte across output lanes:
//   uint8[row, c] = lane0[1:0] | lane1[3:2] | lane2[5:4] | lane3[7:6]
// The dequantized output order is lane-major:
//   out[0:R]=lane0, out[R:2R]=lane1, out[2R:3R]=lane2, out[3R:4R]=lane3,
// where R=packed_rows, so row = oc % R and lane = oc / R.
//
// MLX 2-bit format: uint32[out, ceil(in/16)], each uint32 = 16 codes at 2 bits
// each, least-significant code first, padding with 0.
//
// MLX uses per-group quantization: scales/biases have shape [out_features, num_groups]
// where num_groups = in_features / group_size. For group_size = 128.
inline std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
bitnet_repack_weights(
    const mlx::core::array& packed_weight,  // uint8 [out/4, in]
    const mlx::core::array& weight_scale)  // scalar (bf16 or fp16)
{
    namespace mx = mlx::core;
    constexpr int kBitnetGroupSize = 128;

    auto shape = packed_weight.shape();
    int packed_rows = shape[0];
    int in_features = shape[1];
    int out_features = packed_rows * 4;

    if (in_features % kBitnetGroupSize != 0) {
        throw std::runtime_error(
            "BitNet: in_features " + std::to_string(in_features) +
            " must be divisible by group_size " +
            std::to_string(kBitnetGroupSize));
    }
    int num_groups = in_features / kBitnetGroupSize;

    int in_rounded = ((in_features + 15) / 16) * 16;
    int cols_uint32 = in_rounded / 16;

    // Convert scale to fp16 and materialize
    mx::array ws_fp16 = mx::astype(weight_scale, mx::float16);
    mx::eval(ws_fp16);
    auto ws = static_cast<float>(ws_fp16.data<mx::float16_t>()[0]);

    // Materialize packed weight and read uint8 data
    mx::eval(packed_weight);
    auto w_data = packed_weight.data<uint8_t>();

    // Allocate outputs:
    // wq: [out_features, cols_uint32]
    // scales: [out_features, num_groups] - per-group quantization
    // biases: [out_features, num_groups]
    std::vector<uint32_t> wq(out_features * cols_uint32, 0);
    std::vector<mx::float16_t> scales(out_features * num_groups);
    std::vector<mx::float16_t> biases(out_features * num_groups);

    auto ws_h = static_cast<mx::float16_t>(ws);
    auto neg_ws_h = static_cast<mx::float16_t>(-ws);

    for (int oc = 0; oc < out_features; ++oc) {
        int row = oc % packed_rows;
        int lane = oc / packed_rows;
        int bit_shift = lane * 2;

        // Replicate the single BitNet scale across all groups for this output row
        for (int g = 0; g < num_groups; ++g) {
            scales[oc * num_groups + g] = ws_h;
            biases[oc * num_groups + g] = neg_ws_h;
        }

        // Pack 16 input values per uint32
        for (int g = 0; g < cols_uint32; ++g) {
            uint32_t packed = 0;
            for (int i = 0; i < 16; ++i) {
                int c = g * 16 + i;
                uint32_t val = 0;
                if (c < in_features) {
                    val = (w_data[row * in_features + c] >> bit_shift) & 0x03;
                }
                packed |= (val << (i * 2));
            }
            wq[oc * cols_uint32 + g] = packed;
        }
    }

    auto wq_arr = mx::array(wq.data(), {out_features, cols_uint32}, mx::uint32);
    // Scales and biases: [out_features, num_groups] for per-group quantization
    auto scales_arr = mx::array(scales.data(), {out_features, num_groups}, mx::float16);
    auto biases_arr = mx::array(biases.data(), {out_features, num_groups}, mx::float16);

    return {std::move(wq_arr), std::move(scales_arr), std::move(biases_arr)};
}

} // namespace mlx_lm
