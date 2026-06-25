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

} // namespace mlx_lm
