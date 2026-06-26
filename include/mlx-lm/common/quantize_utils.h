// Copyright © 2025 — Ported to C++
#pragma once

#include <mlx-lm/common/base_config.h>
#include <mlx/mlx.h>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Register quantized weights in the QuantizedWeightRegistry.
//
// For each key ending in ".scales" with a matching ".weight", registers
// the quantization metadata (scales, biases, group_size, bits) so that
// linear_fwd() from quantized_linear.h will use mx::quantized_matmul.
//
// Requires the model's weight_map so we can map weight names to member
// array addresses (which linear_fwd uses for registry lookups).
//
// Removes .scales and .biases entries from the weights map after
// registration (the packed uint32 .weight entries stay).
void register_quantized_weights(
    std::unordered_map<std::string, mlx::core::array>& weights,
    const BaseConfiguration& config,
    const std::unordered_map<std::string, mlx::core::array*>& weight_map);

// Auto-quantize unquantized bf16/fp16 weights to 4-bit on-the-fly at load time.
//
// Iterates weights in weight_map whose keys end in ".weight", have ndim==2,
// and are float16/bfloat16. For each such weight, calls mx::quantize() to
// produce {packed_uint32, scales, biases}, replaces the weight with the packed
// uint32 version, and registers scales/biases in QuantizedWeightRegistry.
//
// Skips if base_config.per_layer_quantization already exists (model is
// already quantized). This allows loading bf16/fp16 HF checkpoints directly
// with --auto-quantize and having them quantized to 4-bit in-place.
void auto_quantize_weights(
    std::unordered_map<std::string, mlx::core::array>& weights,
    const std::unordered_map<std::string, mlx::core::array*>& weight_map,
    const BaseConfiguration& base_config);

// Pre-quantize 2D F32 weights to 1-bit ternary {-1,0,+1} * scale.
// Matches 1bitLLM weight_quant() for runtime quantization.
void quantize_weights_to_ternary(
    std::unordered_map<std::string, mlx::core::array>& weights);

// Legacy: dequantize weights at load time (uses more memory).
// Kept for models that haven't been updated to use quantized_linear.h yet.
std::unordered_map<std::string, mlx::core::array> dequantize_weights(
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& config);

} // namespace mlx_lm
