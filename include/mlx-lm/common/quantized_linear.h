// Copyright © 2025 — Ported to C++
// QuantizedLinear — quantized weight storage and registry-based dispatch.
//
// Matches Swift's QuantizedLinear: keeps weights packed as uint32 and uses
// mx::quantized_matmul at inference time instead of dequantizing at load time.
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Quantization metadata for a single weight.
struct QuantizationInfo {
    mlx::core::array scales;
    std::optional<mlx::core::array> biases;
    int group_size;
    int bits;
    std::string mode = "affine";
};

// Global registry mapping weight array addresses to quantization metadata.
//
// At load time, quantized weights are NOT dequantized. Instead, the packed
// uint32 weight is stored in the model's member array as-is, and the
// corresponding scales/biases/group_size/bits are registered here.
//
// At inference time, linear_forward() checks this registry: if the weight
// has an entry, it uses mx::quantized_matmul; otherwise, regular mx::matmul.
class QuantizedWeightRegistry {
public:
    static QuantizedWeightRegistry& instance() {
        static QuantizedWeightRegistry reg;
        return reg;
    }

    void register_weight(const mlx::core::array* weight_ptr,
                         mlx::core::array scales,
                         std::optional<mlx::core::array> biases,
                         int group_size, int bits,
                         const std::string& mode = "affine") {
        registry_.insert_or_assign(
            weight_ptr,
            QuantizationInfo{std::move(scales), std::move(biases), group_size, bits, mode});
    }

    const QuantizationInfo* find(const mlx::core::array* weight_ptr) const {
        auto it = registry_.find(weight_ptr);
        return (it != registry_.end()) ? &it->second : nullptr;
    }

    // Drop a weight's quant metadata (frees its scales/biases). Used after
    // fusing two projections into one so the originals can be released.
    void unregister(const mlx::core::array* weight_ptr) {
        registry_.erase(weight_ptr);
    }

    void clear() { registry_.clear(); }
    size_t size() const { return registry_.size(); }

private:
    QuantizedWeightRegistry() = default;
    std::unordered_map<const mlx::core::array*, QuantizationInfo> registry_;
};

// Activation quantization: quantize to N bits symmetrically.
// Matches 1bitLLM's activation_quant(): scale = max_val/max(|x|), round(clip(x*scale))
// Activation quantization matching 1bitLLM's activation_quant:
// Per-token symmetric quantization to N bits.
// Qn = -2^(bits-1), Qp = 2^(bits-1)-1
// scale = Qp / max(|x|) along last dimension (per-token)
// result = round(x * scale).clamp(Qn, Qp) / scale
inline mlx::core::array quantize_activation(
    const mlx::core::array& x,
    int bits = 8)
{
    if (bits >= 16) return x;
    float Qp = static_cast<float>((1 << (bits - 1)) - 1);  // 127 for 8-bit
    float Qn = static_cast<float>(-(1 << (bits - 1)));     // -128 for 8-bit
    int last_dim = x.ndim() - 1;
    auto abs_x = mlx::core::abs(x);
    // Max along last dimension (per-token / per-row)
    std::vector<int> axes = {last_dim};
    bool keepdims = true;
    auto max_abs = mlx::core::max(abs_x, axes, keepdims);
    // Clamp min to avoid division by zero
    max_abs = mlx::core::maximum(max_abs, mlx::core::array(1e-5f));
    auto scale = mlx::core::divide(mlx::core::array(Qp), max_abs);
    auto scaled = mlx::core::multiply(x, scale);
    auto clipped = mlx::core::clip(scaled,
        std::make_optional(mlx::core::array(Qn)),
        std::make_optional(mlx::core::array(Qp)));
    auto q = mlx::core::round(clipped);
    return mlx::core::divide(q, scale);
}

// Quantization-aware linear forward pass.
//
// If the weight is registered as quantized, uses mx::quantized_matmul.
// Otherwise, falls back to regular mx::matmul(x, transpose(w)).
// Matches Swift's QuantizedLinear.callAsFunction / Linear.callAsFunction.
//
// Supports an optional activation_bits parameter for models that need
// activation quantization (1bitLLM BitLinear style).
//
// Each model's static linear_fwd() should delegate to this function.
inline mlx::core::array linear_forward(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array* bias = nullptr,
    int activation_bits = 0)
{
    auto* qi = QuantizedWeightRegistry::instance().find(&w);

    auto input = (activation_bits > 0) ? quantize_activation(x, activation_bits) : x;

    if (qi) {
        auto result = mlx::core::quantized_matmul(
              input, w, qi->scales, qi->biases,
              /*transpose=*/true, qi->group_size, qi->bits,
              /*mode=*/qi->mode);
        if (bias) result = mlx::core::add(result, *bias);
        return result;
    }

    // Non-quantized path: use fused addmm when bias is present.
    if (bias) {
        return mlx::core::addmm(*bias, input, mlx::core::transpose(w));
    }
    return mlx::core::matmul(input, mlx::core::transpose(w));
}

} // namespace mlx_lm
