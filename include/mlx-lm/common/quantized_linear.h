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
                         int group_size, int bits) {
        registry_.insert_or_assign(
            weight_ptr,
            QuantizationInfo{std::move(scales), std::move(biases), group_size, bits});
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

// Quantization-aware linear forward pass.
//
// If the weight is registered as quantized, uses mx::quantized_matmul.
// Otherwise, falls back to regular mx::matmul(x, transpose(w)).
// Matches Swift's QuantizedLinear.callAsFunction / Linear.callAsFunction.
//
// Each model's static linear_fwd() should delegate to this function.
inline mlx::core::array linear_forward(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array* bias = nullptr)
{
    namespace mx = mlx::core;

    auto* qi = QuantizedWeightRegistry::instance().find(&w);

    if (qi) {
        auto result = mx::quantized_matmul(
              x, w, qi->scales, qi->biases,
              /*transpose=*/true, qi->group_size, qi->bits);
        if (bias) result = mx::add(result, *bias);
        return result;
    }

    // Non-quantized path: use fused addmm when bias is present.
    // addmm computes D = beta*C + alpha*(A @ B) in a single kernel.
    if (bias) {
        return mx::addmm(*bias, x, mx::transpose(w));
    }
    return mx::matmul(x, mx::transpose(w));
}

} // namespace mlx_lm
