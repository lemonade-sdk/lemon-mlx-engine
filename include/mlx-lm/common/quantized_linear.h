// Copyright © 2025 — Ported to C++
// QuantizedLinear — quantized weight storage and registry-based dispatch.
//
// Matches Swift's QuantizedLinear: keeps weights packed as uint32 and uses
// mx::quantized_matmul at inference time instead of dequantizing at load time.
#pragma once

#include <mlx/mlx.h>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry_.find(weight_ptr);
        if (it != registry_.end()) {
            // Shared array (e.g. delta-merge): bump refcount, keep first meta.
            it->second.refcount++;
        } else {
            registry_.emplace(
                weight_ptr,
                Entry{QuantizationInfo{std::move(scales), std::move(biases),
                                       group_size, bits},
                      /*refcount=*/1});
        }
        // Capture into active load scope (if any) so unload can unregister.
        if (load_scope_ptrs_ != nullptr) {
            load_scope_ptrs_->push_back(weight_ptr);
        }
    }

    const QuantizationInfo* find(const mlx::core::array* weight_ptr) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry_.find(weight_ptr);
        return (it != registry_.end()) ? &it->second.info : nullptr;
    }

    // Drop one ownership claim. Erases metadata only when refcount hits 0
    // (safe if two ModelContainers share the same packed weight pointer).
    void unregister(const mlx::core::array* weight_ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = registry_.find(weight_ptr);
        if (it == registry_.end()) return;
        if (--it->second.refcount <= 0) {
            registry_.erase(it);
        }
    }

    void unregister_many(const std::vector<const mlx::core::array*>& ptrs) {
        for (auto* p : ptrs) unregister(p);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        registry_.clear();
    }
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return registry_.size();
    }

    // RAII: all register_weight calls while alive are recorded into `out`.
    // ModelContainer uses this so destructor can unregister on unload.
    struct LoadScope {
        explicit LoadScope(std::vector<const mlx::core::array*>& out)
            : prev_(QuantizedWeightRegistry::instance().load_scope_ptrs_) {
            QuantizedWeightRegistry::instance().load_scope_ptrs_ = &out;
        }
        ~LoadScope() {
            QuantizedWeightRegistry::instance().load_scope_ptrs_ = prev_;
        }
        LoadScope(const LoadScope&) = delete;
        LoadScope& operator=(const LoadScope&) = delete;
    private:
        std::vector<const mlx::core::array*>* prev_;
    };

private:
    struct Entry {
        QuantizationInfo info;
        int refcount = 1;
    };
    QuantizedWeightRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<const mlx::core::array*, Entry> registry_;
    // Non-owning: points at the active LoadScope's vector (or null).
    std::vector<const mlx::core::array*>* load_scope_ptrs_ = nullptr;
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
