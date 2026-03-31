// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/types.h>
#include <mlx/mlx.h>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// CRTP base for all language models. No virtual functions.
//
// Derived classes must implement:
//   PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size)
//   LMOutput      call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state)
//   std::vector<KVCache> new_cache_impl(const GenerateParameters& params)
//   std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights)
template <typename Derived>
class LanguageModel {
public:
    // Prepare the cache state and consume the LMInput.
    PrepareResult prepare(const LMInput& input, std::vector<KVCache>& cache, int window_size = -1) {
        return derived().prepare_impl(input, cache, window_size);
    }

    // Produce a single step (token) from the model.
    LMOutput operator()(const LMInput::Text& input,
                        std::vector<KVCache>* cache = nullptr,
                        const LMOutput::State* state = nullptr) {
        return derived().call_impl(input, cache, state);
    }

    // Simplified interface (just tokens + cache).
    mlx::core::array forward(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr) {
        return derived().forward_impl(inputs, cache);
    }

    // Create a new array of KVCache.
    std::vector<KVCache> new_cache(const GenerateParameters& params = {}) {
        return derived().new_cache_impl(params);
    }

    // Optionally preprocess weights.
    std::unordered_map<std::string, mlx::core::array>
    sanitize(std::unordered_map<std::string, mlx::core::array> weights) {
        return derived().sanitize_impl(std::move(weights));
    }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }
};

// Mixin: provides automatic new_cache_impl for models that expose kv_heads().
// Derived must have: const std::vector<int>& kv_heads() const;
template <typename Derived>
class KVCacheDimensionProvider {
public:
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params) const {
        const auto& heads = static_cast<const Derived*>(this)->kv_heads();
        int num_layers = static_cast<int>(heads.size());

        std::vector<KVCache> caches;
        caches.reserve(num_layers);

        if (params.max_kv_size.has_value()) {
            for (int i = 0; i < num_layers; ++i) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            }
        } else if (params.ctx_size > 0) {
            // Pre-allocate KV cache to ctx_size to avoid grow-and-copy.
            for (int i = 0; i < num_layers; ++i) {
                caches.emplace_back(KVCacheSimple(params.ctx_size));
            }
        } else {
            for (int i = 0; i < num_layers; ++i) {
                caches.emplace_back(KVCacheSimple{});
            }
        }
        return caches;
    }
};

} // namespace mlx_lm
