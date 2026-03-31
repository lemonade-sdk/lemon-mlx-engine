// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <vector>

namespace mlx_lm {

// Forward declaration — defined in attention_utils.h.
class AttentionMask;

// Create a causal attention mask.
mlx::core::array create_causal_mask(
    int n,
    int offset,
    std::optional<int> window_size = std::nullopt);

// KV cache interface — no virtual functions.
// Uses CRTP so each implementation gets compile-time dispatch.
template <typename Derived>
class KVCacheBase {
public:
    int offset() const { return static_cast<const Derived*>(this)->offset_impl(); }

    std::optional<int> max_size() const { return static_cast<const Derived*>(this)->max_size_impl(); }

    std::pair<mlx::core::array, mlx::core::array>
    update(const mlx::core::array& keys, const mlx::core::array& values) {
        return static_cast<Derived*>(this)->update_impl(keys, values);
    }

    bool is_trimmable() const { return static_cast<const Derived*>(this)->is_trimmable_impl(); }

    int trim(int n) { return static_cast<Derived*>(this)->trim_impl(n); }
};

// Simple (growing) KV cache.
class KVCacheSimple : public KVCacheBase<KVCacheSimple> {
    friend class KVCacheBase<KVCacheSimple>;

    int offset_ = 0;
    int initial_capacity_ = 256;
    std::optional<mlx::core::array> keys_;
    std::optional<mlx::core::array> values_;

    int offset_impl() const { return offset_; }
    std::optional<int> max_size_impl() const { return std::nullopt; }
    bool is_trimmable_impl() const { return true; }

    std::pair<mlx::core::array, mlx::core::array>
    update_impl(const mlx::core::array& new_keys, const mlx::core::array& new_values);

    int trim_impl(int n);

public:
    KVCacheSimple() = default;
    explicit KVCacheSimple(int initial_capacity) : initial_capacity_(initial_capacity) {}

    // Access stored state for KV sharing. Returns {keys, values} if populated.
    std::vector<mlx::core::array> state() const {
        std::vector<mlx::core::array> result;
        if (keys_.has_value()) result.push_back(keys_.value());
        if (values_.has_value()) result.push_back(values_.value());
        return result;
    }

    // Access raw keys/values for conversion to QuantizedKVCache.
    const std::optional<mlx::core::array>& raw_keys() const { return keys_; }
    const std::optional<mlx::core::array>& raw_values() const { return values_; }
};

// Rotating (fixed-size) KV cache with overwriting.
class RotatingKVCache : public KVCacheBase<RotatingKVCache> {
    friend class KVCacheBase<RotatingKVCache>;

    int max_size_;
    int keep_;
    int offset_ = 0;
    std::optional<mlx::core::array> keys_;
    std::optional<mlx::core::array> values_;
    int idx_ = 0;

    int offset_impl() const { return offset_; }
    std::optional<int> max_size_impl() const { return max_size_; }
    bool is_trimmable_impl() const { return false; }

    std::pair<mlx::core::array, mlx::core::array>
    update_impl(const mlx::core::array& new_keys, const mlx::core::array& new_values);

    int trim_impl(int /*n*/) { return 0; }

public:
    RotatingKVCache(int max_size, int keep = 4)
        : max_size_(max_size), keep_(keep) {}

    std::vector<mlx::core::array> state() const {
        std::vector<mlx::core::array> result;
        if (keys_.has_value()) result.push_back(keys_.value());
        if (values_.has_value()) result.push_back(values_.value());
        return result;
    }
};

// Quantized KV cache — stores keys/values in quantized form to reduce memory.
// Matches Swift's QuantizedKVCache. On update, quantizes incoming KV and
// returns dequantized data so it's transparent to models.
class QuantizedKVCache : public KVCacheBase<QuantizedKVCache> {
    friend class KVCacheBase<QuantizedKVCache>;

public:
    // Quantized storage: (packed_weight, scales, biases)
    struct QTuple {
        mlx::core::array weight;
        mlx::core::array scales;
        mlx::core::array biases;
    };

    QuantizedKVCache(int group_size, int bits)
        : group_size_(group_size), bits_(bits) {}

    // Construct from an existing KVCacheSimple (converts its data to quantized).
    static QuantizedKVCache from_simple(const KVCacheSimple& simple, int group_size, int bits);

    std::vector<mlx::core::array> state() const { return {}; }

private:
    std::optional<QTuple> keys_;
    std::optional<QTuple> values_;
    int offset_ = 0;
    int group_size_;
    int bits_;

    int offset_impl() const { return offset_; }
    std::optional<int> max_size_impl() const { return std::nullopt; }
    bool is_trimmable_impl() const { return false; }
    int trim_impl(int /*n*/) { return 0; }

    std::pair<mlx::core::array, mlx::core::array>
    update_impl(const mlx::core::array& new_keys, const mlx::core::array& new_values);
};

// Mamba-style state space model cache.
// Stores conv_state (index 0) and ssm_state (index 1).
class MambaCache {
    std::optional<mlx::core::array> states_[2];
    int offset_ = 0;

public:
    MambaCache() = default;

    int offset() const { return offset_; }
    std::optional<int> max_size() const { return std::nullopt; }
    bool is_trimmable() const { return false; }
    int trim(int) { return 0; }

    // Dummy update to satisfy KVCache variant interface
    std::pair<mlx::core::array, mlx::core::array>
    update(const mlx::core::array& keys, const mlx::core::array& values) {
        return {keys, values};
    }

    // Index-based access (0 = conv_state, 1 = ssm_state)
    std::optional<mlx::core::array>& operator[](int idx) { return states_[idx]; }
    const std::optional<mlx::core::array>& operator[](int idx) const { return states_[idx]; }

    void set_offset(int o) { offset_ = o; }

    std::vector<mlx::core::array> state() const { return {}; }
};

// Compound cache for hybrid models (e.g., FalconH1, BaichuanM1).
// Stores a MambaCache (for SSM/conv state) alongside a KV cache (for attention).
// Standard operations (offset, update, trim) delegate to the KV sub-cache.
class CompoundCache {
    MambaCache mamba_;
    std::variant<KVCacheSimple, RotatingKVCache> kv_;

public:
    CompoundCache() : kv_(KVCacheSimple{}) {}
    CompoundCache(MambaCache m, KVCacheSimple k) : mamba_(std::move(m)), kv_(std::move(k)) {}
    CompoundCache(MambaCache m, RotatingKVCache k) : mamba_(std::move(m)), kv_(std::move(k)) {}

    // Delegate to KV sub-cache
    int offset() const {
        return std::visit([](const auto& c) { return c.offset(); }, kv_);
    }
    std::optional<int> max_size() const {
        return std::visit([](const auto& c) { return c.max_size(); }, kv_);
    }
    std::pair<mlx::core::array, mlx::core::array>
    update(const mlx::core::array& keys, const mlx::core::array& values) {
        return std::visit([&](auto& c) { return c.update(keys, values); }, kv_);
    }
    bool is_trimmable() const {
        return std::visit([](const auto& c) { return c.is_trimmable(); }, kv_);
    }
    int trim(int n) {
        return std::visit([n](auto& c) { return c.trim(n); }, kv_);
    }
    std::vector<mlx::core::array> state() const {
        return std::visit([](const auto& c) { return c.state(); }, kv_);
    }

    // Access the MambaCache sub-part (for SSM/conv state)
    MambaCache* as_mamba() { return &mamba_; }
    const MambaCache* as_mamba() const { return &mamba_; }
};

// Type-erased KV cache so we can store heterogeneous caches in a vector.
// Wraps any CRTP cache type with a small set of operations, using
// std::variant internally instead of virtual dispatch.
class KVCache {
public:
    using CacheVariant = std::variant<KVCacheSimple, RotatingKVCache, MambaCache, CompoundCache, QuantizedKVCache>;

    KVCache() : impl_(KVCacheSimple{}) {}

    explicit KVCache(KVCacheSimple c)    : impl_(std::move(c)) {}
    explicit KVCache(RotatingKVCache c)  : impl_(std::move(c)) {}
    explicit KVCache(MambaCache c)       : impl_(std::move(c)) {}
    explicit KVCache(CompoundCache c)    : impl_(std::move(c)) {}
    explicit KVCache(QuantizedKVCache c) : impl_(std::move(c)) {}

    // Access the underlying MambaCache (returns nullptr if not a MambaCache).
    // For CompoundCache, returns the MambaCache sub-part.
    MambaCache* as_mamba() {
        if (auto* cc = std::get_if<CompoundCache>(&impl_)) return cc->as_mamba();
        return std::get_if<MambaCache>(&impl_);
    }
    const MambaCache* as_mamba() const {
        if (auto* cc = std::get_if<CompoundCache>(&impl_)) return cc->as_mamba();
        return std::get_if<MambaCache>(&impl_);
    }

    int offset() const {
        return std::visit([](const auto& c) { return c.offset(); }, impl_);
    }

    std::optional<int> max_size() const {
        return std::visit([](const auto& c) { return c.max_size(); }, impl_);
    }

    std::pair<mlx::core::array, mlx::core::array>
    update(const mlx::core::array& keys, const mlx::core::array& values) {
        return std::visit([&](auto& c) { return c.update(keys, values); }, impl_);
    }

    bool is_trimmable() const {
        return std::visit([](const auto& c) { return c.is_trimmable(); }, impl_);
    }

    int trim(int n) {
        return std::visit([n](auto& c) { return c.trim(n); }, impl_);
    }

    // Access stored KV state for sharing between layers.
    std::vector<mlx::core::array> state() const {
        return std::visit([](const auto& c) { return c.state(); }, impl_);
    }

    // Check if the underlying cache is a QuantizedKVCache.
    bool is_quantized() const { return std::get_if<QuantizedKVCache>(&impl_) != nullptr; }

private:
    CacheVariant impl_;
};

// Dynamically convert KVCacheSimple entries to QuantizedKVCache.
// Called per-token after `quantized_kv_start` tokens have been generated.
// Matches Swift's maybeQuantizeKVCache().
void maybe_quantize_kv_cache(
    std::vector<KVCache>& cache,
    std::optional<int> kv_bits,
    int kv_group_size = 64,
    int quantized_kv_start = 0);

} // namespace mlx_lm
