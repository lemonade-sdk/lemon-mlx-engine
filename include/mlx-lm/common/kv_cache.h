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
    // Extra headroom reserved on the FIRST allocation, on top of the initial
    // write (the prompt). Sizing the buffer once to prompt+reserve makes decode
    // write in place at offset_ with no grow-and-copy and a stable address
    // (also the prerequisite for HIP graph capture). 0 = legacy grow-by-doubling.
    int reserve_ = 0;
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
    KVCacheSimple(int initial_capacity, int reserve)
        : initial_capacity_(initial_capacity), reserve_(reserve) {}
    // Reserve generation headroom so the one-shot first allocation covers the
    // whole run (prompt + reserve), avoiding any later grow-and-copy.
    void set_reserve(int reserve) { reserve_ = reserve; }

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

    // Partial-rollback API for MTP speculative decoding.
    size_t get_position() const { return static_cast<size_t>(offset_); }
    void set_position(size_t pos);
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

    // Partial-rollback API: get_position returns current offset.
    // set_position rolls back the logical offset only (physical ring unchanged).
    // This is safe for speculative decoding since the ring buffer retains
    // all data; only the write position changes.
    size_t get_position() const { return static_cast<size_t>(offset_); }
    void set_position(size_t pos);
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

public:
    // Partial-rollback API for MTP speculative decoding.
    size_t get_position() const { return static_cast<size_t>(offset_); }
    void set_position(size_t pos);
};

// Mamba-style state space model cache.
// Stores conv_state (index 0) and ssm_state (index 1).
class MambaCache {
    std::optional<mlx::core::array> states_[2];
    int offset_ = 0;

public:
    // Snapshot of MambaCache state for speculative decoding rollback.
    struct Snapshot {
        std::optional<mlx::core::array> states[2];
        int offset = 0;
    };

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

    // Snapshot/restore for MTP speculative decoding.
    // Save current state before trunk verification. If drafts are rejected,
    // restore and re-run on accepted tokens to keep Mamba and KV caches in sync.
    Snapshot snapshot() const {
        Snapshot s;
        s.states[0] = states_[0];
        s.states[1] = states_[1];
        s.offset = offset_;
        return s;
    }

    void restore(const Snapshot& s) {
        states_[0] = s.states[0];
        states_[1] = s.states[1];
        offset_ = s.offset;
    }

    // Partial-rollback API: MambaCache does not support token-level rollback
    // via set_position (recurrent state cannot be trimmed). Use snapshot/restore
    // or the spec-capture API below for speculative decoding.
    size_t get_position() const { return static_cast<size_t>(offset_); }
    void set_position(size_t /*pos*/) { /* no-op: recurrent state cannot be rolled back */ }

    // --- Speculative-decoding intermediates ("gated-delta intermediates") ---
    //
    // To roll the recurrent state back to an accepted prefix WITHOUT re-running
    // the trunk, the gated-delta layer records, during a multi-token verify
    // forward, the SSM state after EACH token plus the full conv input. We then
    // pick the state as-of the accepted prefix. Capture is opt-in (set per step
    // by the speculative loop) so the large prompt prefill never pays for it.
    void set_capture_spec(bool v) { capture_spec_ = v; if (!v) clear_spec(); }
    bool capture_spec() const { return capture_spec_; }
    bool has_spec() const { return spec_ssm_states_.has_value(); }

    // Called by the gated-delta layer when capture_spec() is true.
    //   ssm_states_seq: [B, T, Hv, Dv, Dk] — recurrent state AFTER each token.
    //   conv_input:     [B, (kernel-1)+T, conv_dim] — full conv input this step.
    //   base_offset:    cache offset before this verify forward.
    void store_spec(const mlx::core::array& ssm_states_seq,
                    const mlx::core::array& conv_input,
                    int base_offset) {
        spec_ssm_states_ = ssm_states_seq;
        spec_conv_input_ = conv_input;
        spec_base_offset_ = base_offset;
    }

    void clear_spec() {
        spec_ssm_states_ = std::nullopt;
        spec_conv_input_ = std::nullopt;
    }

    // Roll the recurrent + conv state back to the first `keep` tokens of the
    // captured verify chunk (keep >= 1). Sets ssm_state to the state after token
    // keep-1, conv_state to the (kernel-1)-token window ending at token keep-1,
    // and offset to base_offset + keep. No-op if no intermediates were captured.
    // The conv window size (kernel-1) is derived from the captured shapes.
    void rollback_spec(int keep) {
        if (!spec_ssm_states_.has_value() || !spec_conv_input_.has_value()) return;
        namespace mx = mlx::core;
        const auto& seq = spec_ssm_states_.value();   // [B, T, Hv, Dv, Dk]
        const auto& ci = spec_conv_input_.value();    // [B, (k-1)+T, conv_dim]
        int T = seq.shape(1);
        int win = ci.shape(1) - T;                    // (kernel-1) prev-state rows
        if (keep < 1) keep = 1;
        if (keep > T) keep = T;

        // ssm_state after `keep` tokens == seq[:, keep-1].
        states_[1] = mx::squeeze(
            mx::slice(seq, {0, keep - 1, 0, 0, 0},
                      {seq.shape(0), keep, seq.shape(2), seq.shape(3), seq.shape(4)}),
            1);

        // conv_state == window [keep, keep + (kernel-1)) of the full conv input.
        states_[0] = mx::slice(ci, {0, keep, 0},
                               {ci.shape(0), keep + win, ci.shape(2)});

        offset_ = spec_base_offset_ + keep;
        clear_spec();
    }

private:
    bool capture_spec_ = false;
    std::optional<mlx::core::array> spec_ssm_states_;  // [B, T, Hv, Dv, Dk]
    std::optional<mlx::core::array> spec_conv_input_;  // [B, (k-1)+T, conv_dim]
    int spec_base_offset_ = 0;
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

    // Partial-rollback dispatch for MTP speculative decoding.
    // - KVCacheSimple / QuantizedKVCache / RotatingKVCache: return / apply the offset.
    // - MambaCache: out of scope for this scaffolding cut --
    //   recurrent rollback requires the gated-delta intermediates kernel.
    //   get_position returns current offset, set_position is a no-op.
    // - CompoundCache: delegates to sub-caches.
    size_t get_position() const {
        return std::visit([](const auto& c) -> size_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, KVCacheSimple> ||
                          std::is_same_v<T, QuantizedKVCache> ||
                          std::is_same_v<T, RotatingKVCache> ||
                          std::is_same_v<T, MambaCache>) {
                return c.get_position();
            } else {
                return 0;  // CompoundCache: unsupported
            }
        }, impl_);
    }
    void set_position(size_t pos) {
        std::visit([pos](auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, KVCacheSimple> ||
                          std::is_same_v<T, QuantizedKVCache> ||
                          std::is_same_v<T, RotatingKVCache> ||
                          std::is_same_v<T, MambaCache>) {
                c.set_position(pos);
            }
        }, impl_);
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
