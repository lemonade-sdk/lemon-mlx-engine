// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/kv_cache.h>
#include <mlx/mlx.h>

// Direct GPU KV cache write — bypasses MLX's functional array model.
// Implemented in ROCm eval.cpp; only available on GPU backend.
extern "C" void mlx_gpu_memcpy_async(void* dst, const void* src, size_t bytes);

// Weak symbol so non-ROCm builds link without the implementation.
__attribute__((weak)) void mlx_gpu_memcpy_async(void*, const void*, size_t) {}

namespace {

bool is_gpu_backend() {
    return mlx::core::default_device() == mlx::core::Device::gpu;
}

// Write new_data into buffer at [offset] along axis 2, in-place via GPU memcpy.
// buffer: [B, H, capacity, D], new_data: [B, H, n, D] — must be contiguous.
// Only copies n*D elements per (B,H) slice — O(n) not O(capacity).
void kv_write_inplace(
    mlx::core::array& buffer,
    const mlx::core::array& new_data,
    int offset) {
  namespace mx = mlx::core;

  mx::eval(new_data);

  int B = buffer.shape(0);
  int H = buffer.shape(1);
  int D = buffer.shape(3);
  int capacity = buffer.shape(2);
  int n = new_data.shape(2);
  size_t elem_size = mx::size_of(buffer.dtype());
  size_t slice_bytes = static_cast<size_t>(n) * D * elem_size;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      size_t dst_off = ((static_cast<size_t>(b) * H + h) * capacity + offset) * D * elem_size;
      size_t src_off = (static_cast<size_t>(b) * H + h) * n * D * elem_size;

      auto* dst = static_cast<char*>(buffer.data<void>()) + dst_off;
      auto* src = static_cast<const char*>(new_data.data<void>()) + src_off;

      mlx_gpu_memcpy_async(dst, src, slice_bytes);
    }
  }
}
} // anonymous namespace

namespace mlx_lm {

namespace mx = mlx::core;

// --- create_causal_mask ---

mlx::core::array create_causal_mask(
    int n, int offset, std::optional<int> window_size)
{
    auto rinds = mx::arange(0, offset + n, mx::int32);
    auto linds = (offset != 0)
        ? mx::arange(offset, offset + n, mx::int32)
        : rinds;
    linds = mx::expand_dims(linds, 1);
    rinds = mx::expand_dims(rinds, 0);
    auto mask = mx::greater_equal(linds, rinds);
    if (window_size.has_value()) {
        auto ws = mx::array(window_size.value(), mx::int32);
        mask = mx::logical_and(mask, mx::less(linds, mx::add(rinds, ws)));
    }
    return mask;
}

// --- KVCacheSimple ---

std::pair<mlx::core::array, mlx::core::array>
KVCacheSimple::update_impl(
    const mlx::core::array& new_keys,
    const mlx::core::array& new_values)
{
    int n_new = new_keys.shape(2);

    if (!keys_.has_value()) {
        int B = new_keys.shape(0);
        int H = new_keys.shape(1);
        int D = new_keys.shape(3);
        int alloc_len = std::max(n_new, initial_capacity_);

        keys_ = mx::zeros({B, H, alloc_len, D}, new_keys.dtype());
        values_ = mx::zeros({B, H, alloc_len, D}, new_values.dtype());

        // Write initial K/V into position 0
        keys_ = mx::slice_update(keys_.value(), new_keys,
            mx::Shape{0, 0, 0, 0}, mx::Shape{B, H, n_new, D});
        values_ = mx::slice_update(values_.value(), new_values,
            mx::Shape{0, 0, 0, 0}, mx::Shape{B, H, n_new, D});

        offset_ = n_new;
        return {mx::slice(keys_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,n_new,D}),
                mx::slice(values_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,n_new,D})};
    }

    int B = keys_.value().shape(0);
    int H = keys_.value().shape(1);
    int D = keys_.value().shape(3);
    int current_alloc = keys_.value().shape(2);

    if (offset_ + n_new <= current_alloc) {
        keys_ = mx::slice_update(keys_.value(), new_keys,
            mx::Shape{0, 0, offset_, 0}, mx::Shape{B, H, offset_ + n_new, D});
        values_ = mx::slice_update(values_.value(), new_values,
            mx::Shape{0, 0, offset_, 0}, mx::Shape{B, H, offset_ + n_new, D});
        offset_ += n_new;
        return {mx::slice(keys_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D}),
                mx::slice(values_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D})};
    }

    int new_alloc = std::max(current_alloc * 2, offset_ + n_new);
    auto new_k = mx::zeros({B, H, new_alloc, D}, keys_.value().dtype());
    auto new_v = mx::zeros({B, H, new_alloc, D}, values_.value().dtype());

    // Copy existing data + append new
    new_k = mx::slice_update(new_k,
        mx::slice(keys_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D}),
        mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D});
    new_v = mx::slice_update(new_v,
        mx::slice(values_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D}),
        mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D});

    new_k = mx::slice_update(new_k, new_keys,
        mx::Shape{0,0,offset_,0}, mx::Shape{B,H,offset_+n_new,D});
    new_v = mx::slice_update(new_v, new_values,
        mx::Shape{0,0,offset_,0}, mx::Shape{B,H,offset_+n_new,D});

    keys_ = new_k;
    values_ = new_v;
    offset_ += n_new;
    return {mx::slice(keys_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D}),
            mx::slice(values_.value(), mx::Shape{0,0,0,0}, mx::Shape{B,H,offset_,D})};
}

int KVCacheSimple::trim_impl(int n) {
    if (!keys_.has_value() || n <= 0) return 0;

    int seq_len = keys_.value().shape(2);
    int to_trim = std::min(n, seq_len);

    if (to_trim == seq_len) {
        keys_ = std::nullopt;
        values_ = std::nullopt;
    } else {
        keys_ = mx::slice(keys_.value(), mx::Shape{0, 0, 0, 0},
                           {keys_.value().shape(0), keys_.value().shape(1),
                            seq_len - to_trim, keys_.value().shape(3)});
        values_ = mx::slice(values_.value(), mx::Shape{0, 0, 0, 0},
                             {values_.value().shape(0), values_.value().shape(1),
                              seq_len - to_trim, values_.value().shape(3)});
    }

    offset_ -= to_trim;
    return to_trim;
}

// --- RotatingKVCache ---

std::pair<mlx::core::array, mlx::core::array>
RotatingKVCache::update_impl(
    const mlx::core::array& new_keys,
    const mlx::core::array& new_values)
{
    int n_new = new_keys.shape(2);

    if (!keys_.has_value()) {
        keys_ = new_keys;
        values_ = new_values;
        idx_ = n_new;
        offset_ = n_new;
        return {keys_.value(), values_.value()};
    }

    int current_len = keys_.value().shape(2);

    if (current_len + n_new <= max_size_) {
        keys_ = mx::concatenate({keys_.value(), new_keys}, 2);
        values_ = mx::concatenate({values_.value(), new_values}, 2);
        idx_ = current_len + n_new;
    } else {
        keys_ = mx::concatenate({keys_.value(), new_keys}, 2);
        values_ = mx::concatenate({values_.value(), new_values}, 2);

        int total = keys_.value().shape(2);
        if (total > max_size_) {
            int excess = total - max_size_;
            if (excess > 0 && keep_ < total) {
                auto prefix_k = mx::slice(keys_.value(), mx::Shape{0, 0, 0, 0},
                    {keys_.value().shape(0), keys_.value().shape(1), keep_, keys_.value().shape(3)});
                auto suffix_k = mx::slice(keys_.value(), {0, 0, keep_ + excess, 0},
                    {keys_.value().shape(0), keys_.value().shape(1), total, keys_.value().shape(3)});
                keys_ = mx::concatenate({prefix_k, suffix_k}, 2);

                auto prefix_v = mx::slice(values_.value(), mx::Shape{0, 0, 0, 0},
                    {values_.value().shape(0), values_.value().shape(1), keep_, values_.value().shape(3)});
                auto suffix_v = mx::slice(values_.value(), {0, 0, keep_ + excess, 0},
                    {values_.value().shape(0), values_.value().shape(1), total, values_.value().shape(3)});
                values_ = mx::concatenate({prefix_v, suffix_v}, 2);
            }
        }
        idx_ = keys_.value().shape(2);
    }

    offset_ += n_new;
    return {keys_.value(), values_.value()};
}

// --- QuantizedKVCache ---

static QuantizedKVCache::QTuple quantize_kv(
    const mx::array& tensor, int group_size, int bits)
{
    auto shape = tensor.shape();
    int B = shape[0], H = shape[1], T = shape[2], D = shape[3];
    auto flat = mx::reshape(tensor, {B * H * T, D});
    auto result = mx::quantize(flat, group_size, bits);
    auto& w = result[0]; auto& s = result[1]; auto& b = result[2];
    int packed_D = w.shape(1); int scales_D = s.shape(1);
    w = mx::reshape(w, {B, H, T, packed_D});
    s = mx::reshape(s, {B, H, T, scales_D});
    b = mx::reshape(b, {B, H, T, scales_D});
    return {w, s, b};
}

static mx::array dequantize_kv(
    const QuantizedKVCache::QTuple& qt, int group_size, int bits)
{
    auto shape = qt.weight.shape();
    int B = shape[0], H = shape[1], T = shape[2], packed_D = shape[3];
    auto flat_w = mx::reshape(qt.weight, {B * H * T, packed_D});
    auto flat_s = mx::reshape(qt.scales, {B * H * T, qt.scales.shape(3)});
    auto flat_b = mx::reshape(qt.biases, {B * H * T, qt.biases.shape(3)});
    auto deq = mx::dequantize(flat_w, flat_s, flat_b, group_size, bits);
    int D = deq.shape(1);
    return mx::reshape(deq, {B, H, T, D});
}

std::pair<mx::array, mx::array>
QuantizedKVCache::update_impl(
    const mx::array& new_keys,
    const mx::array& new_values)
{
    auto qk = quantize_kv(new_keys, group_size_, bits_);
    auto qv = quantize_kv(new_values, group_size_, bits_);

    if (keys_.has_value()) {
        keys_ = QTuple{
            mx::concatenate({keys_->weight, qk.weight}, 2),
            mx::concatenate({keys_->scales, qk.scales}, 2),
            mx::concatenate({keys_->biases, qk.biases}, 2)
        };
        values_ = QTuple{
            mx::concatenate({values_->weight, qv.weight}, 2),
            mx::concatenate({values_->scales, qv.scales}, 2),
            mx::concatenate({values_->biases, qv.biases}, 2)
        };
    } else {
        keys_ = qk;
        values_ = qv;
    }

    offset_ += new_keys.shape(2);
    return {dequantize_kv(keys_.value(), group_size_, bits_),
            dequantize_kv(values_.value(), group_size_, bits_)};
}

QuantizedKVCache QuantizedKVCache::from_simple(
    const KVCacheSimple& simple, int group_size, int bits)
{
    QuantizedKVCache qc(group_size, bits);
    qc.offset_ = simple.offset();
    if (simple.raw_keys().has_value())
        qc.keys_ = quantize_kv(simple.raw_keys().value(), group_size, bits);
    if (simple.raw_values().has_value())
        qc.values_ = quantize_kv(simple.raw_values().value(), group_size, bits);
    return qc;
}

void maybe_quantize_kv_cache(
    std::vector<KVCache>& cache,
    std::optional<int> kv_bits,
    int kv_group_size,
    int quantized_kv_start)
{
    if (!kv_bits.has_value()) return;
    if (cache.empty()) return;
    if (cache[0].is_quantized()) return;
    if (cache[0].offset() <= quantized_kv_start) return;

    for (size_t i = 0; i < cache.size(); i++) {
        auto& c = cache[i];
        if (!c.is_quantized() && c.offset() > 0) {
            auto st = c.state();
            if (st.size() == 2) {
                QuantizedKVCache qc(kv_group_size, kv_bits.value());
                qc.update(st[0], st[1]);
                cache[i] = KVCache(std::move(qc));
            }
        }
    }
}

} // namespace mlx_lm
