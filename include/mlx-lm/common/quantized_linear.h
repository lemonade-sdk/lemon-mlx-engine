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

#ifdef MLX_BUILD_NPU
// NPU dispatch for experimental use. Opt-in via NPU_ENABLE=1 env var.
// The NPU path is useful for testing compute-constrained scenarios
// and for running two models in parallel (NPU + GPU).
namespace detail {

// Convert MLX uint32 2-bit packed weights to BitNet U8 ternary format.
// MLX: each uint32 packs 16 × 2-bit codes (code 0,1,2,3)
// BitNet: each uint8 packs 4 × 2-bit codes (code 0→-1, 1→0, 2→+1)
// This is a straight repack since both use the same code→ternary mapping.
static bool repack_2bit_to_u8(
    const mlx::core::array& w_uint32,  // [N, ceil(K/16)] uint32
    std::vector<uint8_t>& out_u8,      // [ceil(N/4), K] uint8
    int N, int K)
{
    mx::eval(w_uint32);
    auto data = w_uint32.data<uint32_t>();
    int cols = w_uint32.shape(1);
    
    int packed_rows = (N + 3) / 4;
    out_u8.assign(packed_rows * K, 0);
    
    for (int oc = 0; oc < N; oc++) {
        int row = oc / 4;
        int lane = oc % 4;
        for (int k = 0; k < K; k++) {
            int word_idx = k / 16;
            int bit_offset = (k % 16) * 2;
            if (word_idx >= cols) continue;
            uint32_t word = data[oc * cols + word_idx];
            int code = (word >> bit_offset) & 0x03;
            if (code > 2) code = 1; // clamp invalid codes (code 3 = 2*scale+bias, shouldn't occur)
            out_u8[row * K + k] |= (code << (lane * 2));
        }
    }
    return true;
}

inline bool npu_try_ternary(
    const mlx::core::array& input,  // [1, K] bf16
    const mlx::core::array& w,      // [N, ceil(K/16)] uint32 (2-bit packed)
    int N, int K,
    const QuantizationInfo* qi,
    mlx::core::array& output)       // [1, N] bf16 output (filled on success)
{
    // Opt-in via NPU_ENABLE=1 (disabled by default)
    static const char* env = std::getenv("NPU_ENABLE");
    static const bool npu_enabled = env && std::string(env) == "1";
    if (!npu_enabled) return false;

    // Only for decode (B=1) path with 2-bit weights
    if (input.ndim() != 2 || input.shape(0) != 1) return false;
    if (w.ndim() != 2 || qi == nullptr || qi->bits != 2) return false;

    static bool npu_checked = false;
    static bool npu_avail = false;
    if (!npu_checked) {
        npu_avail = npu::init();
        npu_checked = true;
    }
    if (!npu_avail) return false;

    mx::eval(qi->scales);
    float ws = (float)qi->scales.data<mx::float16_t>()[0];

    std::vector<uint8_t> packed_u8;
    if (!repack_2bit_to_u8(w, packed_u8, N, K)) return false;

    mx::eval(input);
    auto act_ptr = input.data<mx::float16_t>();
    std::vector<float> acts_f32(K);
    for (int i = 0; i < K; i++) acts_f32[i] = (float)act_ptr[i];

    std::vector<float> result(N);
    if (!npu::ternary_gemv(packed_u8.data(), acts_f32.data(), result.data(),
                            ws, false, N, K)) {
        return false;
    }

    std::vector<mx::float16_t> result_bf16(N);
    for (int i = 0; i < N; i++) result_bf16[i] = (mx::float16_t)result[i];
    output = mx::array(result_bf16.data(), {1, N}, mx::float16);
    mx::eval(output);
    
    std::fprintf(stderr, "[NPU] Ternary GEMV %dx%d done ✅\n", N, K);
    return true;
}
} // namespace detail
#endif

// Quantization-aware linear forward pass.
//
// If the weight is registered as quantized, uses mx::quantized_matmul.
// May fall back to NPU dispatch for ternary (2-bit) decode when NPU is
// available (experimental, gated by NPU_DISABLE env var).
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
#ifdef MLX_BUILD_NPU
        // Try NPU dispatch for ternary (2-bit) decode path
        if (qi->bits == 2) {
            mlx::core::array npu_result;
            if (detail::npu_try_ternary(input, w, (int)w.shape(0), (int)input.shape(1), qi, npu_result)) {
                if (bias) npu_result = mlx::core::add(npu_result, *bias);
                return npu_result;
            }
        }
#endif
        // GPU path: quantized_matmul
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
