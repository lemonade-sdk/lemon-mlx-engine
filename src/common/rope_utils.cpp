// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of RoPEUtils.swift and SuScaledRoPE.swift

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <mlx-lm/common/rope_utils.h>
#include <cmath>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ===========================================================================
// Llama3RoPE
// ===========================================================================

Llama3RoPE::Llama3RoPE(
    int dims,
    int max_position_embeddings,
    bool traditional,
    float base,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& scaling_config)
    : dims_(dims),
      max_position_embeddings_(max_position_embeddings),
      traditional_(traditional),
      freqs_(mx::array(0.0f))  // placeholder, computed below
{
    if (!scaling_config.has_value()) {
        throw std::runtime_error("Llama3RoPE requires scaling_config");
    }

    const auto& sc = scaling_config.value();

    auto get_float = [&](const std::string& key, float default_val) -> float {
        auto it = sc.find(key);
        if (it != sc.end() && it->second.is_float()) return it->second.as_float();
        return default_val;
    };

    float factor = get_float("factor", 1.0f);
    float low_freq_factor = get_float("low_freq_factor", 1.0f);
    float high_freq_factor = get_float("high_freq_factor", 4.0f);
    float old_context_len = get_float("original_max_position_embeddings", 8192.0f);

    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;

    // indices = [0, 2, 4, ..., dims-2]
    auto indices = mx::arange(0, dims, 2);
    // frequencies = base^(indices / dims)
    auto frequencies = mx::power(
        mx::array(base),
        mx::divide(mx::astype(indices, mx::float32), mx::array(static_cast<float>(dims))));
    // wavelens = 2 * pi * frequencies
    auto wavelens = mx::multiply(mx::array(2.0f * static_cast<float>(M_PI)), frequencies);

    // Where wavelen > low_freq_wavelen: scale frequencies by factor
    frequencies = mx::where(
        mx::greater(wavelens, mx::array(low_freq_wavelen)),
        mx::multiply(frequencies, mx::array(factor)),
        frequencies);

    // Medium frequency band: wavelen > high_freq_wavelen AND wavelen < low_freq_wavelen
    auto is_medium_freq = mx::logical_and(
        mx::greater(wavelens, mx::array(high_freq_wavelen)),
        mx::less(wavelens, mx::array(low_freq_wavelen)));

    // smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
    auto smooth_factors = mx::divide(
        mx::subtract(
            mx::divide(mx::array(old_context_len), wavelens),
            mx::array(low_freq_factor)),
        mx::array(high_freq_factor - low_freq_factor));

    // smooth_freqs = frequencies / ((1 - smooth_factors) / factor + smooth_factors)
    auto smooth_freqs = mx::divide(
        frequencies,
        mx::add(
            mx::divide(mx::subtract(mx::array(1.0f), smooth_factors), mx::array(factor)),
            smooth_factors));

    freqs_ = mx::where(is_medium_freq, smooth_freqs, frequencies);
}

mx::array Llama3RoPE::operator()(const mx::array& x, int offset) {
    return mx::fast::rope(
        x,
        dims_,
        traditional_,
        std::optional<float>{},  // base = nil (using freqs instead)
        1.0f,                     // scale
        offset,
        freqs_);
}

// ===========================================================================
// YarnRoPE
// ===========================================================================

// Helper: find the correction dimension for a given number of rotations.
static float yarn_find_correction_dim(
    int dims, int original_max_position_embeddings, float base, float num_rotations) {
    return static_cast<float>(dims)
        * std::log(static_cast<float>(original_max_position_embeddings) / (num_rotations * 2.0f * static_cast<float>(M_PI)))
        / (2.0f * std::log(base));
}

// Helper: find the correction range (low, high) indices.
static std::pair<int, int> yarn_find_correction_range(
    int dims, int original_max_position_embeddings, float base,
    float beta_fast, float beta_slow) {
    int low = static_cast<int>(
        std::floor(yarn_find_correction_dim(dims, original_max_position_embeddings, base, beta_fast)));
    int high = static_cast<int>(
        std::ceil(yarn_find_correction_dim(dims, original_max_position_embeddings, base, beta_slow)));
    return {std::max(low, 0), std::min(high, dims - 1)};
}

// Helper: compute mscale from scale and mscale parameter.
static float yarn_get_mscale(float scale, float mscale) {
    if (scale <= 1.0f) return 1.0f;
    return 0.1f * mscale * std::log(scale) + 1.0f;
}

// Helper: create a linear ramp mask [0..dim) clamped to [0, 1].
static mx::array yarn_linear_ramp_mask(float min_val, float max_val, int dim) {
    float max_v = max_val;
    if (min_val == max_v) {
        max_v += 0.001f;
    }
    // linear_func = (arange(dim) - min_val) / (max_v - min_val)
    auto linear_func = mx::divide(
        mx::subtract(mx::astype(mx::arange(dim), mx::float32), mx::array(min_val)),
        mx::array(max_v - min_val));
    return mx::clip(linear_func, mx::array(0.0f), mx::array(1.0f));
}

YarnRoPE::YarnRoPE(
    int dims,
    bool traditional,
    int max_position_embeddings,
    float base,
    float scaling_factor,
    int original_max_position_embeddings,
    float beta_fast,
    float beta_slow,
    float mscale,
    float mscale_all_dim)
    : dims_(dims),
      traditional_(traditional),
      max_position_embeddings_(max_position_embeddings),
      base_(base),
      scaling_factor_(scaling_factor),
      original_max_position_embeddings_(original_max_position_embeddings),
      beta_fast_(beta_fast),
      beta_slow_(beta_slow),
      mscale_(mscale),
      mscale_all_dim_(mscale_all_dim),
      computed_mscale_(1.0f),
      freqs_(mx::array(0.0f))  // placeholder, computed below
{
    if (dims % 2 != 0) {
        throw std::runtime_error("YarnRoPE: dimensions must be even");
    }

    // Compute mscale
    computed_mscale_ = yarn_get_mscale(scaling_factor, mscale)
                     / yarn_get_mscale(scaling_factor, mscale_all_dim);

    // Exponent indices: [0, 2, 4, ..., dims-2] / dims
    auto exponent = mx::divide(
        mx::astype(mx::arange(0, dims, 2), mx::float32),
        mx::array(static_cast<float>(dims)));

    // freq_extra = base^exponent (extrapolation frequencies)
    auto freq_extra = mx::power(mx::array(base), exponent);
    // freq_inter = scaling_factor * base^exponent (interpolation frequencies)
    auto freq_inter = mx::multiply(mx::array(scaling_factor), mx::power(mx::array(base), exponent));

    // Find correction range
    auto [low, high] = yarn_find_correction_range(
        dims, original_max_position_embeddings, base, beta_fast, beta_slow);

    // freq_mask = 1.0 - linear_ramp_mask(low, high, dims/2)
    auto freq_mask = mx::subtract(
        mx::array(1.0f),
        yarn_linear_ramp_mask(static_cast<float>(low), static_cast<float>(high), dims / 2));

    // freqs = (freq_inter * freq_extra) / (freq_inter * freq_mask + freq_extra * (1 - freq_mask))
    auto numerator = mx::multiply(freq_inter, freq_extra);
    auto denominator = mx::add(
        mx::multiply(freq_inter, freq_mask),
        mx::multiply(freq_extra, mx::subtract(mx::array(1.0f), freq_mask)));
    freqs_ = mx::divide(numerator, denominator);
}

mx::array YarnRoPE::operator()(const mx::array& x, int offset) {
    mx::array input = x;

    if (computed_mscale_ != 1.0f) {
        // Scale only the dimensions that will be rotated: x[..., :dims] *= mscale
        // We need to split, scale, and reassemble
        int last_dim = x.shape(-1);
        if (dims_ < last_dim) {
            auto rotated = mx::slice(x, {0, 0, 0, 0},
                {x.shape(0), x.shape(1), x.shape(2), dims_});
            auto passthrough = mx::slice(x, {0, 0, 0, dims_},
                {x.shape(0), x.shape(1), x.shape(2), last_dim});
            rotated = mx::multiply(mx::array(computed_mscale_), rotated);
            input = mx::concatenate({rotated, passthrough}, -1);
        } else {
            input = mx::multiply(mx::array(computed_mscale_), x);
        }
    }

    return mx::fast::rope(
        input,
        dims_,
        traditional_,
        std::optional<float>{},  // base = nil (using freqs)
        1.0f,                     // scale
        offset,
        freqs_);
}

// ===========================================================================
// SuScaledRoPE
// ===========================================================================

// Helper: compute default scale for SuScaledRoPE
static float su_default_scale(float factor, int original_max_position_embeddings) {
    return std::sqrt(
        1.0f + std::log(factor) / std::log(static_cast<float>(original_max_position_embeddings)));
}

SuScaledRoPE::SuScaledRoPE(
    int dims,
    float base,
    int max_position_embeddings,
    int original_max_position_embeddings,
    const std::vector<float>& short_factor,
    const std::vector<float>& long_factor,
    std::optional<float> short_m_scale,
    std::optional<float> long_m_scale)
    : dims_(dims),
      original_max_position_embeddings_(original_max_position_embeddings),
      short_freqs_(mx::array(0.0f)),   // placeholder
      long_freqs_(mx::array(0.0f)),    // placeholder
      short_scale_(1.0f),
      long_scale_(1.0f)
{
    if (dims % 2 != 0) {
        throw std::runtime_error("SuScaledRoPE: dimensions must be even");
    }

    // exponent = arange(0, dims, 2) / dims
    auto exponent = mx::divide(
        mx::astype(mx::arange(0, dims, 2), mx::float32),
        mx::array(static_cast<float>(dims)));
    // freqs = base^exponent
    auto freqs = mx::power(mx::array(base), exponent);

    // short_freqs = short_factor * freqs
    short_freqs_ = mx::multiply(
        mx::astype(mx::array(short_factor.data(), {static_cast<int>(short_factor.size())}), mx::float32),
        freqs);
    // long_freqs = long_factor * freqs
    long_freqs_ = mx::multiply(
        mx::astype(mx::array(long_factor.data(), {static_cast<int>(long_factor.size())}), mx::float32),
        freqs);

    // Compute scales
    float factor = static_cast<float>(max_position_embeddings) /
                   static_cast<float>(original_max_position_embeddings);

    if (short_m_scale.has_value()) {
        short_scale_ = short_m_scale.value();
    } else {
        short_scale_ = (factor <= 1.0f) ? 1.0f
            : su_default_scale(factor, original_max_position_embeddings);
    }

    if (long_m_scale.has_value()) {
        long_scale_ = long_m_scale.value();
    } else {
        long_scale_ = (factor <= 1.0f) ? 1.0f
            : su_default_scale(factor, original_max_position_embeddings);
    }
}

mx::array SuScaledRoPE::operator()(const mx::array& x, int offset) {
    int seq_len = offset + x.shape(-2);

    const mx::array& freqs = (seq_len > original_max_position_embeddings_) ? long_freqs_ : short_freqs_;
    float scale = (seq_len > original_max_position_embeddings_) ? long_scale_ : short_scale_;

    mx::array input = x;
    if (scale != 1.0f) {
        // Scale only the rotated dimensions: x[..., :dims] *= scale
        int last_dim = x.shape(-1);
        if (dims_ < last_dim) {
            auto rotated = mx::slice(x, {0, 0, 0, 0},
                {x.shape(0), x.shape(1), x.shape(2), dims_});
            auto passthrough = mx::slice(x, {0, 0, 0, dims_},
                {x.shape(0), x.shape(1), x.shape(2), last_dim});
            rotated = mx::multiply(mx::array(scale), rotated);
            input = mx::concatenate({rotated, passthrough}, -1);
        } else {
            input = mx::multiply(mx::array(scale), x);
        }
    }

    return mx::fast::rope(
        input,
        dims_,
        false,                    // traditional = false (per Swift implementation)
        std::optional<float>{},   // base = nil (using freqs)
        1.0f,                     // scale
        offset,
        freqs);
}

// ===========================================================================
// SimpleRoPE
// ===========================================================================

mx::array SimpleRoPE::operator()(const mx::array& x, int offset) {
    return mx::fast::rope(x, dims, traditional, base, scale, offset);
}

// ===========================================================================
// apply_rope — dispatch to the active variant
// ===========================================================================

mx::array apply_rope(RoPEVariant& rope, const mx::array& x, int offset) {
    return std::visit([&](auto& r) -> mx::array {
        return r(x, offset);
    }, rope);
}

// ===========================================================================
// initialize_rope — factory function
// ===========================================================================

RoPEVariant initialize_rope(
    int dims,
    float base,
    bool traditional,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& scaling_config,
    std::optional<int> max_position_embeddings,
    const nlohmann::json* rope_scaling_json)
{
    // Determine rope type from config
    std::string rope_type = "default";
    if (scaling_config.has_value()) {
        const auto& sc = scaling_config.value();
        auto it = sc.find("type");
        if (it == sc.end()) it = sc.find("rope_type");
        if (it != sc.end() && it->second.is_string()) {
            rope_type = it->second.as_string();
        }
    }

    if (rope_type == "default" || rope_type == "linear") {
        float scale = 1.0f;
        if (rope_type == "linear" && scaling_config.has_value()) {
            auto it = scaling_config->find("factor");
            if (it != scaling_config->end() && it->second.is_float()) {
                scale = 1.0f / it->second.as_float();
            }
        }
        return SimpleRoPE{dims, traditional, base, scale};

    } else if (rope_type == "llama3") {
        return Llama3RoPE(
            dims,
            max_position_embeddings.value_or(2048),
            traditional,
            base,
            scaling_config);

    } else if (rope_type == "yarn") {
        float factor = 32.0f;
        int orig_max = 4096;
        float beta_fast = 32.0f;
        float beta_slow = 1.0f;
        float mscale = 1.0f;
        float mscale_all_dim = 0.0f;

        if (scaling_config.has_value()) {
            const auto& sc = scaling_config.value();
            auto get_f = [&](const std::string& key, float def) -> float {
                auto it = sc.find(key);
                if (it != sc.end() && it->second.is_float()) return it->second.as_float();
                return def;
            };
            factor = get_f("factor", 32.0f);
            float orig_f = get_f("original_max_position_embeddings", 4096.0f);
            orig_max = static_cast<int>(orig_f);
            beta_fast = get_f("beta_fast", 32.0f);
            beta_slow = get_f("beta_slow", 1.0f);
            mscale = get_f("mscale", 1.0f);
            mscale_all_dim = get_f("mscale_all_dim", 0.0f);
        }

        return YarnRoPE(
            dims,
            traditional,
            max_position_embeddings.value_or(2048),
            base,
            factor,
            orig_max,
            beta_fast,
            beta_slow,
            mscale,
            mscale_all_dim);

    } else if (rope_type == "longrope") {
        if (!scaling_config.has_value()) {
            throw std::runtime_error("longrope requires scaling_config");
        }
        const auto& sc = scaling_config.value();

        // original_max_position_embeddings from StringOrNumber map
        auto orig_it = sc.find("original_max_position_embeddings");
        if (orig_it == sc.end() || !orig_it->second.is_float()) {
            throw std::runtime_error("longrope requires original_max_position_embeddings");
        }
        int orig_max = static_cast<int>(orig_it->second.as_float());

        // short_factor and long_factor are arrays — extract from JSON if available
        std::vector<float> short_factor;
        std::vector<float> long_factor;

        if (rope_scaling_json != nullptr) {
            if (rope_scaling_json->contains("short_factor")) {
                short_factor = (*rope_scaling_json)["short_factor"].get<std::vector<float>>();
            } else {
                throw std::runtime_error("longrope requires short_factor");
            }
            if (rope_scaling_json->contains("long_factor")) {
                long_factor = (*rope_scaling_json)["long_factor"].get<std::vector<float>>();
            } else {
                throw std::runtime_error("longrope requires long_factor");
            }
        } else {
            throw std::runtime_error(
                "longrope requires rope_scaling_json to extract short_factor/long_factor arrays");
        }

        return SuScaledRoPE(
            dims,
            base,
            max_position_embeddings.value_or(131072),
            orig_max,
            short_factor,
            long_factor);

    } else if (rope_type == "mrope") {
        // MRoPE returns basic RoPE. The actual multi-modal rotary embedding logic
        // is handled in the attention layer of multimodal models (e.g. Qwen2VL).
        return SimpleRoPE{dims, traditional, base, 1.0f};

    } else {
        throw std::runtime_error("Unsupported RoPE type: " + rope_type);
    }
}

} // namespace mlx_lm
