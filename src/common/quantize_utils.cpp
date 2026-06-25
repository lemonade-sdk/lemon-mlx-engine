// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/quantize_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

static mx::array dequantize_1bit(
    const mx::array& packed,
    const mx::array& scales,
    const mx::array& biases,
    int group_size,
    int in_features)
{
    auto p = mx::astype(packed, mx::int32);
    std::vector<mx::array> bit_planes;
    bit_planes.reserve(32);
    for (int i = 0; i < 32; ++i) {
        auto b = mx::bitwise_and(mx::right_shift(p, mx::array(i)), mx::array(1));
        bit_planes.push_back(b);
    }

    // Keep each uint32's 32 consecutive values together in the output row.
    auto unpacked = mx::reshape(mx::stack(bit_planes, -1), {packed.shape(0), in_features});
    auto values = mx::astype(unpacked, mx::float16);

    int num_groups = in_features / group_size;
    auto scales_expanded = mx::broadcast_to(
        mx::reshape(scales, {scales.shape(0), num_groups, 1}),
        {scales.shape(0), num_groups, group_size});
    scales_expanded = mx::reshape(scales_expanded, {scales.shape(0), in_features});

    auto biases_expanded = mx::broadcast_to(
        mx::reshape(biases, {biases.shape(0), num_groups, 1}),
        {biases.shape(0), num_groups, group_size});
    biases_expanded = mx::reshape(biases_expanded, {biases.shape(0), in_features});

    return mx::add(mx::multiply(values, scales_expanded), biases_expanded);
}

void register_quantized_weights(
    std::unordered_map<std::string, mx::array>& weights,
    const BaseConfiguration& config,
    const std::unordered_map<std::string, mx::array*>& weight_map)
{
    static const bool dbg = std::getenv("MLX_DEBUG_QUANT") != nullptr;
    if (!config.per_layer_quantization.has_value()) {
        if (dbg) std::cerr << "[quant-dbg] per_layer_quantization NOT SET -> "
                              "NOTHING registered (model will run dense on packed "
                              "weights = garbage)\n";
        return;
    }

    auto& plq = config.per_layer_quantization.value();
    if (!plq.default_quantization.has_value()) return;

    int default_group_size = plq.default_quantization->group_size;
    int default_bits = plq.default_quantization->bits;
    QuantizationMode default_mode = plq.default_quantization->mode;

    auto& reg = QuantizedWeightRegistry::instance();

    // Collect prefixes that have .scales (indicating quantized weights)
    std::vector<std::string> prefixes;
    for (auto& [key, _] : weights) {
        const std::string suffix = ".scales";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            auto prefix = key.substr(0, key.size() - suffix.size());
            if (weights.count(prefix + ".weight")) {
                prefixes.push_back(prefix);
            }
        }
    }

    for (auto& prefix : prefixes) {
        auto weight_key = prefix + ".weight";
        auto scales_key = prefix + ".scales";
        auto biases_key = prefix + ".biases";

        // Check per-layer quantization overrides
        int group_size = default_group_size;
        int bits = default_bits;
        QuantizationMode mode = default_mode;
        auto layer_quant = plq.quantization_for(prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
            mode = layer_quant->mode;
        }

        std::string mode_str = (mode == QuantizationMode::Mxfp4) ? "mxfp4" : "affine";

        // Get scales and optional biases
        auto& scales = weights.at(scales_key);
        std::optional<mx::array> biases;
        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            biases = biases_it->second;
        }

        // Embedding weights use mx::take() for lookup, not matmul.
        // They must be dequantized at load time (quantized_matmul won't help).
        // MLX GPU affine dequantize/quantized_matmul does not support 1-bit,
        // so 1-bit affine weights also need to become dense at load time.
        // MXFP4 mode is not supported by the ROCm quantized_matmul/gather_qmm
        // backends (they only support Affine), so dequantize at load time.
        bool is_embedding = (prefix.find("embed") != std::string::npos);
        bool is_mxfp4 = (mode == QuantizationMode::Mxfp4);
        bool needs_loadtime_dequant = is_embedding || (bits == 1) || is_mxfp4;

        if (needs_loadtime_dequant) {
            // Dequantize in-place so load_weights() gets the float weight
            auto& packed = weights.at(weight_key);
            if (bits == 1) {
                if (!biases.has_value()) {
                    throw std::runtime_error("1-bit affine quantized weights require biases");
                }
                int in_features = packed.shape(1) * 32;
                packed = dequantize_1bit(packed, scales, *biases, group_size, in_features);
            } else if (is_mxfp4) {
                // MXFP4: no biases, uint8 scales. Dequantize using fp mode.
                packed = mx::dequantize(packed, scales, std::nullopt,
                                        group_size, bits, /*mode=*/"mxfp4");
            } else {
                packed = mx::dequantize(packed, scales, biases, group_size, bits);
            }
        } else {
            // Find the model's member array address for this weight
            auto wm_it = weight_map.find(weight_key);
            if (wm_it == weight_map.end()) {
                // Not in weight_map — can't register, skip
                continue;
            }
            mx::array* member_ptr = wm_it->second;
            reg.register_weight(member_ptr, scales, biases, group_size, bits, mode_str);
        }

        // Remove scales/biases from the weight map so they don't get
        // loaded as regular weights
        weights.erase(scales_key);
        if (biases_it != weights.end()) {
            weights.erase(biases_it);
        }
    }
    if (dbg) {
        int registered = 0, missed = 0, embed = 0;
        // recompute for reporting (cheap): count prefixes still resolvable
        std::cerr << "[quant-dbg] processed " << prefixes.size()
                  << " quantized prefixes; registry size now "
                  << QuantizedWeightRegistry::instance().size() << "\n";
        (void)registered; (void)missed; (void)embed;
    }
}

// Legacy dequantize-at-load-time (kept for reference/fallback)
std::unordered_map<std::string, mx::array> dequantize_weights(
    std::unordered_map<std::string, mx::array> weights,
    const BaseConfiguration& config)
{
    if (!config.per_layer_quantization.has_value()) return weights;

    auto& plq = config.per_layer_quantization.value();
    if (!plq.default_quantization.has_value()) return weights;

    int default_group_size = plq.default_quantization->group_size;
    int default_bits = plq.default_quantization->bits;
    QuantizationMode default_mode = plq.default_quantization->mode;

    std::vector<std::string> prefixes;
    for (auto& [key, _] : weights) {
        const std::string suffix = ".scales";
        if (key.size() > suffix.size() &&
            key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0) {
            auto prefix = key.substr(0, key.size() - suffix.size());
            if (weights.count(prefix + ".weight")) {
                prefixes.push_back(prefix);
            }
        }
    }

    for (auto& prefix : prefixes) {
        auto weight_key = prefix + ".weight";
        auto scales_key = prefix + ".scales";
        auto biases_key = prefix + ".biases";

        auto& weight = weights.at(weight_key);
        auto& scales = weights.at(scales_key);

        int group_size = default_group_size;
        int bits = default_bits;
        QuantizationMode mode = default_mode;
        auto layer_quant = plq.quantization_for(prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
            mode = layer_quant->mode;
        }

        std::string mode_str = (mode == QuantizationMode::Mxfp4) ? "mxfp4" : "affine";

        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            weight = mx::dequantize(weight, scales, biases_it->second,
                                    group_size, bits, mode_str);
            weights.erase(biases_it);
        } else {
            weight = mx::dequantize(weight, scales, std::nullopt,
                                    group_size, bits, mode_str);
        }

        weights.erase(scales_key);
    }

    return weights;
}

} // namespace mlx_lm
