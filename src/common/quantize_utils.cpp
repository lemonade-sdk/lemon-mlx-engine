// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/quantize_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

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
        auto layer_quant = plq.quantization_for(prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
        }

        // Get scales and optional biases
        auto& scales = weights.at(scales_key);
        std::optional<mx::array> biases;
        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            biases = biases_it->second;
        }

        // Embedding weights use mx::take() for lookup, not matmul.
        // They must be dequantized at load time (quantized_matmul won't help).
        bool is_embedding = (prefix.find("embed") != std::string::npos);

        if (is_embedding) {
            // Dequantize in-place so load_weights() gets the float weight
            auto& packed = weights.at(weight_key);
            packed = mx::dequantize(packed, scales, biases, group_size, bits);
        } else {
            // Find the model's member array address for this weight
            auto wm_it = weight_map.find(weight_key);
            if (wm_it == weight_map.end()) {
                // Not in weight_map — can't register, skip
                continue;
            }
            mx::array* member_ptr = wm_it->second;
            reg.register_weight(member_ptr, scales, biases, group_size, bits);
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
        auto layer_quant = plq.quantization_for(prefix);
        if (layer_quant.has_value()) {
            group_size = layer_quant->group_size;
            bits = layer_quant->bits;
        }

        auto biases_it = weights.find(biases_key);
        if (biases_it != weights.end()) {
            weight = mx::dequantize(weight, scales, biases_it->second,
                                    group_size, bits);
            weights.erase(biases_it);
        } else {
            weight = mx::dequantize(weight, scales, std::nullopt,
                                    group_size, bits);
        }

        weights.erase(scales_key);
    }

    return weights;
}

} // namespace mlx_lm
