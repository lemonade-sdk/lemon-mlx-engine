// Copyright © 2025 Apple Inc. — Ported to C++
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

namespace mlx_lm {

// Quantization mode.
enum class QuantizationMode {
    Affine,
    Mxfp4,
};

// Quantization parameters.
struct Quantization {
    int group_size = 64;
    int bits = 4;
    QuantizationMode mode = QuantizationMode::Affine;
};

inline void from_json(const nlohmann::json& j, Quantization& q) {
    q.group_size = j.value("group_size", 64);
    q.bits = j.value("bits", 4);
    auto mode_str = j.value("mode", std::string("affine"));
    if (mode_str == "mxfp4") {
        q.mode = QuantizationMode::Mxfp4;
    } else {
        q.mode = QuantizationMode::Affine;
    }
}

// Per-layer quantization option.
enum class QuantizationOptionTag { Skip, Quantize };

struct QuantizationOption {
    QuantizationOptionTag tag;
    Quantization quantization; // only valid if tag == Quantize

    static QuantizationOption skip() { return {QuantizationOptionTag::Skip, {}}; }
    static QuantizationOption quantize(Quantization q) { return {QuantizationOptionTag::Quantize, q}; }
};

// Per-layer quantization map with optional default.
struct PerLayerQuantization {
    std::optional<Quantization> default_quantization;
    std::unordered_map<std::string, QuantizationOption> per_layer;

    std::optional<Quantization> quantization_for(const std::string& layer) const {
        auto it = per_layer.find(layer);
        if (it != per_layer.end()) {
            if (it->second.tag == QuantizationOptionTag::Skip) return std::nullopt;
            return it->second.quantization;
        }
        return default_quantization;
    }
};

// EOS token ID can be a single int or array of ints.
struct IntOrIntArray {
    std::vector<int> values;
};

inline void from_json(const nlohmann::json& j, IntOrIntArray& v) {
    if (j.is_number()) {
        v.values = {j.get<int>()};
    } else if (j.is_array()) {
        v.values = j.get<std::vector<int>>();
    }
}

// Base configuration read from config.json.
struct BaseConfiguration {
    std::string model_type;
    std::optional<PerLayerQuantization> per_layer_quantization;
    std::optional<IntOrIntArray> eos_token_ids;
};

// Parse BaseConfiguration from JSON.
BaseConfiguration parse_base_configuration(const nlohmann::json& config);

} // namespace mlx_lm
