// Copyright © 2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/base_config.h>
#include <set>
#include <string>

namespace mlx_lm {

BaseConfiguration parse_base_configuration(const nlohmann::json& config) {
    BaseConfiguration base;
    base.model_type = config.value("model_type", std::string(""));

    if (config.contains("eos_token_id")) {
        IntOrIntArray eos;
        from_json(config["eos_token_id"], eos);
        base.eos_token_ids = eos;
    }

    // Check for BitNet quantization — BitNet handles its own repacking internally.
    // quant_method can appear inside either "quantization" or "quantization_config".
    auto get_quant_method = [](const nlohmann::json& c) -> std::string {
        if (c.contains("quantization") && c["quantization"].contains("quant_method"))
            return c["quantization"]["quant_method"].get<std::string>();
        if (c.contains("quantization_config") && c["quantization_config"].contains("quant_method"))
            return c["quantization_config"]["quant_method"].get<std::string>();
        return std::string();
    };
    if (get_quant_method(config) == "bitnet") {
        return base;
    }

    // Helper to build PerLayerQuantization from a quantization JSON object.
    // This is used for both "quantization" (MLX format) and
    // "quantization_config" (HuggingFace format).
    auto build_per_layer_quantization = [](const nlohmann::json& q_json) {
        Quantization default_quant;
        default_quant.group_size = q_json.value("group_size", 64);
        default_quant.bits = q_json.value("bits", 4);
        auto mode_str = q_json.value("mode", std::string("affine"));
        if (mode_str == "mxfp4") {
            default_quant.mode = QuantizationMode::Mxfp4;
        } else {
            default_quant.mode = QuantizationMode::Affine;
        }

        PerLayerQuantization plq;
        plq.default_quantization = default_quant;

        // Known non-layer keys to skip
        static const std::set<std::string> skip_keys = {
            "group_size", "bits", "mode",
            "quant_method", "linear_class", "quantization_mode"
        };

        // VLM wrappers (Qwen3.5/3.6-MoE multimodal checkpoints) prefix every
        // language-model weight with "language_model." in safetensors AND in
        // the quantization config. Our per-model sanitize strips that prefix
        // from the weights so they bind to the bare LM module tree; mirror
        // the same strip here so the per-layer overrides line up with the
        // sanitized weight names and the 8-bit MoE-gate overrides actually
        // take effect (otherwise the lookup misses and the layer falls back
        // to the default 4-bit bits, producing a shape mismatch at load).
        static const std::string kVlmPrefix = "language_model.";
        for (auto& [key, value] : q_json.items()) {
            if (skip_keys.count(key)) continue;

            std::string layer_key = key;
            if (layer_key.compare(0, kVlmPrefix.size(), kVlmPrefix) == 0) {
                layer_key = layer_key.substr(kVlmPrefix.size());
            }

            if (value.is_boolean()) {
                if (!value.get<bool>()) {
                    plq.per_layer[layer_key] = QuantizationOption::skip();
                }
            } else if (value.is_object()) {
                Quantization layer_quant;
                from_json(value, layer_quant);
                plq.per_layer[layer_key] = QuantizationOption::quantize(layer_quant);
            }
        }

        return plq;
    };

    if (config.contains("quantization")) {
        base.per_layer_quantization = build_per_layer_quantization(config["quantization"]);
    } else if (config.contains("quantization_config")) {
        // HuggingFace format: read from quantization_config instead.
        base.per_layer_quantization = build_per_layer_quantization(config["quantization_config"]);
    }

    return base;
}

} // namespace mlx_lm
