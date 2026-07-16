// Copyright © 2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/base_config.h>
#include <set>
#include <string>

namespace mlx_lm {

BaseConfiguration parse_base_configuration(const nlohmann::json& config) {
    BaseConfiguration base;
    base.model_type = config.value("model_type", std::string(""));

    // Top-level eos_token_id (most LLMs). VLM / hybrid wrappers (Qwen3.5/3.6)
    // nest it under text_config — without this generation never stops and
    // hits max_tokens, replaying answers after <|endoftext|>.
    if (config.contains("eos_token_id")) {
        IntOrIntArray eos;
        from_json(config["eos_token_id"], eos);
        base.eos_token_ids = eos;
    } else if (config.contains("text_config") &&
               config["text_config"].is_object() &&
               config["text_config"].contains("eos_token_id")) {
        IntOrIntArray eos;
        from_json(config["text_config"]["eos_token_id"], eos);
        base.eos_token_ids = eos;
    }

    if (config.contains("quantization")) {
        const auto& q_json = config["quantization"];

        Quantization default_quant;
        default_quant.group_size = q_json.value("group_size", 64);
        default_quant.bits = q_json.value("bits", 4);

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

        base.per_layer_quantization = plq;
    }

    return base;
}

} // namespace mlx_lm
