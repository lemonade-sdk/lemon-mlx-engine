// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/vlm/vlm_factory.h>
#include <mlx-lm/vlm/models/qwen2_vl.h>
#include <mlx-lm/vlm/models/paligemma.h>
#include <mlx-lm/vlm/models/idefics3.h>
#include <mlx-lm/vlm/models/qwen25_vl.h>
#include <mlx-lm/vlm/models/gemma3.h>
#include <mlx-lm/vlm/models/qwen3_vl.h>
#include <mlx-lm/vlm/models/pixtral.h>
#include <mlx-lm/vlm/models/mistral3.h>
#include <mlx-lm/vlm/models/lfm2_vl.h>
#include <mlx-lm/vlm/models/fastvlm.h>
#include <mlx-lm/vlm/models/qwen35_vl.h>
#include <mlx-lm/common/base_config.h>
#include <mlx-lm/common/safetensors.h>
#include <mlx-lm/common/hub_api.h>
#include <mlx-lm/common/tokenizer.h>
#include <mlx-lm/common/quantize_utils.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

// Helper: create a typed model from JSON config data (for ModelTypeRegistry).
template <typename Config, typename Model>
static void* create_model(const std::string& config_json) {
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    return new Model(config);
}

// Helper: create, sanitize, load weights, and return an owned ModelContext.
using VLMLoaderFn = std::function<ModelContext(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights)>;

template <typename Config, typename Model>
static ModelContext load_typed_model(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights)
{
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    auto model = std::make_shared<Model>(config);

    weights = model->sanitize(std::move(weights));
    model->load_weights(weights);

    return ModelContext::from_model_owned(model);
}

// Internal loader registry
static std::unordered_map<std::string, VLMLoaderFn>& vlm_loaders() {
    static std::unordered_map<std::string, VLMLoaderFn> loaders = {
        {"qwen2_vl",   load_typed_model<Qwen2VLConfiguration, Qwen2VLModel>},
        {"paligemma",  load_typed_model<PaliGemmaConfiguration, PaliGemmaModel>},
        {"idefics3",   load_typed_model<Idefics3Configuration, Idefics3Model>},
        {"smolvlm",    load_typed_model<Idefics3Configuration, Idefics3Model>},
        {"qwen2_5_vl", load_typed_model<Qwen25VLConfiguration, Qwen25VLModel>},
        {"gemma3",     load_typed_model<Gemma3Configuration, Gemma3Model>},
        {"qwen3_vl",   load_typed_model<Qwen3VLConfiguration, Qwen3VLModel>},
        {"pixtral",    load_typed_model<PixtralConfiguration, PixtralModel>},
        {"mistral3",   load_typed_model<Mistral3VLMConfiguration, Mistral3Model>},
        {"lfm2_vl",    load_typed_model<LFM2VLConfiguration, LFM2VLModel>},
        {"fastvlm",    load_typed_model<FastVLMConfiguration, FastVLMModel>},
        {"llava_qwen2", load_typed_model<FastVLMConfiguration, FastVLMModel>},
        {"qwen3_5",     load_typed_model<Qwen35VLConfiguration, Qwen35VLModel>},
    };
    return loaders;
}

// --- VLM Type Registry (public) ---

ModelTypeRegistry& vlm_type_registry() {
    static ModelTypeRegistry registry({
        {"qwen2_vl",   create_model<Qwen2VLConfiguration, Qwen2VLModel>},
        {"paligemma",  create_model<PaliGemmaConfiguration, PaliGemmaModel>},
        {"idefics3",   create_model<Idefics3Configuration, Idefics3Model>},
        {"smolvlm",    create_model<Idefics3Configuration, Idefics3Model>},
        {"qwen2_5_vl", create_model<Qwen25VLConfiguration, Qwen25VLModel>},
        {"gemma3",     create_model<Gemma3Configuration, Gemma3Model>},
        {"qwen3_vl",   create_model<Qwen3VLConfiguration, Qwen3VLModel>},
        {"pixtral",    create_model<PixtralConfiguration, PixtralModel>},
        {"mistral3",   create_model<Mistral3VLMConfiguration, Mistral3Model>},
        {"lfm2_vl",    create_model<LFM2VLConfiguration, LFM2VLModel>},
        {"fastvlm",    create_model<FastVLMConfiguration, FastVLMModel>},
        {"llava_qwen2", create_model<FastVLMConfiguration, FastVLMModel>},
        {"qwen3_5",     create_model<Qwen35VLConfiguration, Qwen35VLModel>},
    });
    return registry;
}

// --- VLM Model Registry ---

AbstractModelRegistry& vlm_model_registry() {
    static AbstractModelRegistry registry({
        {"mlx-community/Qwen2-VL-2B-Instruct-4bit",
            "Describe the image in English"},
        {"mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "Describe the image in English"},
        {"mlx-community/Qwen3-VL-2B-Instruct-4bit",
            "Describe the image in English"},
        {"mlx-community/gemma-3-4b-it-4bit",
            "Describe this image"},
        {"mlx-community/pixtral-12b-2409-4bit",
            "Describe this image"},
        {"mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
            "Describe this image"},
        {"mlx-community/SmolVLM2-2.2B-Instruct-4bit",
            "Describe this image"},
        {"mlx-community/paligemma2-3b-pt-224-4bit",
            "Describe this image"},
    });
    return registry;
}

// --- Load from directory ---

ModelContext load_vlm_from_directory(
    const std::string& model_directory,
    const ModelConfiguration& config)
{
    auto config_path = fs::path(model_directory) / "config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error("config.json not found in " + model_directory);
    }

    std::ifstream config_file(config_path);
    nlohmann::json config_json;
    config_file >> config_json;

    auto base_config = parse_base_configuration(config_json);

    auto& loaders = vlm_loaders();
    auto it = loaders.find(base_config.model_type);
    if (it == loaders.end()) {
        throw std::runtime_error("Unsupported VLM model type: " + base_config.model_type);
    }

    auto weights = load_safetensors_from_directory(model_directory);
    weights = dequantize_weights(std::move(weights), base_config);
    auto ctx = it->second(config_json.dump(), std::move(weights));
    ctx.model_id = config.id.empty() ? model_directory : config.id;

    if (base_config.eos_token_ids.has_value()) {
        ctx.eos_token_ids = base_config.eos_token_ids->values;
    }

    // Load tokenizer from model directory
    auto tokenizer_json_path = fs::path(model_directory) / "tokenizer.json";
    if (fs::exists(tokenizer_json_path)) {
        auto tokenizer = Tokenizer::from_directory(model_directory);
        ctx.encode_fn = [tokenizer](const std::string& text) {
            return tokenizer->encode(text);
        };
        ctx.decode_fn = [tokenizer](const std::vector<int>& ids) {
            return tokenizer->decode(ids);
        };
    }

    return ctx;
}

// --- Load from HF Hub ---

ModelContext load_vlm(
    const std::string& model_id,
    const std::string& cache_dir)
{
    auto& hub = HubApi::shared();
    if (!cache_dir.empty()) {
        hub.set_cache_dir(cache_dir);
    }

    auto model_dir = hub.snapshot_download(model_id);

    ModelConfiguration config;
    config.id = model_id;

    auto& model_registry = vlm_model_registry();
    auto known = model_registry.find(model_id);
    if (known.has_value()) {
        config = known.value();
    }

    return load_vlm_from_directory(model_dir, config);
}

} // namespace mlx_lm
