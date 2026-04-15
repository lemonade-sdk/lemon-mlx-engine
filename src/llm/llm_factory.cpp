// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/llm/models/llama.h>
#include <mlx-lm/llm/models/qwen2.h>
#include <mlx-lm/llm/models/gemma.h>
#include <mlx-lm/llm/models/phi.h>
#include <mlx-lm/llm/models/phi3.h>
#include <mlx-lm/llm/models/qwen3.h>
#include <mlx-lm/llm/models/gemma2.h>
#include <mlx-lm/llm/models/cohere.h>
#include <mlx-lm/llm/models/starcoder2.h>
#include <mlx-lm/llm/models/qwen3_moe.h>
#include <mlx-lm/llm/models/qwen3_next.h>
#include <mlx-lm/llm/models/qwen35_moe.h>
#include <mlx-lm/llm/models/mistral3_text.h>
#include <mlx-lm/llm/models/deepseek_v3.h>
#include <mlx-lm/llm/models/mimo.h>
#include <mlx-lm/llm/models/granite.h>
#include <mlx-lm/llm/models/glm4.h>
#include <mlx-lm/llm/models/ernie4_5.h>
#include <mlx-lm/llm/models/smollm3.h>
#include <mlx-lm/llm/models/minicpm.h>
#include <mlx-lm/llm/models/olmo2.h>
#include <mlx-lm/llm/models/olmo3.h>
#include <mlx-lm/llm/models/nanochat.h>
#include <mlx-lm/llm/models/lille130m.h>
#include <mlx-lm/llm/models/internlm2.h>
#include <mlx-lm/llm/models/exaone4.h>
#include <mlx-lm/llm/models/gemma3_text.h>
#include <mlx-lm/llm/models/gemma3n_text.h>
#include <mlx-lm/llm/models/jamba.h>
#include <mlx-lm/llm/models/apertus.h>
#include <mlx-lm/llm/models/openelm.h>
#include <mlx-lm/llm/models/phimoe.h>
#include <mlx-lm/llm/models/olmoe.h>
#include <mlx-lm/llm/models/glm4_moe.h>
#include <mlx-lm/llm/models/bailing_moe.h>
#include <mlx-lm/llm/models/afmoe.h>
#include <mlx-lm/llm/models/glm4_moe_lite.h>
#include <mlx-lm/llm/models/gptoss.h>
#include <mlx-lm/llm/models/lfm2_moe.h>
#include <mlx-lm/llm/models/baichuan_m1.h>
#include <mlx-lm/llm/models/falcon_h1.h>
#include <mlx-lm/llm/models/lfm2.h>
#include <mlx-lm/llm/models/nemotron_h.h>
#include <mlx-lm/llm/models/granite_moe_hybrid.h>
#include <mlx-lm/common/base_config.h>
#include <mlx-lm/common/safetensors.h>
#include <mlx-lm/common/hub_api.h>
#include <mlx-lm/common/tokenizer.h>
#include <mlx-lm/common/chat_template.h>
#include <mlx-lm/common/quantize_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
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
// The model is stored in a shared_ptr captured by the context's lambdas.
using LLMLoaderFn = std::function<ModelContext(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& base_config)>;

template <typename Config, typename Model>
static ModelContext load_typed_model(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& base_config)
{
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    auto model = std::make_shared<Model>(config);

    weights = model->sanitize(std::move(weights));

    // Register quantized weights in the QuantizedWeightRegistry.
    // This maps model member array addresses → quantization metadata so
    // that linear_fwd() uses mx::quantized_matmul at inference time.
    auto wmap = model->weight_map();
    register_quantized_weights(weights, base_config, wmap);

    model->load_weights(weights);

    return ModelContext::from_model_owned(model);
}

// Internal loader registry — maps model_type to a function that creates,
// sanitizes, loads weights, and returns a fully-initialized ModelContext.
static std::unordered_map<std::string, LLMLoaderFn>& llm_loaders() {
    static std::unordered_map<std::string, LLMLoaderFn> loaders = {
        {"llama",       load_typed_model<LlamaConfiguration, LlamaModel>},
        {"mistral",     load_typed_model<LlamaConfiguration, LlamaModel>},
        {"qwen2",       load_typed_model<Qwen2Configuration, Qwen2Model>},
        {"acereason",   load_typed_model<Qwen2Configuration, Qwen2Model>},
        {"gemma",       load_typed_model<GemmaConfiguration, GemmaModel>},
        {"phi",         load_typed_model<PhiConfiguration, PhiModel>},
        {"phi3",        load_typed_model<Phi3Configuration, Phi3Model>},
        {"phi3small",   load_typed_model<Phi3Configuration, Phi3Model>},
        {"qwen3",       load_typed_model<Qwen3Configuration, Qwen3Model>},
        {"gemma2",      load_typed_model<Gemma2Configuration, Gemma2Model>},
        {"cohere",      load_typed_model<CohereConfiguration, CohereModel>},
        {"command-r",   load_typed_model<CohereConfiguration, CohereModel>},
        {"starcoder2",  load_typed_model<Starcoder2Configuration, Starcoder2Model>},
        {"qwen3_moe",   load_typed_model<Qwen3MoEConfiguration, Qwen3MoEModel>},
        {"qwen3_next",  load_typed_model<Qwen3NextConfiguration, Qwen3NextModel>},
        {"qwen3_5",     load_typed_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
        {"qwen3_5_moe", load_typed_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
        {"mistral3",    load_typed_model<Mistral3TextConfiguration, Mistral3TextModel>},
        {"ministral3",  load_typed_model<Mistral3TextConfiguration, Mistral3TextModel>},
        {"deepseek_v3", load_typed_model<DeepseekV3Configuration, DeepseekV3Model>},
        {"mimo",        load_typed_model<MiMoConfiguration, MiMoModel>},
        {"granite",     load_typed_model<GraniteConfiguration, GraniteModel>},
        {"glm4",        load_typed_model<GLM4Configuration, GLM4Model>},
        {"ernie4_5",    load_typed_model<Ernie45Configuration, Ernie45Model>},
        {"smollm3",     load_typed_model<SmolLM3Configuration, SmolLM3Model>},
        {"minicpm",     load_typed_model<MiniCPMConfiguration, MiniCPMModel>},
        {"olmo2",       load_typed_model<Olmo2Configuration, Olmo2Model>},
        {"olmo3",       load_typed_model<Olmo3Configuration, Olmo3Model>},
        {"nanochat",    load_typed_model<NanoChatConfiguration, NanoChatModel>},
        {"lille-130m",  load_typed_model<Lille130mConfiguration, Lille130mModel>},
        {"internlm2",   load_typed_model<InternLM2Configuration, InternLM2Model>},
        {"exaone4",     load_typed_model<Exaone4Configuration, Exaone4Model>},
        {"gemma3_text", load_typed_model<Gemma3TextConfiguration, Gemma3TextModel>},
        {"apertus",     load_typed_model<ApertusConfiguration, ApertusModel>},
        {"openelm",     load_typed_model<OpenELMConfiguration, OpenELMModel>},
        {"phimoe",      load_typed_model<PhiMoEConfiguration, PhiMoEModel>},
        {"olmoe",       load_typed_model<OlmoEConfiguration, OlmoEModel>},
        {"glm4_moe",    load_typed_model<GLM4MoEConfiguration, GLM4MoEModel>},
        {"bailing_moe", load_typed_model<BailingMoeConfiguration, BailingMoeModel>},
        {"afmoe",       load_typed_model<AfMoEConfiguration, AfMoEModel>},
        {"glm4_moe_lite", load_typed_model<GLM4MoELiteConfiguration, GLM4MoELiteModel>},
        {"gpt_oss",     load_typed_model<GPTOSSConfiguration, GPTOSSModel>},
        {"lfm2_moe",    load_typed_model<LFM2MoEConfiguration, LFM2MoEModel>},
        {"gemma3n_text", load_typed_model<Gemma3nTextConfiguration, Gemma3nTextModel>},
        {"jamba",        load_typed_model<JambaConfiguration, JambaModel>},
        {"baichuan_m1",  load_typed_model<BaichuanM1Configuration, BaichuanM1Model>},
        {"falcon_h1",    load_typed_model<FalconH1Configuration, FalconH1Model>},
        {"lfm2",         load_typed_model<LFM2Configuration, LFM2Model>},
        {"nemotron_h",   load_typed_model<NemotronHConfiguration, NemotronHModel>},
        {"granitemoehybrid", load_typed_model<GraniteMoeHybridConfiguration, GraniteMoeHybridModel>},
        {"bitnet",       load_typed_model<LlamaConfiguration, LlamaModel>},
    };
    return loaders;
}

// --- LLM Type Registry (public, for type checking) ---

ModelTypeRegistry& llm_type_registry() {
    static ModelTypeRegistry registry({
        {"llama",       create_model<LlamaConfiguration, LlamaModel>},
        {"mistral",     create_model<LlamaConfiguration, LlamaModel>},
        {"qwen2",       create_model<Qwen2Configuration, Qwen2Model>},
        {"acereason",   create_model<Qwen2Configuration, Qwen2Model>},
        {"gemma",       create_model<GemmaConfiguration, GemmaModel>},
        {"phi",         create_model<PhiConfiguration, PhiModel>},
        {"phi3",        create_model<Phi3Configuration, Phi3Model>},
        {"phi3small",   create_model<Phi3Configuration, Phi3Model>},
        {"qwen3",       create_model<Qwen3Configuration, Qwen3Model>},
        {"gemma2",      create_model<Gemma2Configuration, Gemma2Model>},
        {"cohere",      create_model<CohereConfiguration, CohereModel>},
        {"command-r",   create_model<CohereConfiguration, CohereModel>},
        {"starcoder2",  create_model<Starcoder2Configuration, Starcoder2Model>},
        {"qwen3_moe",   create_model<Qwen3MoEConfiguration, Qwen3MoEModel>},
        {"qwen3_next",  create_model<Qwen3NextConfiguration, Qwen3NextModel>},
        {"qwen3_5",     create_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
        {"qwen3_5_moe", create_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
        {"mistral3",    create_model<Mistral3TextConfiguration, Mistral3TextModel>},
        {"ministral3",  create_model<Mistral3TextConfiguration, Mistral3TextModel>},
        {"deepseek_v3", create_model<DeepseekV3Configuration, DeepseekV3Model>},
        {"mimo",        create_model<MiMoConfiguration, MiMoModel>},
        {"granite",     create_model<GraniteConfiguration, GraniteModel>},
        {"glm4",        create_model<GLM4Configuration, GLM4Model>},
        {"ernie4_5",    create_model<Ernie45Configuration, Ernie45Model>},
        {"smollm3",     create_model<SmolLM3Configuration, SmolLM3Model>},
        {"minicpm",     create_model<MiniCPMConfiguration, MiniCPMModel>},
        {"olmo2",       create_model<Olmo2Configuration, Olmo2Model>},
        {"olmo3",       create_model<Olmo3Configuration, Olmo3Model>},
        {"nanochat",    create_model<NanoChatConfiguration, NanoChatModel>},
        {"lille-130m",  create_model<Lille130mConfiguration, Lille130mModel>},
        {"internlm2",   create_model<InternLM2Configuration, InternLM2Model>},
        {"exaone4",     create_model<Exaone4Configuration, Exaone4Model>},
        {"gemma3_text", create_model<Gemma3TextConfiguration, Gemma3TextModel>},
        {"apertus",     create_model<ApertusConfiguration, ApertusModel>},
        {"openelm",     create_model<OpenELMConfiguration, OpenELMModel>},
        {"phimoe",      create_model<PhiMoEConfiguration, PhiMoEModel>},
        {"olmoe",       create_model<OlmoEConfiguration, OlmoEModel>},
        {"glm4_moe",    create_model<GLM4MoEConfiguration, GLM4MoEModel>},
        {"bailing_moe", create_model<BailingMoeConfiguration, BailingMoeModel>},
        {"afmoe",       create_model<AfMoEConfiguration, AfMoEModel>},
        {"glm4_moe_lite", create_model<GLM4MoELiteConfiguration, GLM4MoELiteModel>},
        {"gpt_oss",     create_model<GPTOSSConfiguration, GPTOSSModel>},
        {"lfm2_moe",    create_model<LFM2MoEConfiguration, LFM2MoEModel>},
        {"gemma3n_text", create_model<Gemma3nTextConfiguration, Gemma3nTextModel>},
        {"jamba",        create_model<JambaConfiguration, JambaModel>},
        {"baichuan_m1",  create_model<BaichuanM1Configuration, BaichuanM1Model>},
        {"falcon_h1",    create_model<FalconH1Configuration, FalconH1Model>},
        {"lfm2",         create_model<LFM2Configuration, LFM2Model>},
        {"nemotron_h",   create_model<NemotronHConfiguration, NemotronHModel>},
        {"granitemoehybrid", create_model<GraniteMoeHybridConfiguration, GraniteMoeHybridModel>},
        {"bitnet",       create_model<LlamaConfiguration, LlamaModel>},
    });
    return registry;
}

// --- LLM Model Registry ---

AbstractModelRegistry& llm_model_registry() {
    static AbstractModelRegistry registry({
        {"mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            "What is the difference between a fruit and a vegetable?"},
        {"mlx-community/Llama-3.2-1B-Instruct-4bit",
            "What is the difference between a fruit and a vegetable?"},
        {"mlx-community/Llama-3.2-3B-Instruct-4bit",
            "What is the difference between a fruit and a vegetable?"},
        {"mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "Describe the Swift language."},
        {"mlx-community/Qwen3-4B-4bit",
            "Why is the sky blue?"},
        {"mlx-community/Qwen3-8B-4bit",
            "Why is the sky blue?"},
        {"mlx-community/gemma-2-2b-it-4bit",
            "What is machine learning?"},
        {"mlx-community/Phi-3.5-mini-instruct-4bit",
            "What is the theory of relativity?"},
        {"mlx-community/starcoder2-3b-4bit",
            "Write a Python function to sort a list."},
        {"mlx-community/c4ai-command-r-08-2024-4bit",
            "Explain quantum computing."},
    });
    return registry;
}

// --- Load from directory ---

ModelContext load_llm_from_directory(
    const std::string& model_directory,
    const ModelConfiguration& config)
{
    // Read config.json
    auto config_path = fs::path(model_directory) / "config.json";
    if (!fs::exists(config_path)) {
        throw std::runtime_error("config.json not found in " + model_directory);
    }

    std::ifstream config_file(config_path);
    nlohmann::json config_json;
    config_file >> config_json;

    auto base_config = parse_base_configuration(config_json);

    // Find the loader for this model type
    auto& loaders = llm_loaders();
    auto it = loaders.find(base_config.model_type);
    if (it == loaders.end()) {
        throw std::runtime_error("Unsupported model type: " + base_config.model_type);
    }

    // Load weights from safetensors
    auto weights = load_safetensors_from_directory(model_directory);

    // Create model, sanitize weights, register quantized weights, load them.
    // Quantized weights stay packed (uint32) and use quantized_matmul at runtime.
    auto ctx = it->second(config_json.dump(), std::move(weights), base_config);
    ctx.model_id = config.id.empty() ? model_directory : config.id;

    if (base_config.eos_token_ids.has_value()) {
        ctx.eos_token_ids = base_config.eos_token_ids->values;
    }

    // Load tokenizer from model directory
    std::shared_ptr<Tokenizer> tokenizer;
    auto tokenizer_json_path = fs::path(model_directory) / "tokenizer.json";
    if (fs::exists(tokenizer_json_path)) {
        tokenizer = Tokenizer::from_directory(model_directory);
        ctx.encode_fn = [tokenizer](const std::string& text) {
            return tokenizer->encode(text);
        };
        ctx.decode_fn = [tokenizer](const std::vector<int>& ids) {
            return tokenizer->decode(ids);
        };
    }

    // Load chat template from tokenizer_config.json
    auto chat_tmpl = load_chat_template(model_directory);
    if (chat_tmpl.has_value() && tokenizer) {
        auto shared_tmpl = std::make_shared<ChatTemplate>(std::move(*chat_tmpl));

        // If eos_token_ids not set from config.json, resolve from tokenizer_config eos_token
        if (!ctx.eos_token_ids.has_value() && !shared_tmpl->eos_token().empty()) {
            int eos_id = tokenizer->token_to_id(shared_tmpl->eos_token());
            if (eos_id >= 0) {
                ctx.eos_token_ids = std::vector<int>{eos_id};
            }
        }

        // Shared extra context allows callers (e.g., chat.cpp --no-think)
        // to inject template variables after model loading.
        auto extra_ctx = std::make_shared<nlohmann::json>();
        ctx.template_extra_context = extra_ctx;

        ctx.apply_chat_template_fn = [shared_tmpl, tokenizer, extra_ctx](
            const std::vector<Message>& messages) -> std::vector<int> {
            auto rendered = shared_tmpl->apply(messages, /*add_generation_prompt=*/true, *extra_ctx);
            return tokenizer->encode(rendered);
        };
    }

    return ctx;
}

// --- Load from HF Hub ---

ModelContext load_llm(
    const std::string& model_id,
    const std::string& cache_dir)
{
    // If model_id is a local directory with config.json, use it directly
    if (fs::exists(fs::path(model_id) / "config.json")) {
        ModelConfiguration config;
        config.id = model_id;
        return load_llm_from_directory(model_id, config);
    }

    auto& hub = HubApi::shared();
    if (!cache_dir.empty()) {
        hub.set_cache_dir(cache_dir);
    }

    // Download model
    auto model_dir = hub.snapshot_download(model_id);

    ModelConfiguration config;
    config.id = model_id;

    // Check registry for known configuration
    auto& model_registry = llm_model_registry();
    auto known = model_registry.find(model_id);
    if (known.has_value()) {
        config = known.value();
    }

    return load_llm_from_directory(model_dir, config);
}

} // namespace mlx_lm
