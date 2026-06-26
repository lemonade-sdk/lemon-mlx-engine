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
#include <mlx-lm/llm/models/bitnet.h>
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
#include <mlx-lm/common/gguf_loader.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mx = mlx::core;

namespace mlx_lm {

// Helper: create a typed model from JSON config data (for ModelTypeRegistry).
template <typename Config, typename Model>
static void* create_model(const std::string& config_json) {
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    return new Model(config);
}

// BitNet type dispatch: BitNetModel supports both true relu² BitNet and
// Falcon-E-style silu BitLinear checkpoints (without sub-norms).
static void* create_bitnet_model(const std::string& config_json) {
    auto j = nlohmann::json::parse(config_json);
    if (!j.contains("hidden_act")) j["hidden_act"] = "relu2";
    BitNetConfiguration config = j.get<BitNetConfiguration>();
    return new BitNetModel(config);
}

// Helper: create, sanitize, load weights, and return an owned ModelContext.
// The model is stored in a shared_ptr captured by the context's lambdas.
using LLMLoaderFn = std::function<ModelContext(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& base_config,
    bool auto_quantize)>;

// Force every weight resident in device memory NOW. MLX loads weights lazily
// (mmap-backed, materialized to VRAM on first use during a forward pass). That
// is fine for a unified-memory APU, but on a discrete GPU — especially over a
// non-coherent link (TB5 eGPU) — interleaving per-weight H2D copies with compute
// stalls the first forward. Eagerly evaluating all weights separates load from
// compute and guarantees the whole model is in VRAM before inference.
static void materialize_weights(
    std::unordered_map<std::string, mlx::core::array>& weights)
{
    std::vector<mlx::core::array> all;
    all.reserve(weights.size());
    for (auto& kv : weights) all.push_back(kv.second);
    if (!all.empty()) {
        mlx::core::eval(all);
        mlx::core::synchronize();
    }
}

template <typename Config, typename Model>
static ModelContext load_typed_model(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& base_config,
    bool auto_quantize)
{
    auto j = nlohmann::json::parse(config_json);
    Config config = j.get<Config>();
    auto model = std::make_shared<Model>(config);

    weights = model->sanitize(std::move(weights));

    auto wmap = model->weight_map();

    // Auto-quantize unquantized bf16/fp16 weights to 4-bit on-the-fly.
    // Runs before register_quantized_weights so the model loads from
    // already-quantized weight entries and registry metadata.
    if (auto_quantize) {
        auto_quantize_weights(weights, wmap, base_config);
    }

    // Register quantized weights in the QuantizedWeightRegistry.
    // This maps model member array addresses → quantization metadata so
    // that linear_fwd() uses mx::quantized_matmul at inference time.
    register_quantized_weights(weights, base_config, wmap);

    // Warn about missing weight keys before loading (catches HF naming mismatches)
    {
        int missing = 0;
        std::string first_missing;
        for (auto& [name, target] : wmap) {
            if (weights.find(name) == weights.end()) {
                if (missing == 0) first_missing = name;
                missing++;
            }
        }
        if (missing > 0) {
            std::cerr << "[load] WARNING: " << missing << " weight(s) not found in checkpoint"
                      << " (first: " << first_missing << ")."
                      << " Weights will be left unset (may cause inference errors)."
                      << " This usually means the checkpoint uses a different key naming convention."
                      << std::endl;
        }
    }

    materialize_weights(weights);
    model->load_weights(weights);

    return ModelContext::from_model_owned(model);
}

// BitNet dispatch: models with model_type="bitnet" can be either true BitNet
// b1.58 (hidden_act="relu2", has sub_norms) or Falcon-E-style BitLinear
// checkpoints (hidden_act="silu", no sub_norms). BitNetModel handles both and
// preserves runtime 2-bit weights instead of dequantizing to fp16.
static ModelContext load_bitnet_model(
    const std::string& config_json,
    std::unordered_map<std::string, mlx::core::array> weights,
    const BaseConfiguration& base_config,
    bool auto_quantize)
{
    auto j = nlohmann::json::parse(config_json);
    if (!j.contains("hidden_act")) {
        j["hidden_act"] = "relu2";
    }
    return load_typed_model<BitNetConfiguration, BitNetModel>(
        j.dump(), std::move(weights), base_config, auto_quantize);
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
        {"qwen3_5_mtp", load_typed_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
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
        {"bitnet",         load_bitnet_model},
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
        {"qwen3_5_mtp", create_model<Qwen35MoEConfiguration, Qwen35MoEModel>},
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
        {"bitnet",         create_bitnet_model},
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
    nlohmann::json config_json;

    // Check for GGUF file first (single-file format, no config.json)
    // If config.json exists, use the standard safetensors path.
    auto config_path = fs::path(model_directory) / "config.json";
    if (!fs::exists(config_path)) {
        // No config.json. Check if the directory contains a .gguf file.
        std::string gguf_file;
        for (const auto& e : fs::directory_iterator(model_directory)) {
            if (e.path().extension() == ".gguf") {
                gguf_file = e.path().string();
                break;
            }
        }
        if (!gguf_file.empty()) {
            // Synthesize config from GGUF metadata
            auto meta = mlx::core::load_gguf(gguf_file).second;
            config_json = gguf_config_from_metadata(meta);

            auto base_config = parse_base_configuration(config_json);
            auto& loaders = llm_loaders();
            auto it = loaders.find(base_config.model_type);
            if (it == loaders.end()) {
                throw std::runtime_error("Unsupported GGUF architecture: '" +
                    base_config.model_type + "'");
            }

            auto weights = load_gguf_weights(gguf_file);
            // Materialize and load the model
            auto ctx = it->second(config_json.dump(), std::move(weights),
                                 base_config, config.auto_quantize);
            ctx.model_id = config.id.empty() ? model_directory : config.id;
            return ctx;
        }
        throw std::runtime_error("config.json not found in " + model_directory);
    }

    std::ifstream config_file(config_path);
    config_file >> config_json;

    // Detect MTP delta models (model_type="qwen3_5_mtp") and redirect
    // to the delta loading path which merges with the base model.
    // MTP delta models contain only the MTP head weights (single decoder
    // layer + fc/norm) and cannot be loaded standalone.
    std::string model_type = config_json.value("model_type", "");
    if (model_type == "qwen3_5_mtp") {
        std::string model_id = config.id.empty() ? model_directory : config.id;
        std::cerr << "[MTP] Delta model detected via load_llm, redirecting to load_mtp_delta_model\n";
        auto ctx = load_mtp_delta_model(model_id);
        ctx.model_id = model_id;

        if (!ctx.eos_token_ids.has_value()) {
            if (config_json.contains("text_config") && config_json["text_config"].contains("eos_token_id")) {
                ctx.eos_token_ids = {config_json["text_config"]["eos_token_id"].get<int>()};
            }
        }

        std::shared_ptr<Tokenizer> tokenizer;
        try {
            tokenizer = Tokenizer::from_directory(model_directory);
            ctx.encode_fn = [tokenizer](const std::string& text) {
                return tokenizer->encode(text);
            };
            ctx.decode_fn = [tokenizer](const std::vector<int>& ids) {
                return tokenizer->decode(ids);
            };
        } catch (const std::exception& e) {
            std::cerr << "[load] tokenizer load failed: " << e.what() << std::endl;
        }

        // Load chat template
        auto chat_tmpl = load_chat_template(model_directory);
        if (chat_tmpl.has_value() && tokenizer) {
            auto shared_tmpl = std::make_shared<ChatTemplate>(std::move(*chat_tmpl));
            if (ctx.eos_token_ids.has_value() && !shared_tmpl->eos_token().empty()) {
                int eos_id = tokenizer->token_to_id(shared_tmpl->eos_token());
                if (eos_id >= 0) {
                    ctx.eos_token_ids = std::vector<int>{eos_id};
                }
            }
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

    auto base_config = parse_base_configuration(config_json);

    // Find the loader for this model type
    auto& loaders = llm_loaders();
    auto it = loaders.find(base_config.model_type);
    if (it == loaders.end()) {
        // Try common HF architecture aliases before giving up
        static const std::unordered_map<std::string, std::string> aliases = {
            {"llama3", "llama"},
            {"qwen3_moe_base", "qwen3_moe"},
            {"gemma3", "gemma3_text"},
        };
        if (auto ait = aliases.find(base_config.model_type); ait != aliases.end()) {
            it = loaders.find(ait->second);
        }
    }
    if (it == loaders.end()) {
        std::string supported;
        for (auto& [k, _] : loaders) supported += "  - " + k + "\n";
        throw std::runtime_error(
            "Unsupported model type: '" + base_config.model_type + "'.\n"
            "Supported types:\n" + supported +
            "\nIf this is a standard Llama-family model, try converting it to MLX format first:\n"
            "  pip install mlx-lm && mlx_lm.convert --hf-model <hf-repo-id> --out-dir <output-dir>");
    }

    // Load weights from safetensors
    auto weights = load_safetensors_from_directory(model_directory);

    // Create model, sanitize weights, register quantized weights, load them.
    // Quantized weights stay packed (uint32) and use quantized_matmul at runtime.
    auto ctx = it->second(config_json.dump(), std::move(weights), base_config, config.auto_quantize);
    ctx.model_id = config.id.empty() ? model_directory : config.id;

    if (base_config.eos_token_ids.has_value()) {
        ctx.eos_token_ids = base_config.eos_token_ids->values;
    }

    std::shared_ptr<Tokenizer> tokenizer;
    try {
        tokenizer = Tokenizer::from_directory(model_directory);
        ctx.encode_fn = [tokenizer](const std::string& text) {
            return tokenizer->encode(text);
        };
        ctx.decode_fn = [tokenizer](const std::vector<int>& ids) {
            return tokenizer->decode(ids);
        };
    } catch (const std::exception& e) {
        std::cerr << "[load] tokenizer load failed: " << e.what() << std::endl;
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

// --- MTP Delta Model Loading ---
// MTP (Multi-Token Prediction) delta models contain only the MTP head weights
// (single decoder layer + fc/norm). The base text model weights must be loaded
// separately and merged. This function handles that flow automatically.

static std::string derive_base_model_id(const std::string& delta_model_id) {
    // Strip "-MTP" from the repo name:
    //   mlx-community/Qwen3.5-4B-MTP-4bit -> mlx-community/Qwen3.5-4B-4bit
    auto result = delta_model_id;
    auto pos = result.find("-MTP");
    while (pos != std::string::npos) {
        result.erase(pos, 4);
        pos = result.find("-MTP", pos);
    }
    return result;
}

// Recover the HuggingFace repo id ("org/repo") from a local HF cache path of
// the form  .../models--<org>--<repo>/snapshots/<hash>/ . Returns "" if the
// path is not an HF cache layout (e.g. an arbitrary local directory).
static std::string repo_id_from_cache_path(const std::string& path_str) {
    for (const auto& part : fs::path(path_str)) {
        const std::string seg = part.string();
        if (seg.rfind("models--", 0) == 0) {
            const std::string body = seg.substr(8);  // after "models--"
            const auto sep = body.find("--");
            if (sep == std::string::npos) return "";
            return body.substr(0, sep) + "/" + body.substr(sep + 2);
        }
    }
    return "";
}

ModelContext load_mtp_delta_model(
    const std::string& delta_model_id,
    const std::string& cache_dir)
{
    auto& hub = HubApi::shared();
    if (!cache_dir.empty()) {
        hub.set_cache_dir(cache_dir);
    }

    // Step 1: Download / resolve delta model (MTP head weights).
    std::string delta_dir;
    if (fs::exists(fs::path(delta_model_id) / "config.json")) {
        delta_dir = delta_model_id;
    } else if (hub.is_cached(delta_model_id)) {
        delta_dir = hub.model_directory(delta_model_id);
    } else {
        delta_dir = hub.snapshot_download(delta_model_id);
    }

    // Step 2: Derive and resolve base model (full text backbone).
    // If the delta was given as a local HF cache path, the base repo lives in a
    // SIBLING cache dir under its own snapshot hash — not the delta's hash — so
    // recover the repo id and resolve the base through the hub. Only fall back to
    // naive "-MTP" string-stripping on the raw id when it is a plain repo id.
    std::string base_model_id;
    std::string base_dir;
    const std::string delta_repo_id = repo_id_from_cache_path(delta_model_id);
    if (!delta_repo_id.empty()) {
        base_model_id = derive_base_model_id(delta_repo_id);
        std::cerr << "[MTP] Delta model: " << delta_model_id
                  << ", base model: " << base_model_id << "\n";
        if (hub.is_cached(base_model_id)) {
            base_dir = hub.model_directory(base_model_id);
        } else {
            base_dir = hub.snapshot_download(base_model_id);
        }
    } else {
        base_model_id = derive_base_model_id(delta_model_id);
        std::cerr << "[MTP] Delta model: " << delta_model_id
                  << ", base model: " << base_model_id << "\n";
        if (fs::exists(fs::path(base_model_id) / "config.json")) {
            base_dir = base_model_id;
        } else if (hub.is_cached(base_model_id)) {
            base_dir = hub.model_directory(base_model_id);
        } else {
            base_dir = hub.snapshot_download(base_model_id);
        }
    }

    // Step 3: Load base model safetensors (full 32-layer text backbone).
    auto weights = load_safetensors_from_directory(base_dir);
    std::cerr << "[MTP] Loaded base model weights: " << weights.size() << " tensors\n";

    // Step 4: Load delta model safetensors (MTP head), prefix keys with "mtp.".
    auto delta_weights = load_safetensors_from_directory(delta_dir);
    int mtp_keys = 0;
    for (auto& [key, value] : delta_weights) {
        std::string prefixed = "mtp." + key;
        weights.insert_or_assign(prefixed, std::move(value));
        mtp_keys++;
    }
    std::cerr << "[MTP] Merged " << mtp_keys << " MTP head weights with base model\n";

    // Step 5: Read BASE model config.json for architectural parameters.
    // The base model config matches the actual text backbone weights we loaded.
    // The delta model's config has model_type="qwen3_5_mtp" which is for the
    // MTP head only, not the full text model architecture.
    auto base_config_path = fs::path(base_dir) / "config.json";
    std::ifstream base_config_file(base_config_path);
    nlohmann::json base_model_config_json;
    base_config_file >> base_model_config_json;

    // Step 5b: Read MTP delta model config.json for MTP head parameters.
    // The MTP model's text_config contains some correct architectural parameters
    // (hidden_size=2560, intermediate_size=9216, rope_theta=10000000) but the
    // head_dim and num_attention_heads fields are inconsistent with actual weights.
    // We extract only the reliable fields (hidden_size, intermediate_size, rope)
    // and leave head_dim/num_attention_heads/num_key_value_heads as 0 so that
    // build_mtp_head() derives them from actual weight shapes.
    auto mtp_config_path = fs::path(delta_dir) / "config.json";
    std::optional<MTPHeadConfig> mtp_head_cfg;
    if (fs::exists(mtp_config_path)) {
        std::ifstream mtp_config_file(mtp_config_path);
        nlohmann::json mtp_config_json;
        mtp_config_file >> mtp_config_json;
        if (mtp_config_json.contains("text_config")) {
            auto& tc = mtp_config_json["text_config"];
            mtp_head_cfg = MTPHeadConfig{};
            mtp_head_cfg->hidden_size = tc.value("hidden_size", 0);
            mtp_head_cfg->intermediate_size = tc.value("intermediate_size", 0);
            // DO NOT set head_dim, num_attention_heads, num_key_value_heads —
            // these are wrong in the MTP config.json. build_mtp_head() will
            // derive them from actual weight shapes (o_proj, q_proj, k_proj).
            mtp_head_cfg->head_dim = 0;
            mtp_head_cfg->num_attention_heads = 0;
            mtp_head_cfg->num_key_value_heads = 0;
            mtp_head_cfg->rms_norm_eps = tc.value("rms_norm_eps", 1e-6f);
            mtp_head_cfg->quant_bits = tc.value("quant_bits", 4);
            mtp_head_cfg->quant_group_size = tc.value("quant_group_size", 64);
            // rope_parameters is a nested object
            if (tc.contains("rope_parameters")) {
                auto& rp = tc["rope_parameters"];
                mtp_head_cfg->rope_theta = rp.value("rope_theta", 10000.0f);
                mtp_head_cfg->partial_rotary_factor = rp.value("partial_rotary_factor", 0.25f);
            }
        }
    }

    auto base_config = parse_base_configuration(base_model_config_json);

    // Step 6: Create model from base config (matches loaded weights), sanitize, register quantized weights, load.
    auto j = nlohmann::json::parse(base_model_config_json.dump());
    Qwen35MoEConfiguration config = j.get<Qwen35MoEConfiguration>();
    auto model = std::make_shared<Qwen35MoEModel>(config);

    // Pass MTP head config before loading weights so build_mtp_head() uses
    // config.json for hidden_size, intermediate_size, rope_theta, while
    // deriving head_dim/num_attention_heads/num_key_value_heads from weights.
    if (mtp_head_cfg.has_value()) {
        model->set_mtp_head_config(mtp_head_cfg.value());
    }

    weights = model->sanitize(std::move(weights));

    auto wmap = model->weight_map();
    register_quantized_weights(weights, base_config, wmap);

    materialize_weights(weights);
    model->load_weights(weights);

    ModelContext ctx = ModelContext::from_model_owned(model);
    ctx.model_id = delta_model_id;

    if (base_config.eos_token_ids.has_value()) {
        ctx.eos_token_ids = base_config.eos_token_ids->values;
    }

    std::shared_ptr<Tokenizer> tokenizer;
    try {
        tokenizer = Tokenizer::from_directory(delta_dir);
        ctx.encode_fn = [tokenizer](const std::string& text) {
            return tokenizer->encode(text);
        };
        ctx.decode_fn = [tokenizer](const std::vector<int>& ids) {
            return tokenizer->decode(ids);
        };
    } catch (const std::exception& e) {
        std::cerr << "[load] tokenizer load failed: " << e.what() << std::endl;
    }

    // Load chat template from delta model directory.
    auto chat_tmpl = load_chat_template(delta_dir);
    if (chat_tmpl.has_value() && tokenizer) {
        auto shared_tmpl = std::make_shared<ChatTemplate>(std::move(*chat_tmpl));
        if (!ctx.eos_token_ids.has_value() && !shared_tmpl->eos_token().empty()) {
            int eos_id = tokenizer->token_to_id(shared_tmpl->eos_token());
            if (eos_id >= 0) {
                ctx.eos_token_ids = std::vector<int>{eos_id};
            }
        }
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
    // If model_id is a local .gguf file, handle it directly
    if (fs::exists(fs::path(model_id)) &&
        fs::path(model_id).extension() == ".gguf") {
        // Wrap in a temporary directory and delegate
        auto parent = fs::path(model_id).parent_path();
        if (parent.empty()) parent = ".";
        ModelConfiguration config;
        config.id = model_id;
        return load_llm_from_directory(parent, config);
    }

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


// --- Load from directory (with auto_quantize flag) ---

ModelContext load_llm_from_directory(
    const std::string& model_directory,
    bool auto_quantize)
{
    ModelConfiguration config;
    config.id = model_directory;
    config.auto_quantize = auto_quantize;
    return load_llm_from_directory(model_directory, config);
}


ModelContext load_llm(
    const std::string& model_id,
    const std::string& cache_dir,
    bool auto_quantize)
{
    // If model_id is a local .gguf file, handle it directly
    if (fs::exists(fs::path(model_id)) &&
        fs::path(model_id).extension() == ".gguf") {
        auto parent = fs::path(model_id).parent_path();
        if (parent.empty()) parent = ".";
        ModelConfiguration config;
        config.id = model_id;
        config.auto_quantize = auto_quantize;
        return load_llm_from_directory(parent, config);
    }

    // If model_id is a local directory with config.json, use it directly
    if (fs::exists(fs::path(model_id) / "config.json")) {
        ModelConfiguration config;
        config.id = model_id;
        config.auto_quantize = auto_quantize;
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
    config.auto_quantize = auto_quantize;

    // Check registry for known configuration
    auto& model_registry = llm_model_registry();
    auto known = model_registry.find(model_id);
    if (known.has_value()) {
        config = known.value();
        config.auto_quantize = auto_quantize;  // CLI flag overrides registry default
    }

    return load_llm_from_directory(model_dir, config);
}

} // namespace mlx_lm
