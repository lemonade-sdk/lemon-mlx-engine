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
#include <mlx-lm/llm/models/gemma4.h>
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
#include <vector>

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

    // For 1-bit models (1bitLLM style), pre-quantize F32 weights to ternary
    // before loading. Call the helper for this.
    int model_input_bits = j.value("input_bits", 0);
    if (model_input_bits > 0 && !auto_quantize) {
        std::cerr << "[load] Pre-quantizing F32 weights to 1-bit ternary (input_bits="
                  << model_input_bits << ")\n";
        quantize_weights_to_ternary(weights);
    }

    // Register quantized weights in the QuantizedWeightRegistry.
    // This maps model member array addresses → quantization metadata so
    // that linear_fwd() uses mx::quantized_matmul at inference time.
    register_quantized_weights(weights, base_config, wmap);

    // Remap missing weight keys by trying common HF naming alternatives.
    // This allows loading checkpoints that use different naming conventions
    // (e.g., 'model.model.layers...' vs 'model.layers...', 'transformer.' prefix, etc.)
    {
        // First, remap 1-bit specific key names in the weights themselves
        // (ffn_layernorm -> ffn_sub_norm, inner_attn_ln -> attn_sub_norm)
        std::vector<std::pair<std::string, std::string>> bitnet_remaps = {
            {"ffn_layernorm", "ffn_sub_norm"},
            {"inner_attn_ln", "attn_sub_norm"},
        };
        for (auto& [old_suffix, new_suffix] : bitnet_remaps) {
            std::vector<std::string> keys_to_rename;
            for (auto& [key, _] : weights) {
                if (key.find(old_suffix) != std::string::npos) {
                    keys_to_rename.push_back(key);
                }
            }
            for (const auto& key : keys_to_rename) {
                std::string new_key = key;
                size_t p = new_key.find(old_suffix);
                new_key.replace(p, old_suffix.size(), new_suffix);
                weights.emplace(new_key, std::move(weights.at(key)));
                weights.erase(key);
            }
        }

        int missing = 0;
        std::string first_missing;
        for (auto& [name, target] : wmap) {
            if (weights.find(name) == weights.end()) {
                // Try alternative common HF naming conventions
                bool found_alt = false;
                std::vector<std::pair<std::string, std::string>> alt_remaps = {
                    {"model.", "model.model."},
                    {"model.", "model.model.model."},
                    {"model.", "transformer."},
                    {"model.", "gpt_neox."},
                    {"model.", "llama."},
                    {"model.", ""},
                    {"language_model.model.", "model."},  // Gemma 4
                };
                for (auto& [old_pref, new_pref] : alt_remaps) {
                    if (name.find(new_pref) == 0) {
                        std::string alt_key = old_pref + name.substr(new_pref.size());
                        auto ait = weights.find(alt_key);
                        if (ait != weights.end()) {
                            weights.insert_or_assign(name, ait->second);
                            weights.erase(ait);
                            found_alt = true;
                            break;
                        }
                    }
                }
                // Try 1-bit model specific sub-norm key remapping
                if (!found_alt && !first_missing.empty()) {
                    // ffn_layernorm -> ffn_sub_norm (BitNetModel naming)
                    if (name.find("ffn_layernorm") != std::string::npos) {
                        std::string alt_key = name;
                        size_t p = alt_key.find("ffn_layernorm");
                        alt_key.replace(p, 13, "ffn_sub_norm");
                        if (weights.find(alt_key) != weights.end()) {
                            weights.insert_or_assign(name, weights.at(alt_key));
                            weights.erase(alt_key);
                            found_alt = true;
                        }
                    }
                    // inner_attn_ln -> attn_sub_norm
                    if (name.find("inner_attn_ln") != std::string::npos) {
                        std::string alt_key = name;
                        size_t p = alt_key.find("inner_attn_ln");
                        alt_key.replace(p, 14, "attn_sub_norm");
                        if (weights.find(alt_key) != weights.end()) {
                            weights.insert_or_assign(name, weights.at(alt_key));
                            weights.erase(alt_key);
                            found_alt = true;
                        }
                    }
                }
                if (!found_alt) {
                    if (missing == 0) first_missing = name;
                    missing++;
                }
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
        {"gemma4_text",   load_typed_model<Gemma4Configuration, Gemma4Model>},
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
        {"gemma4_text",   create_model<Gemma4Configuration, Gemma4Model>},
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
        // ── Established 4-bit models ──
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
        // ── Newly added 4-bit models ──
        {"mlx-community/gemma-3-4b-it-qat-4bit",
            "What is the Gemma architecture?"},
        {"mlx-community/gemma-4-e2b-it-4bit",
            "Describe the Gemma 4 model architecture."},

        // ── 1-bit / BitNet models ──
        {"mlx-community/Falcon-E-3B-Instruct-1.58bit",
            "What is the capital of France?"},
        {"microsoft/bitnet-b1.58-2B-4T",
            "Why is the sky blue?"},
        {"1bitLLM/bitnet_b1_58-3B",
            "Explain quantum computing."},
        {"tiiuae/Falcon3-7B-Instruct-1.58bit",
            "What is the capital of France?"},
        // ── Bonsai 1-bit MLX models ──
        {"prism-ml/Bonsai-1.7B-mlx-1bit",
            "What is the capital of France?"},
        {"prism-ml/Bonsai-4B-mlx-1bit",
            "What is the capital of France?"},
        {"prism-ml/Bonsai-8B-mlx-1bit",
            "What is the capital of France?"},
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
            auto gguf_meta = gguf_read_metadata(gguf_file);
            config_json = gguf_config_from_metadata(gguf_meta);

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

    // Strip multi-modal prefix from weight keys (Gemma 4 uses
    // "language_model.model.layers..." instead of "model.layers...")
    {
        std::vector<std::pair<std::string, std::string>> prefix_strips = {
            {"language_model.model.", "model."},
        };
        // We apply this AFTER loading weights, in the weight loading path.
        // For now, mark it for the remapping system.
    }

    // Merge text_config fields into top-level for multi-modal models
    // (Gemma 4, Qwen2.5-VL, etc.) that nest LM params under text_config.
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        auto& tc = config_json["text_config"];
        for (auto it = tc.begin(); it != tc.end(); ++it) {
            if (!config_json.contains(it.key())) {
                config_json[it.key()] = it.value();
            }
        }
        // Override model_type with the text-specific type if available
        if (tc.contains("model_type")) {
            config_json["model_type"] = tc["model_type"];
        }
    }

    // Detect MTP delta models (model_type="qwen3_5_mtp") and redirect
    // to the delta loading path which merges with the base model.
    // MTP delta models contain only the MTP head weights (single decoder
    // layer + fc/norm) and cannot be loaded standalone.
    std::string model_type = config_json.value("model_type", "");
    if (model_type == "qwen3_5_mtp") {
        std::string model_id = config.id.empty() ? model_directory : config.id;
        std::cerr << "[MTP] Delta model detected via load_llm, redirecting to load_mtp_delta_model\n";
        auto ctx = load_mtp_delta_model(model_id, "", config.auto_quantize);
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

    // Check for 1-bit / weight-bits models that need BitNet architecture
    // (they have sub-norms like ffn_layernorm, inner_attn_ln which LlamaModel lacks)
    {
        std::string quant_method;
        auto check_quant = [&](const std::string& key) {
            if (config_json.contains(key) && config_json[key].is_object()) {
                auto& obj = config_json[key];
                if (obj.contains("quant_method"))
                    quant_method = obj["quant_method"].get<std::string>();
            }
        };
        check_quant("quantization");
        check_quant("quantization_config");

        // Also check nested quantization.bits for 1-bit MLX format (Bonsai style)
        int quant_bits = 0;
        {
            // Check config_json["quantization"]["bits"]
            if (config_json.contains("quantization") && config_json["quantization"].is_object())
                quant_bits = config_json["quantization"].value("bits", 0);
        }

        bool is_bitnet = (config_json.value("weight_bits", 0) == 1 ||
                          config_json.value("input_bits", 0) == 8 ||
                          quant_method == "bitnet" ||
                          quant_bits == 1);

        if (is_bitnet) {
            std::string orig_type = config_json.value("model_type", "");
            // Qwen3+BitNet: has per-projection RMS norms (from HuggingFace BitNetForCausalLM)
            if ((orig_type == "qwen3" || orig_type == "qwen2") &&
                (quant_method == "bitnet" || config_json.value("weight_bits", 0) == 1)) {
                std::cerr << "[load] Detected Qwen3+BitNet model, enabling per-projection norms\n";
                config_json["bitnet_has_sub_norm"] = true;
                config_json["has_pre_norms"] = true;
            }
            // Bonsai-style: 1-bit MLX affine quantization via quantization.bits=1
            // These use standard Qwen3 architecture with MLX's quantized_matmul
            else if (orig_type == "qwen3" && quant_bits == 1) {
                std::cerr << "[load] Detected Qwen3+1bit model (Bonsai), using standard Qwen3\n";
                // No per-projection norms needed — standard MLX 1-bit format
            }
            // Other 1-bit models route through BitNetModel
            else {
                std::cerr << "[load] Detected 1-bit weight model, routing through BitNetModel\n";
                config_json["model_type"] = "bitnet";
            }
            if (!config_json.contains("hidden_act")) {
                config_json["hidden_act"] = config_json.value("hidden_act", "silu");
            }
        }
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
        // Check the runtime architecture registry (loaded from --register-arch)
        auto* arch_reg = ArchitectureRegistry::instance().find(base_config.model_type);
        if (arch_reg) {
            std::cerr << "[load] Found registered architecture '" << base_config.model_type
                      << "' -> base '" << arch_reg->base_model << "'\n";

            // Apply config defaults from the registration
            for (const auto& [key, val] : arch_reg->config_defaults) {
                if (!config_json.contains(key)) {
                    config_json[key] = val;
                }
            }

            // Inject has_sub_norm into config for BitNetModel to use
            if (arch_reg->has_sub_norm) {
                config_json["bitnet_has_sub_norm"] = true;
            }
            if (arch_reg->activation_bits > 0) {
                config_json["activation_bits"] = arch_reg->activation_bits;
            }

            // Apply key remaps to weights BEFORE loading
            // (ffn_layernorm -> ffn_sub_norm etc)
            std::vector<std::pair<std::string, std::string>> remaps_to_add;
            for (const auto& [old_s, new_s] : arch_reg->key_remaps) {
                if (old_s != new_s) {
                    remaps_to_add.push_back({old_s, new_s});
                }
            }
            if (!remaps_to_add.empty()) {
                // Add remaps to the weights map before sanitize
                // We need to wait until weights are loaded to apply these
                // Store them for now, they'll be picked up by the generic remapping code
                std::cerr << "[load]  " << remaps_to_add.size() << " key remaps registered\n";
            }

            it = loaders.find(arch_reg->base_model);
        }
    }

    if (it == loaders.end()) {
        // Unknown model_type. Try fallback: if config has Llama-like dimensions,
        // create a LlamaModel as a best-effort fallback.
        bool can_fallback = false;
        if (config_json.contains("hidden_size") &&
            config_json.contains("num_hidden_layers") &&
            config_json.contains("num_attention_heads")) {
            can_fallback = true;
            // Detect if it's a Qwen/Gemma-style model by checking for specific config keys
            if (config_json.contains("num_key_value_heads")) {
                can_fallback = true;
            }
        }

        if (can_fallback) {
            // Check for Gemma-like config (uses hidden_activation, not hidden_act)
            if (config_json.contains("hidden_activation") &&
                !config_json.contains("hidden_act")) {
                config_json["hidden_act"] = config_json["hidden_activation"];
            }
            // Default to silu if no activation specified
            if (!config_json.contains("hidden_act")) {
                config_json["hidden_act"] = "silu";
            }
            // Ensure rms_norm_eps
            if (!config_json.contains("rms_norm_eps")) {
                config_json["rms_norm_eps"] = 1e-6;
            }
            // Default to tied embeddings
            if (!config_json.contains("tie_word_embeddings")) {
                config_json["tie_word_embeddings"] = true;
            }
            // Default to 2048 max context
            if (!config_json.contains("max_position_embeddings")) {
                config_json["max_position_embeddings"] = 2048;
            }

            std::cerr << "[load] Unknown model_type '" << base_config.model_type
                      << "' but config has Llama-compatible dimensions."
                      << " Attempting fallback LlamaModel."
                      << " (hidden_size=" << config_json["hidden_size"]
                      << ", layers=" << config_json["num_hidden_layers"]
                      << ", heads=" << config_json["num_attention_heads"]
                      << ")\n";
            it = loaders.find("llama");
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

    // Strip multi-modal prefixes from weight keys (Gemma 4 uses
    // "language_model.model.xxx" instead of "model.xxx")
    {
        static const std::vector<std::pair<std::string, std::string>> prefix_strips = {
            {"language_model.model.", "model."},
        };
        for (auto& [old_p, new_p] : prefix_strips) {
            std::vector<std::string> keys_to_rename;
            for (auto& [key, _] : weights) {
                if (key.compare(0, old_p.size(), old_p) == 0)
                    keys_to_rename.push_back(key);
            }
            for (auto& old_key : keys_to_rename) {
                std::string new_key = new_p + old_key.substr(old_p.size());
                auto nh = weights.extract(old_key);
                nh.key() = new_key;
                weights.insert(std::move(nh));
            }
        }
    }

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
    const std::string& cache_dir,
    bool auto_quantize)
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

    if (auto_quantize) {
        auto_quantize_weights(weights, wmap, base_config);
    }

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
        auto parent = fs::path(model_id).parent_path();
        if (parent.empty()) parent = ".";
        ModelConfiguration config;
        config.id = model_id;
        return load_llm_from_directory(parent, config);
    }

    // If model_id is a local path, validate and load
    if (fs::exists(fs::path(model_id))) {
        if (fs::is_directory(fs::path(model_id))) {
            if (!fs::exists(fs::path(model_id) / "config.json")) {
                throw std::runtime_error(
                    "Model directory found but missing config.json: " + model_id +
                    ". A valid model directory must contain config.json and model.safetensors files.");
            }
        } else {
            throw std::runtime_error(
                "Model path is a file, not a directory: " + model_id +
                ". Expected a directory with config.json and .safetensors, or a .gguf file.");
        }
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

    // If model_id is a local path, validate and load
    if (fs::exists(fs::path(model_id))) {
        if (fs::is_directory(fs::path(model_id))) {
            if (!fs::exists(fs::path(model_id) / "config.json")) {
                throw std::runtime_error(
                    "Model directory found but missing config.json: " + model_id +
                    ". A valid model directory must contain config.json and model.safetensors files.");
            }
        } else {
            throw std::runtime_error(
                "Model path is a file, not a directory: " + model_id +
                ". Expected a directory with config.json and .safetensors, or a .gguf file.");
        }
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
