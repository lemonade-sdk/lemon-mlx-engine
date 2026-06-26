// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/registry.h>
#include <functional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// LLM type registry — maps model_type strings to factory functions.
// Pre-populated with all supported LLM architectures.
ModelTypeRegistry& llm_type_registry();

// LLM model registry — known model configurations.
AbstractModelRegistry& llm_model_registry();

// Load an LLM model from a local directory.
// The directory must contain config.json and *.safetensors files.
ModelContext load_llm_from_directory(
    const std::string& model_directory,
    const ModelConfiguration& config = {});

// Load an LLM model from a local directory with auto-quantization.
// When auto_quantize=true, any unquantized bf16/fp16 model is automatically
// quantized to 4-bit on-the-fly at load time.
ModelContext load_llm_from_directory(
    const std::string& model_directory,
    bool auto_quantize);

// Load an LLM model from a Hugging Face model ID.
// Downloads if not cached locally.
ModelContext load_llm(
    const std::string& model_id,
    const std::string& cache_dir = "");

// Load an LLM model from a Hugging Face model ID with auto-quantization.
// When auto_quantize=true, any unquantized bf16/fp16 model is automatically
// quantized to 4-bit on-the-fly at load time.
ModelContext load_llm(
    const std::string& model_id,
    const std::string& cache_dir,
    bool auto_quantize);

// Load an MTP delta model (MTP head only) by merging with the base model.
// Derives the base model ID by stripping "-MTP" from the delta model ID.
//   mlx-community/Qwen3.5-4B-MTP-4bit -> mlx-community/Qwen3.5-4B-4bit
ModelContext load_mtp_delta_model(
    const std::string& delta_model_id,
    const std::string& cache_dir = "",
    bool auto_quantize = false);

} // namespace mlx_lm
