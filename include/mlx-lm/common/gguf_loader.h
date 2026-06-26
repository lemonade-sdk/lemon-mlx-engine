// Copyright © 2025 — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Check if a file is a GGUF file by extension or magic bytes.
bool is_gguf_file(const std::string& path);

// Synthesize a config.json-equivalent from GGUF metadata.
nlohmann::json gguf_config_from_metadata(
    const std::unordered_map<std::string, mlx::core::GGUFMetaData>& meta);

// Load weights from a GGUF file with remapping to HuggingFace names.
std::unordered_map<std::string, mlx::core::array>
load_gguf_weights(const std::string& path);

} // namespace mlx_lm
