// Copyright © 2025 — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Check if a file is a GGUF file by extension or magic bytes.
bool is_gguf_file(const std::string& path);

// Read GGUF metadata (string key-value pairs) without loading tensors.
// Returns the metadata map from the GGUF header.
std::unordered_map<std::string, std::string>
gguf_read_metadata(const std::string& path);

// Synthesize a config.json-equivalent from GGUF metadata string map.
nlohmann::json gguf_config_from_metadata(
    const std::unordered_map<std::string, std::string>& meta);

// Load weights from a GGUF file with full quant format support.
// Dequantizes all tensors to fp16 and remaps to HuggingFace naming.
std::unordered_map<std::string, mlx::core::array>
load_gguf_weights(const std::string& path);

} // namespace mlx_lm
