// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <optional>

namespace mlx_lm {

// Parameters for text generation.
struct GenerateParameters {
    // Step size for processing the prompt.
    int prefill_step_size = 512;

    // Maximum tokens to generate.
    std::optional<int> max_tokens;

    // Maximum KV cache size. When set, uses RotatingKVCache.
    std::optional<int> max_kv_size;

    // Number of bits for KV cache quantization. nullopt = no quantization.
    std::optional<int> kv_bits;

    // Group size for KV cache quantization.
    int kv_group_size = 64;

    // Step to begin quantized KV cache.
    int quantized_kv_start = 0;

    // Sampling temperature.
    float temperature = 0.6f;

    // Top-p (nucleus) sampling.
    float top_p = 1.0f;

    // Repetition penalty factor.
    std::optional<float> repetition_penalty;

    // Tokens to consider for repetition penalty.
    int repetition_context_size = 20;

    // Pre-allocate KV cache for this many tokens (0=auto grow).
    // Avoids repeated grow-and-copy during generation.
    int ctx_size = 0;
};

} // namespace mlx_lm
