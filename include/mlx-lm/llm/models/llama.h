// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// Llama / Mistral configuration.
struct LlamaConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    std::optional<int> head_dim;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    std::optional<int> max_position_embeddings;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    bool mlp_bias = false;
    std::string hidden_act = "silu";
    // Some MLX BitLinear checkpoints store weight_scale as an inverse divisor
    // (scale = 1 / weight_scale). True BitNet/autobitlinear checkpoints store
    // the direct multiplier.
    bool bitnet_invert_weight_scales = false;
    // For 1-bit models with silu activation that still have sub-norms
    // (1bitLLM style). Setting this to true enables attn_sub_norm and
    // ffn_sub_norm even when hidden_act != "relu2".
    bool bitnet_has_sub_norm = false;
    // Activation quantization bits (0 = off). 1bitLLM uses 8-bit activation
    // quantization. When set, linear_fwd will quantize activations before
    // each matmul to match BitLinear's activation_quant behavior.
    int activation_bits = 0;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, LlamaConfiguration& c);

// Compute the adjusted base frequency for llama3 RoPE scaling.
float compute_base_frequency(
    float base, int dims, const std::string& rope_type,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling);

// Llama Dynamic NTK Scaling RoPE.
class LlamaDynamicNTKScalingRoPE {
    int dims_;
    int max_position_embeddings_;
    bool traditional_;
    std::optional<float> base_;
    float scale_;
    std::string rope_type_;
    std::optional<mlx::core::array> freqs_;

    void compute_freqs(
        const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling);

public:
    LlamaDynamicNTKScalingRoPE(
        int dims,
        std::optional<int> max_position_embeddings = std::nullopt,
        bool traditional = false,
        float base = 10000.0f,
        float scale = 1.0f,
        const std::string& rope_type = "default",
        const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling = std::nullopt);

    mlx::core::array operator()(const mlx::core::array& x, int offset = 0);
};

// Llama Attention.
class LlamaAttention {
    const LlamaConfiguration& args_;
    float scale_;

    // Weight matrices (Linear layers are just arrays + bias).
    mlx::core::array wq_weight_;
    std::optional<mlx::core::array> wq_bias_;
    mlx::core::array wk_weight_;
    std::optional<mlx::core::array> wk_bias_;
    mlx::core::array wv_weight_;
    std::optional<mlx::core::array> wv_bias_;
    mlx::core::array wo_weight_;
    std::optional<mlx::core::array> wo_bias_;

    LlamaDynamicNTKScalingRoPE rope_;

    // Linear forward helper
    mlx::core::array linear(const mlx::core::array& x,
                            const mlx::core::array& weight,
                            const std::optional<mlx::core::array>& bias) const;

public:
    explicit LlamaAttention(const LlamaConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    // Access to weight maps for loading.
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Llama MLP.
class LlamaMLP {
    mlx::core::array gate_weight_;
    std::optional<mlx::core::array> gate_bias_;
    mlx::core::array down_weight_;
    std::optional<mlx::core::array> down_bias_;
    mlx::core::array up_weight_;
    std::optional<mlx::core::array> up_bias_;

    mlx::core::array linear(const mlx::core::array& x,
                            const mlx::core::array& weight,
                            const std::optional<mlx::core::array>& bias) const;

public:
    explicit LlamaMLP(const LlamaConfiguration& args);

    mlx::core::array operator()(const mlx::core::array& x);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Llama Transformer Block (single layer).
class LlamaTransformerBlock {
    LlamaAttention attention_;
    LlamaMLP mlp_;

    // RMSNorm parameters
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    mlx::core::array rms_norm(const mlx::core::array& x,
                               const mlx::core::array& weight) const;

public:
    explicit LlamaTransformerBlock(const LlamaConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Llama Model Inner (embedding + layers + norm).
class LlamaModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<LlamaTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

    mlx::core::array rms_norm(const mlx::core::array& x,
                               const mlx::core::array& weight) const;

public:
    explicit LlamaModelInner(const LlamaConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& inputs,
        std::vector<KVCache>* cache = nullptr);

    // Embedding used as LM head when weights are tied.
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// LlamaModel — top-level model for Llama and Mistral.
// Uses CRTP for LanguageModel interface and KVCacheDimensionProvider.
class LlamaModel
    : public LanguageModel<LlamaModel>,
      public KVCacheDimensionProvider<LlamaModel> {

    friend class LanguageModel<LlamaModel>;
    friend class KVCacheDimensionProvider<LlamaModel>;

    LlamaConfiguration config_;
    LlamaModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    // LanguageModel CRTP interface
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache,
                       const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array>
    sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit LlamaModel(const LlamaConfiguration& args);

    // KVCacheDimensionProvider requires this
    const std::vector<int>& kv_heads() const { return kv_heads_; }

    int vocab_size() const { return config_.vocab_size; }

    // Load weights from a weight map.
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);

    // Expose all weight pointers for loading.
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
