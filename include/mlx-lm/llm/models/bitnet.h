// BitNet 1.58-bit model for lemon-mlx-engine
// Architecture: Llama variant with relu2 activation and ternary weights
// Weights stored as uint8 packed ternary, dequantized at load time
// "Little bones" — Gord Downie
#pragma once

#include <mlx-lm/llm/models/llama.h>

namespace mlx_lm {

// BitNet reuses Llama configuration — differences are in the forward pass
using BitNetConfiguration = LlamaConfiguration;

// BitNet reuses Llama's from_json since they share the same config type.

// Dequantize uint8 packed ternary weights at load time
mlx::core::array dequantize_bitnet_weight(
    const mlx::core::array& packed_weight,
    const mlx::core::array& weight_scale,
    int out_features);

// --- BitNet layers (relu_squared + sub-layer norms) ---

class BitNetMLP {
    mlx::core::array gate_proj_w_;
    mlx::core::array down_proj_w_;
    mlx::core::array up_proj_w_;
    mlx::core::array ffn_sub_norm_weight_;
    float rms_norm_eps_;

public:
    explicit BitNetMLP(float rms_norm_eps = 1e-5f)
        : gate_proj_w_(mlx::core::zeros({1})),
          down_proj_w_(mlx::core::zeros({1})),
          up_proj_w_(mlx::core::zeros({1})),
          ffn_sub_norm_weight_(mlx::core::zeros({1})),
          rms_norm_eps_(rms_norm_eps) {}
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map(const std::string& prefix);
};

class BitNetAttention {
    const BitNetConfiguration& args_;
    float scale_;
    LlamaDynamicNTKScalingRoPE rope_;
    mlx::core::array q_proj_w_;
    mlx::core::array k_proj_w_;
    mlx::core::array v_proj_w_;
    mlx::core::array o_proj_w_;
    mlx::core::array attn_sub_norm_weight_;

public:
    explicit BitNetAttention(const BitNetConfiguration& args);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x, const AttentionMask& mask,
                                KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map(const std::string& prefix);
};

class BitNetTransformerBlock {
    BitNetAttention self_attn_;
    BitNetMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit BitNetTransformerBlock(const BitNetConfiguration& args);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x, const AttentionMask& mask,
                                KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map(const std::string& prefix);
};

class BitNetModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<BitNetTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit BitNetModelInner(const BitNetConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& inputs,
        std::vector<KVCache>* cache = nullptr);

    mlx::core::array embed_as_linear(const mlx::core::array& x) const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Top-level BitNet model — CRTP pattern matching LlamaModel
class BitNetModel
    : public LanguageModel<BitNetModel>,
      public KVCacheDimensionProvider<BitNetModel> {

    friend class LanguageModel<BitNetModel>;
    friend class KVCacheDimensionProvider<BitNetModel>;

    BitNetConfiguration config_;
    BitNetModelInner model_;
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
    explicit BitNetModel(const BitNetConfiguration& args);

    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }

    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
