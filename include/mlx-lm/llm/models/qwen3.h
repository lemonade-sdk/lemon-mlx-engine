// Copyright © 2024-2025 Apple Inc. — Ported to C++
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

struct Qwen3Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 1000000.0f;
    int head_dim;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;
    bool has_pre_norms = false;  // Per-projection rms_norm (BitNet variants)
    bool bitnet_invert_weight_scales = false;  // 1/scale for bitlinear checkpoints
};

void from_json(const nlohmann::json& j, Qwen3Configuration& c);

class Qwen3Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    // Optional per-projection norms (used by BitNet variants)
    mlx::core::array wq_pre_norm_, wk_pre_norm_, wv_pre_norm_, wo_pre_norm_;
    bool has_pre_norms_ = false;
    float rms_norm_eps_;

    float rope_theta_;
    float rope_scale_;

public:
    explicit Qwen3Attention(const Qwen3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    void enable_pre_norms() { has_pre_norms_ = true; }
};

class Qwen3MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    // Optional per-projection norms (used by BitNet variants)
    mlx::core::array gate_pre_norm_, up_pre_norm_, down_pre_norm_;
    bool has_pre_norms_ = false;

public:
    Qwen3MLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    void enable_pre_norms() { has_pre_norms_ = true; }
};

class Qwen3TransformerBlock {
    Qwen3Attention attention_;
    Qwen3MLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3TransformerBlock(const Qwen3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
    Qwen3Attention& attention() { return attention_; }
    Qwen3MLP& mlp() { return mlp_; }
};

class Qwen3ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen3TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

public:
    explicit Qwen3ModelInner(const Qwen3Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3Model
    : public LanguageModel<Qwen3Model>,
      public KVCacheDimensionProvider<Qwen3Model> {

    friend class LanguageModel<Qwen3Model>;
    friend class KVCacheDimensionProvider<Qwen3Model>;

    Qwen3Configuration config_;
    Qwen3ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Qwen3Model(const Qwen3Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
