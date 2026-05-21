// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of MiMo.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct MiMoConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int max_position_embeddings = 32768;
    float rope_theta = 10000.0f;
    bool rope_traditional = false;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    bool tie_word_embeddings = false;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, MiMoConfiguration& c);

class MiMoAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    mlx::core::array wq_bias_, wk_bias_, wv_bias_; // MiMo uses bias on QKV
    float rope_theta_;
    bool rope_traditional_;
    float rope_scale_;
public:
    explicit MiMoAttention(const MiMoConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiMoMLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
public:
    MiMoMLP(int dim, int hidden_dim);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiMoTransformerBlock {
    MiMoAttention self_attn_;
    MiMoMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
public:
    explicit MiMoTransformerBlock(const MiMoConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiMoModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<MiMoTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
public:
    explicit MiMoModelInner(const MiMoConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class MiMoModel
    : public LanguageModel<MiMoModel>,
      public KVCacheDimensionProvider<MiMoModel> {

    friend class LanguageModel<MiMoModel>;
    friend class KVCacheDimensionProvider<MiMoModel>;

    MiMoConfiguration config_;
    MiMoModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    // MTP scaffolding (I7 sub-task 1): stash model.mtp_layers.* weights.
    std::unordered_map<std::string, mlx::core::array> mtp_weights_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit MiMoModel(const MiMoConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    bool has_mtp() const { return !mtp_weights_.empty(); }
    const std::unordered_map<std::string, mlx::core::array>& mtp_weights() const {
        return mtp_weights_;
    }
};

} // namespace mlx_lm
