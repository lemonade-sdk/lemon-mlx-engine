// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Lille130m.swift
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Lille130mConfiguration {
    int block_size;
    float layer_norm_eps;
    int hidden_size;     // n_embd
    int num_attention_heads;  // n_head
    int num_key_value_heads;  // n_kv_heads
    int num_hidden_layers;    // n_layer
    float rope_theta;
    int vocab_size;
    bool tie_word_embeddings = true;

    // Quantization (optional — read from config.json "quantization" key)
    int quant_bits = 0;
    int quant_group_size = 0;

    int resolved_head_dim() const { return hidden_size / num_attention_heads; }
};

void from_json(const nlohmann::json& j, Lille130mConfiguration& c);

// Lille130m uses: combined QKV proj (qkv_proj), norm-first attention,
// traditional RoPE, MLP with internal norm, different weight naming
// (attention/feed_forward instead of self_attn/mlp, tok_embeddings, etc.)

class Lille130mAttention {
    int num_heads_, num_kv_heads_, head_dim_;
    float scale_;
    mlx::core::array qkv_proj_weight_;
    mlx::core::array out_proj_weight_;
    mlx::core::array norm_weight_;
    float norm_eps_;
    float rope_theta_;
public:
    explicit Lille130mAttention(const Lille130mConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Lille130mMLP {
    mlx::core::array norm_weight_;
    mlx::core::array gate_weight_, up_weight_, down_weight_;
    float norm_eps_;
public:
    explicit Lille130mMLP(const Lille130mConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Lille130mBlock {
    Lille130mAttention attention_;
    Lille130mMLP feed_forward_;
public:
    explicit Lille130mBlock(const Lille130mConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Lille130mModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Lille130mBlock> layers_;
    mlx::core::array norm_weight_;
    float norm_eps_;
public:
    explicit Lille130mModelInner(const Lille130mConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Lille130mModel
    : public LanguageModel<Lille130mModel>,
      public KVCacheDimensionProvider<Lille130mModel> {

    friend class LanguageModel<Lille130mModel>;
    friend class KVCacheDimensionProvider<Lille130mModel>;

    Lille130mConfiguration config_;
    Lille130mModelInner transformer_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Lille130mModel(const Lille130mConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
