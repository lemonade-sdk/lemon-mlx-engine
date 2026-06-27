// Copyright © 2026 — Gemma 4 model implementation
// Supports: sliding_attention (regular) and full_attention (global heads) layers,
// per-layer input gating, and Q/K norms.
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx-lm/llm/models/llama.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Gemma4Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    int head_dim;
    int global_head_dim = 512;
    float rope_theta = 1000000.0f;
    float rope_theta_sliding = 10000.0f;
    float final_logit_softcapping = 30.0f;
    int sliding_window = 512;
    bool tie_word_embeddings = true;
    bool attention_bias = false;
    std::string hidden_act = "gelu_pytorch_tanh";
    std::vector<std::string> layer_types;  // "sliding_attention" or "full_attention"
    int num_kv_shared_layers = 0;
    bool use_double_wide_mlp = true;
    bool enable_moe_block = false;
};

void from_json(const nlohmann::json& j, Gemma4Configuration& c);

class Gemma4Attention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int global_head_dim_;
    float scale_;
    float sliding_scale_;
    bool is_full_attention_ = false;

    // Regular projections (sliding + full)
    mlx::core::array wq_weight_, wk_weight_, wv_weight_, wo_weight_;
    std::optional<mlx::core::array> wq_bias_, wk_bias_, wv_bias_, wo_bias_;
    // Q/K norms
    mlx::core::array q_norm_weight_, k_norm_weight_;

    float rms_norm_eps_;
    float rope_theta_;
    int sliding_window_;

    // RoPE module
    LlamaDynamicNTKScalingRoPE rope_;

public:
    explicit Gemma4Attention(const Gemma4Configuration& args, bool is_full_attention);
    mlx::core::array operator()(const mlx::core::array& x, const AttentionMask& mask, KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma4MLP {
    mlx::core::array gate_weight_, down_weight_, up_weight_;
    std::optional<mlx::core::array> gate_bias_, down_bias_, up_bias_;

    mlx::core::array linear(const mlx::core::array& x, const mlx::core::array& w,
                            const std::optional<mlx::core::array>& b) const;

public:
    explicit Gemma4MLP(const Gemma4Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma4TransformerBlock {
    Gemma4Attention attention_;
    Gemma4MLP mlp_;
    // Layer norms
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    mlx::core::array pre_feedforward_layernorm_weight_;
    mlx::core::array post_feedforward_layernorm_weight_;
    // Per-layer input gating
    mlx::core::array per_layer_input_gate_weight_;
    mlx::core::array per_layer_projection_weight_;
    mlx::core::array post_per_layer_input_norm_weight_;
    mlx::core::array layer_scalar_;
    float rms_norm_eps_;

public:
    explicit Gemma4TransformerBlock(const Gemma4Configuration& args, int layer_idx);
    mlx::core::array operator()(const mlx::core::array& x, const AttentionMask& mask, KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma4ModelInner {
    mlx::core::array embed_tokens_weight_;
    mlx::core::array embed_tokens_per_layer_weight_;
    std::vector<Gemma4TransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    int hidden_size_;

public:
    explicit Gemma4ModelInner(const Gemma4Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Gemma4Model
    : public LanguageModel<Gemma4Model>,
      public KVCacheDimensionProvider<Gemma4Model> {

    friend class LanguageModel<Gemma4Model>;
    friend class KVCacheDimensionProvider<Gemma4Model>;

    Gemma4Configuration config_;
    Gemma4ModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    // Final logit softcapping
    mlx::core::array per_layer_model_projection_weight_;
    mlx::core::array per_layer_projection_norm_weight_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Gemma4Model(const Gemma4Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
