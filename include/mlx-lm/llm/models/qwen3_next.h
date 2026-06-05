// Copyright © 2024-2025 Apple Inc.
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx/mlx.h>
#include <mlx-lm/common/gated_delta.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Qwen3NextConfiguration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int linear_num_value_heads;
    int linear_num_key_heads;
    int linear_key_head_dim;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    int num_experts;
    int num_experts_per_tok;
    int decoder_sparse_step;
    int shared_expert_intermediate_size;
    std::vector<int> mlp_only_layers;
    int moe_intermediate_size;
    float rms_norm_eps;
    int vocab_size;
    int num_key_value_heads;
    float rope_theta = 1000000.0f;
    float partial_rotary_factor = 1.0f;
    int max_position_embeddings = 32768;
    bool norm_topk_prob = false;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    std::optional<int> head_dim;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    int full_attention_interval = 4;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Qwen3NextConfiguration& c);

// RMSNorm with optional gating
class Qwen3NextRMSNormGated {
    mlx::core::array weight_;
    float eps_;

public:
    Qwen3NextRMSNormGated(int dimensions, float eps);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& gate = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Standard attention for full-attention layers
class Qwen3NextAttention {
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    float scale_;

    mlx::core::array q_proj_weight_, k_proj_weight_, v_proj_weight_, o_proj_weight_;
    std::optional<mlx::core::array> q_proj_bias_, k_proj_bias_, v_proj_bias_, o_proj_bias_;
    mlx::core::array q_norm_weight_, k_norm_weight_;
    float rms_norm_eps_;
    float rope_theta_;
    int rope_dims_;

public:
    explicit Qwen3NextAttention(const Qwen3NextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gated Delta Net — linear attention for most layers
class Qwen3NextGatedDeltaNet {
    int hidden_size_;
    int num_v_heads_;
    int num_k_heads_;
    int head_k_dim_;
    int head_v_dim_;
    int key_dim_;
    int value_dim_;
    int conv_kernel_size_;
    int conv_dim_;

    mlx::core::array conv1d_weight_;
    mlx::core::array in_proj_qkvz_weight_;
    mlx::core::array in_proj_ba_weight_;
    mlx::core::array dt_bias_;
    mlx::core::array a_log_;
    Qwen3NextRMSNormGated norm_;
    mlx::core::array out_proj_weight_;

public:
    explicit Qwen3NextGatedDeltaNet(const Qwen3NextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt,
                                 MambaCache* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Dense MLP
class Qwen3NextMLP {
    mlx::core::array gate_proj_weight_, down_proj_weight_, up_proj_weight_;

public:
    Qwen3NextMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Sparse MoE block with shared expert
class Qwen3NextSparseMoeBlock {
    bool norm_topk_prob_;
    int num_experts_;
    int top_k_;

    mlx::core::array gate_weight_;
    SwitchGLU switch_mlp_;

    Qwen3NextMLP shared_expert_;
    mlx::core::array shared_expert_gate_weight_;

public:
    Qwen3NextSparseMoeBlock(const Qwen3NextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Decoder layer — either linear attention or standard attention, with MLP or MoE
class Qwen3NextDecoderLayer {
    bool is_linear_;
    bool use_moe_;

    std::optional<Qwen3NextAttention> self_attn_;
    std::optional<Qwen3NextGatedDeltaNet> linear_attn_;

    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    std::optional<Qwen3NextMLP> dense_mlp_;
    std::optional<Qwen3NextSparseMoeBlock> moe_mlp_;

public:
    Qwen3NextDecoderLayer(const Qwen3NextConfiguration& args, int layer_idx);

    bool is_linear() const { return is_linear_; }

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& attention_mask,
        const std::optional<mlx::core::array>& ssm_mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen3NextModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen3NextDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    int full_attention_interval_;

public:
    explicit Qwen3NextModelInner(const Qwen3NextConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    const std::vector<Qwen3NextDecoderLayer>& get_layers() const { return layers_; }
};

class Qwen3NextModel
    : public LanguageModel<Qwen3NextModel> {

    friend class LanguageModel<Qwen3NextModel>;

    Qwen3NextConfiguration config_;
    Qwen3NextModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    // Stash mtp.* weights for MTPHead.
    std::unordered_map<std::string, mlx::core::array> mtp_weights_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

    // Custom new_cache: MambaCache for linear layers, KVCacheSimple for attention layers
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Qwen3NextModel(const Qwen3NextConfiguration& args);
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
