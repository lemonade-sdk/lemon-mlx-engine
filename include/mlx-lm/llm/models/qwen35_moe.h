// Copyright (c) 2024-2025 Apple Inc.
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/llm/llm_model.h>
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx/mlx.h>
#include <mlx-lm/common/gated_delta.h>
#include <mlx-lm/llm/models/qwen3_next.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct Qwen35MoEConfiguration {
    int hidden_size = 4096;
    int num_hidden_layers = 32;
    int intermediate_size = 14336;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    int linear_num_value_heads = 64;
    int linear_num_key_heads = 16;
    int linear_key_head_dim = 192;
    int linear_value_head_dim = 128;
    int linear_conv_kernel_dim = 4;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int decoder_sparse_step = 1;
    int shared_expert_intermediate_size = 0;
    int moe_intermediate_size = 0;
    float rms_norm_eps = 1e-6f;
    int vocab_size = 151936;
    float rope_theta = 100000.0f;
    float partial_rotary_factor = 0.25f;
    int max_position_embeddings = 131072;
    bool norm_topk_prob = true;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    std::optional<int> head_dim;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    int full_attention_interval = 4;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Qwen35MoEConfiguration& c);

// ── Qwen3.5 VLM Configuration (vision tower + multimodal) ──────────────

struct Qwen35VLVisionConfiguration {
    std::string model_type;
    int depth = 24;
    int hidden_size = 1024;
    int intermediate_size = 4096;
    int out_hidden_size = 2560;
    int num_heads = 16;
    int patch_size = 16;
    int spatial_merge_size = 2;
    int temporal_patch_size = 2;
    int num_position_embeddings = 2304;
    int in_channels = 3;
    std::string hidden_act = "gelu_pytorch_tanh";
    std::vector<int> deepstack_visual_indexes;
    float rms_norm_eps = 1e-6f;
    // PatchMerger intermediate dimension. 0 = auto (hidden * spatial_merge^2).
    // Qwen3.5-4B uses 5120 explicitly.
    int merger_intermediate_size = 0;
};

void from_json(const nlohmann::json& j, Qwen35VLVisionConfiguration& c);

struct Qwen35VLBaseConfiguration {
    std::string model_type;
    int vocab_size = 248320;
    int image_token_id = 248056;
    int video_token_id = 248057;
    int vision_start_token_id = 248053;
    int vision_end_token_id = 248054;
    int vision_token_id = 248055;
};

void from_json(const nlohmann::json& j, Qwen35VLBaseConfiguration& c);

struct Qwen35VLConfiguration {
    Qwen35MoEConfiguration text_config;
    Qwen35VLVisionConfiguration vision_config;
    Qwen35VLBaseConfiguration base_config;
};

void from_json(const nlohmann::json& j, Qwen35VLConfiguration& c);

// Standard attention with sigmoid gating on q_proj output
class Qwen35MoEAttention {
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
    explicit Qwen35MoEAttention(const Qwen35MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gated Delta Net -- linear attention for most layers
class Qwen35MoEGatedDeltaNet {
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
    mlx::core::array in_proj_qkv_weight_;
    mlx::core::array in_proj_z_weight_;
    mlx::core::array in_proj_b_weight_;
    mlx::core::array in_proj_a_weight_;
    mlx::core::array dt_bias_;
    mlx::core::array a_log_;
    Qwen3NextRMSNormGated norm_;
    mlx::core::array out_proj_weight_;

public:
    explicit Qwen35MoEGatedDeltaNet(const Qwen35MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt,
                                 MambaCache* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Dense MLP (reuses gate/up/down pattern)
class Qwen35MoEMLP {
    mlx::core::array gate_proj_weight_, down_proj_weight_, up_proj_weight_;

public:
    Qwen35MoEMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Sparse MoE block with shared expert
class Qwen35MoESparseMoeBlock {
    bool norm_topk_prob_;
    int num_experts_;
    int top_k_;

    mlx::core::array gate_weight_;
    SwitchGLU switch_mlp_;

    Qwen35MoEMLP shared_expert_;
    mlx::core::array shared_expert_gate_weight_;

public:
    Qwen35MoESparseMoeBlock(const Qwen35MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Decoder layer -- either linear attn or standard attn, with dense MLP or MoE
class Qwen35MoEDecoderLayer {
    bool is_linear_;
    bool use_moe_;

    std::optional<Qwen35MoEAttention> self_attn_;
    std::optional<Qwen35MoEGatedDeltaNet> linear_attn_;

    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    std::optional<Qwen35MoEMLP> dense_mlp_;
    std::optional<Qwen35MoESparseMoeBlock> moe_mlp_;

public:
    Qwen35MoEDecoderLayer(const Qwen35MoEConfiguration& args, int layer_idx);

    bool is_linear() const { return is_linear_; }

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& attention_mask,
        const std::optional<mlx::core::array>& ssm_mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen35MoEModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen35MoEDecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    int full_attention_interval_;

public:
    explicit Qwen35MoEModelInner(const Qwen35MoEConfiguration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);

    // Overloaded operator() for multimodal use: accepts pre-computed embeddings.
    // position_ids, visual_mask, and deepstack_embeds are suppressed (unused for Qwen3.5).
    mlx::core::array operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt,
        const AttentionMask& mask = AttentionMask{},
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& visual_mask = std::nullopt,
        const std::vector<mlx::core::array>* deepstack_embeds = nullptr);

    mlx::core::array embed_tokens(const mlx::core::array& input_ids) const;
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    mlx::core::array apply_lm_head(const mlx::core::array& hidden) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    const std::vector<Qwen35MoEDecoderLayer>& get_layers() const { return layers_; }
};

class Qwen35MoEModel
    : public LanguageModel<Qwen35MoEModel> {

    friend class LanguageModel<Qwen35MoEModel>;

    Qwen35MoEConfiguration config_;
    Qwen35MoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    // Stash mtp.* weights for MTPHead.
    std::unordered_map<std::string, mlx::core::array> mtp_weights_;
    std::optional<class MTPHead> mtp_head_;
    std::optional<MTPHeadConfig> mtp_head_cfg_;

    // Build MTPHead from config and load stashed weights.
    void build_mtp_head();

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

    // Custom new_cache: MambaCache for linear layers, KVCacheSimple for attention layers
    std::vector<KVCache> new_cache_impl(const GenerateParameters& params);

public:
    explicit Qwen35MoEModel(const Qwen35MoEConfiguration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    bool has_mtp() const { return mtp_head_.has_value(); }

    // Set MTP head config before load_weights(). Used by load_mtp_delta_model()
    // to pass MTP-specific architectural parameters from the delta model's config.json.
    void set_mtp_head_config(const MTPHeadConfig& cfg) { mtp_head_cfg_ = cfg; }

    const std::unordered_map<std::string, mlx::core::array>& mtp_weights() const {
        return mtp_weights_;
    }
    MTPHead* get_mtp_head() {
        return mtp_head_ ? &mtp_head_.value() : nullptr;
    }
    const MTPHead* get_mtp_head() const {
        return mtp_head_ ? &mtp_head_.value() : nullptr;
    }

    // Delegate embedding/lm_head access to the inner model.
    // Required for SFINAE binding of embed_fn and apply_lm_head_fn
    // in ModelContext, which MTP speculative decoding depends on.
    // Note: embed_as_linear is bound as embed_fn by SFINAE, but MTP
    // passes raw token IDs (not embeddings), so we delegate to
    // embed_tokens which does the correct embedding lookup (mx::take).
    // We also ensure the input is at least 2D so mx::take produces
    // 3D [B, T, H] output as the MTP head expects.
    mlx::core::array embed_as_linear(const mlx::core::array& x) const {
        auto tokens = x;
        if (tokens.ndim() < 2) {
            tokens = mlx::core::reshape(tokens, {1, static_cast<int>(tokens.size())});
        }
        return model_.embed_tokens(tokens);
    }
    mlx::core::array apply_lm_head(const mlx::core::array& hidden) const {
        // For untied embeddings, use the separate lm_head weight if available.
        // For tied embeddings (or if lm_head_weight_ was cleared), delegate to
        // the inner model which uses embed_tokens_weight_.
        if (lm_head_weight_.has_value()) {
            return mlx::core::matmul(hidden, mlx::core::transpose(lm_head_weight_.value()));
        }
        return model_.apply_lm_head(hidden);
    }

    // Create a single KVCache for the MTP head (one decoder layer).
    std::vector<KVCache> new_mtp_cache(const GenerateParameters& params) const;
};

} // namespace mlx_lm
