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

struct Qwen35Configuration {
    int hidden_size;
    int num_hidden_layers;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int linear_num_value_heads;
    int linear_num_key_heads;
    int linear_key_head_dim;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    float rms_norm_eps;
    int vocab_size;
    float rope_theta = 100000.0f;
    float partial_rotary_factor = 0.25f;
    int max_position_embeddings = 131072;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    std::optional<int> head_dim;
    std::optional<std::unordered_map<std::string, StringOrNumber>> rope_scaling;
    int full_attention_interval = 4;

    // MoE fields
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int decoder_sparse_step = 1;
    int shared_expert_intermediate_size = 0;
    int moe_intermediate_size = 0;
    bool norm_topk_prob = true;

    int resolved_head_dim() const {
        return head_dim.value_or(hidden_size / num_attention_heads);
    }
};

void from_json(const nlohmann::json& j, Qwen35Configuration& c);

// RMSNorm with optional gating (silu(gate) * rms_norm(x))
class Qwen35RMSNormGated {
    mlx::core::array weight_;
    float eps_;

public:
    Qwen35RMSNormGated(int dimensions, float eps);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const std::optional<mlx::core::array>& gate = std::nullopt);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Gated Delta Net -- linear attention for most layers
class Qwen35GatedDeltaNet {
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
    Qwen35RMSNormGated norm_;
    mlx::core::array out_proj_weight_;

public:
    explicit Qwen35GatedDeltaNet(const Qwen35Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs,
                                 const std::optional<mlx::core::array>& mask = std::nullopt,
                                 MambaCache* cache = nullptr);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Standard attention for full-attention layers
class Qwen35Attention {
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
    explicit Qwen35Attention(const Qwen35Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x,
                                 const AttentionMask& mask,
                                 KVCache* cache);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Dense MLP
class Qwen35MLP {
    mlx::core::array gate_proj_weight_, down_proj_weight_, up_proj_weight_;

public:
    Qwen35MLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Sparse MoE block with shared expert
class Qwen35SparseMoeBlock {
    bool norm_topk_prob_;
    int num_experts_;
    int top_k_;

    mlx::core::array gate_weight_;
    SwitchGLU switch_mlp_;

    Qwen35MLP shared_expert_;
    mlx::core::array shared_expert_gate_weight_;

public:
    Qwen35SparseMoeBlock(const Qwen35Configuration& args);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Decoder layer -- either linear attention or standard attention, with MLP or MoE
class Qwen35DecoderLayer {
    bool is_linear_;
    bool use_moe_;

    std::optional<Qwen35Attention> self_attn_;
    std::optional<Qwen35GatedDeltaNet> linear_attn_;

    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    std::optional<Qwen35MLP> dense_mlp_;
    std::optional<Qwen35SparseMoeBlock> moe_mlp_;

public:
    Qwen35DecoderLayer(const Qwen35Configuration& args, int layer_idx);

    bool is_linear() const { return is_linear_; }

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& attention_mask,
        const std::optional<mlx::core::array>& ssm_mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

class Qwen35ModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<Qwen35DecoderLayer> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
    int full_attention_interval_;

public:
    explicit Qwen35ModelInner(const Qwen35Configuration& args);
    mlx::core::array operator()(const mlx::core::array& inputs, std::vector<KVCache>* cache = nullptr);
    mlx::core::array embed_as_linear(const mlx::core::array& x) const;
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    const std::vector<Qwen35DecoderLayer>& get_layers() const { return layers_; }
};

class Qwen35Model
    : public LanguageModel<Qwen35Model> {

    friend class LanguageModel<Qwen35Model>;

    Qwen35Configuration config_;
    Qwen35ModelInner model_;
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
    explicit Qwen35Model(const Qwen35Configuration& args);
    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // MTP accessors (see mtp_head.h / sub-task 1 of I7).
    bool has_mtp() const { return !mtp_weights_.empty(); }
    const std::unordered_map<std::string, mlx::core::array>& mtp_weights() const {
        return mtp_weights_;
    }
};

} // namespace mlx_lm
