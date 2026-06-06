#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/llm/models/mtp_config.h>
#include <mlx-lm/llm/models/mtp_moe.h>
#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Single full-attention decoder layer for the MTP head. Mirrors
// `MTPDecoderLayer.__call__` from qwen3_5.py:325. Self-attention + MLP +
// pre/post RMSNorm; no GatedDeltaNet variant -- MTP always uses standard
// attention per the upstream MTP reference.
class MTPDecoderLayer {
public:
    explicit MTPDecoderLayer(const MTPHeadConfig& args);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();

private:
    MTPHeadConfig args_;

    // Attention weights.
    mlx::core::array q_proj_weight_;
    mlx::core::array k_proj_weight_;
    mlx::core::array v_proj_weight_;
    mlx::core::array o_proj_weight_;
    mlx::core::array q_norm_weight_;
    mlx::core::array k_norm_weight_;

    // Layer norms.
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;

    // MLP weights (SwiGLU).
    mlx::core::array gate_proj_weight_;
    mlx::core::array up_proj_weight_;
    mlx::core::array down_proj_weight_;
};

class MTPHead {
public:
    explicit MTPHead(const MTPHeadConfig& args);

    // Factory: create an MTPHead with MoE decoder layer.
    static MTPHead create_moe(const MTPHeadConfig& args);

    // Run one MTP draft step.
    //   hidden_state:    [B, T, H]  -- pre-norm hidden from the trunk
    //   token_embedding: [B, T, H]  -- embed(last_emitted_token)
    //   mask: attention mask (typically `AttentionMask::none()` for T=1)
    //   cache: single-layer KV cache owned by the caller.
    // Returns the pre-norm hidden state of the MTP layer (same shape as input).
    mlx::core::array operator()(
        const mlx::core::array& hidden_state,
        const mlx::core::array& token_embedding,
        const AttentionMask& mask,
        KVCache* cache);

    // Apply the final RMSNorm. Caller passes through `lm_head` / tied
    // embeddings to obtain draft logits.
    mlx::core::array apply_output_norm(const mlx::core::array& h) const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Load mtp.* weights as harvested by the model loader.
    // Strips any prefix up to and including "mtp.".
    void load_mtp_weights(
        const std::unordered_map<std::string, mlx::core::array>& mtp_weights);

private:
    // Sentinel constructor — creates MTPHead for MoE use without
    // initializing dense_layer_, avoiding a wasted SwiGLU allocation
    // that create_moe() immediately resets.
    MTPHead(const MTPHeadConfig& args, int /* moe_sentinel */);

    MTPHeadConfig args_;
    mlx::core::array pre_fc_norm_hidden_weight_;
    mlx::core::array pre_fc_norm_embedding_weight_;
    mlx::core::array fc_weight_;  // [H, 2*H]

    // Dense layer (used when num_experts == 0).
    std::optional<MTPDecoderLayer> dense_layer_;
    // MoE layer (used when num_experts > 0).
    std::unique_ptr<class MTPDecoderLayerMoE> moe_layer_;

    mlx::core::array norm_weight_;
};

}  // namespace mlx_lm
