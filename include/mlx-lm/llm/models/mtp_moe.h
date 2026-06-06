// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// MTP Decoder Layer with MoE (SwitchGLU) MLP for speculative decoding on MoE models.
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/llm/models/mtp_config.h>
#include <mlx/mlx.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

namespace mlx_lm {

// Single full-attention decoder layer for the MTP head with MoE MLP.
// Mirrors MTPDecoderLayer but replaces the dense SwiGLU MLP with a
// SwitchGLU (sparse MoE) block, matching the trunk model's architecture.
class MTPDecoderLayerMoE {
public:
    MTPDecoderLayerMoE(const MTPHeadConfig& args, int num_experts, int top_k = 1);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();

private:
    MTPHeadConfig args_;
    int num_experts_;
    int top_k_;

    // Attention weights (same as dense MTPDecoderLayer).
    mlx::core::array q_proj_weight_;
    mlx::core::array k_proj_weight_;
    mlx::core::array v_proj_weight_;
    mlx::core::array o_proj_weight_;
    mlx::core::array q_norm_weight_;
    mlx::core::array k_norm_weight_;

    // Layer norms.
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;

    // MoE MLP weights.
    mlx::core::array gate_weight_;  // routing gate [num_experts, hidden_size]
    SwitchGLU switch_mlp_;

    // Shared expert (used by Qwen3.5 MoE and Qwen3Next MoE).
    mlx::core::array shared_expert_gate_weight_;
    SwitchGLU shared_expert_;
};

}  // namespace mlx_lm
