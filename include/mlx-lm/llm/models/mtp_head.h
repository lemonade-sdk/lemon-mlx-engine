// Copyright (c) 2024-2026 Apple Inc. -- Ported to C++
// MTP (Multi-Token Prediction) head -- I7 sub-task 2.
//
// Port of MTPHead + MTPDecoderLayer from mlx-lm-private qwen35_mtp branch:
//   mlx_lm/models/qwen3_5.py:310 MTPDecoderLayer
//   mlx_lm/models/qwen3_5.py:336 MTPHead
//
// Structure:
//   pre_fc_norm_hidden:    RMSNorm(H)
//   pre_fc_norm_embedding: RMSNorm(H)
//   fc:                    Linear(2*H -> H, no bias)
//   layers[0]:             MTPDecoderLayer = full attention + MLP (or SparseMoE)
//   norm:                  RMSNorm(H)
//
// Used by mtp_speculative_generate_step to draft a single follow-on token
// from (last_hidden_state, last_emitted_token) without re-running the trunk.
//
// This header is deliberately model-agnostic on the attention layer: callers
// supply a callable that performs a single attention forward. In the
// scaffolding cut we instantiate it with `Qwen35Attention` and either
// `Qwen35MLP` or `Qwen35SparseMoeBlock`.

#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/llm/models/qwen35.h>
#include <mlx/mlx.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// Single full-attention decoder layer for the MTP head. Mirrors
// `MTPDecoderLayer.__call__` from qwen3_5.py:325. No GatedDeltaNet variant --
// MTP always uses standard attention, per the upstream MTP reference.
class MTPDecoderLayer {
public:
    MTPDecoderLayer(const Qwen35Configuration& args, bool use_moe);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();

private:
    bool use_moe_;
    Qwen35Attention self_attn_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;
    // Exactly one of these is engaged depending on use_moe_.
    std::optional<Qwen35MLP> dense_mlp_;
    std::optional<Qwen35SparseMoeBlock> moe_mlp_;
};

class MTPHead {
public:
    explicit MTPHead(const Qwen35Configuration& args);

    // Run one MTP draft step.
    //   hidden_state:    [B, T, H]  -- pre-norm hidden from the trunk
    //   token_embedding: [B, T, H]  -- embed(last_emitted_token)
    //   mask: causal mask (typically T=1, mask is std::nullopt-equivalent)
    //   cache: pointer to a single-layer KV cache vector owned by the caller.
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

    // Load mtp.* weights as harvested by the model loader (I7 sub-task 1).
    // The keys are expected to have a "mtp." prefix; this strips it.
    void load_mtp_weights(
        const std::unordered_map<std::string, mlx::core::array>& mtp_weights);

private:
    mlx::core::array pre_fc_norm_hidden_weight_;
    mlx::core::array pre_fc_norm_embedding_weight_;
    mlx::core::array fc_weight_;  // [H, 2*H]
    MTPDecoderLayer layer_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;
};

} // namespace mlx_lm
