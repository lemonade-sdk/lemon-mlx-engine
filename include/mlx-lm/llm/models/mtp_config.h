#pragma once

namespace mlx_lm {

// Configuration block for MTPHead. Mirrors the subset of
// Qwen35Configuration that MTP actually needs. We keep this self-contained
// so MTPHead can be constructed in tests without pulling in Qwen35Model.
struct MTPHeadConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;       // 0 = hidden_size / num_attention_heads
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int rope_dims = 0;      // 0 = resolved via partial_rotary_factor
    float partial_rotary_factor = 0.25f;  // fraction of head dim using rotary

    // MoE-specific fields (zero = dense mode).
    int num_experts = 0;
    int num_experts_per_tok = 1;
    int shared_expert_intermediate_size = 0;

    int resolved_head_dim() const {
        return head_dim != 0 ? head_dim : hidden_size / num_attention_heads;
    }

    int resolved_rope_dims() const {
        return rope_dims != 0
            ? rope_dims
            : static_cast<int>(resolved_head_dim() * partial_rotary_factor);
    }

    bool is_moe() const { return num_experts > 0; }
};

} // namespace mlx_lm
