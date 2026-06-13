// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Fused HIP kernel for MTP delta intermediate computation.
// Provides a ROCm-optimized path for gated delta operations used in MTP draft generation.
#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mlx_lm {

// Configuration for the MTP delta fused kernel.
struct MTPDeltaConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int num_value_heads = 0;
    int num_key_heads = 0;
    int key_head_dim = 0;
    int value_head_dim = 0;
    int conv_kernel_dim = 4;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 100000.0f;
    float partial_rotary_factor = 0.25f;

    int value_dim() const { return num_value_heads * value_head_dim; }
    int key_dim() const { return num_key_heads * key_head_dim; }
    int conv_dim() const { return key_dim() * 2 + value_dim(); }
    int heads_ratio() const { return (num_key_heads > 0) ? num_value_heads / num_key_heads : 1; }
    int resolved_head_dim() const { return (num_attention_heads > 0) ? hidden_size / num_attention_heads : key_head_dim; }
};

// Fused gated delta computation for MTP draft forward pass.
// On ROCm platforms, this launches a custom HIP kernel that fuses:
//   1. Conv1d + SiLU
//   2. Q/K/V splitting + RMSNorm
//   3. Beta/g gating computation
//   4. GDN recurrence step
//   5. Output projection preparation
//
// On non-ROCm platforms, falls back to MLX graph compose.
//
// Parameters:
//   inputs        - [B, S, H] input hidden state
//   conv_weight   - [conv_dim, 1, conv_kernel_dim] conv1d weights
//   qkv_weight    - [2*key_dim + value_dim, H] in_proj_qkv weights
//   z_weight      - [value_dim, H] in_proj_z weights
//   dt_bias       - [num_value_heads] timestep bias
//   a_log         - [num_value_heads] log of A parameter
//   state         - optional [B, Hv, Dv, Dk] SSM state
//   config        - kernel configuration
//
// Returns: {output [B, S, H], new_state [B, Hv, Dv, Dk]}
std::pair<mlx::core::array, std::optional<mlx::core::array>>
mtp_delta_fused(
    const mlx::core::array& inputs,
    const mlx::core::array& conv_weight,
    const mlx::core::array& qkv_weight,
    const mlx::core::array& z_weight,
    const mlx::core::array& dt_bias,
    const mlx::core::array& a_log,
    const std::optional<mlx::core::array>& state,
    const MTPDeltaConfig& config);

// Full MTP draft forward pass using the fused delta kernel.
// Orchestrates the complete MTP draft step:
//   1. Embed current token
//   2. Run MTP head (dense or MoE)
//   3. Apply delta fusion for linear attention layers
//   4. Sample draft tokens
//
// Parameters:
//   hidden_state  - [B, T, H] hidden state from trunk
//   token_embed   - [B, T, H] embedded draft token
//   mtp_weights   - pre-loaded MTP weight arrays
//   config        - delta kernel configuration
//   use_moe       - if true, uses MoE MLP path
//
// Returns: pre-norm hidden state ready for lm_head (same shape as input)
mlx::core::array mtp_draft_forward(
    const mlx::core::array& hidden_state,
    const mlx::core::array& token_embed,
    const std::unordered_map<std::string, mlx::core::array>& mtp_weights,
    const MTPDeltaConfig& config,
    bool use_moe = false);

} // namespace mlx_lm
