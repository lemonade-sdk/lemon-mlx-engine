// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SwitchLayers.swift — MoE routing via gather_mm
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace mlx_lm {

// Sort indices for efficient gather_mm dispatch.
// Returns (sorted_x, sorted_indices, inverse_order).
std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
gather_sort(const mlx::core::array& x, const mlx::core::array& indices);

// Unsort results back to original order.
mlx::core::array scatter_unsort(
    const mlx::core::array& x,
    const mlx::core::array& inv_order,
    const mlx::core::Shape* shape = nullptr);

// SwitchLinear — a linear layer with multiple expert weight matrices.
// Uses gather_mm for efficient expert dispatch.
class SwitchLinear {
    mlx::core::array weight_;  // [num_experts, output_dims, input_dims]
    std::optional<mlx::core::array> bias_;  // [num_experts, output_dims] or nullopt

    int input_dims_;
    int output_dims_;
    int num_experts_;

    // Cache for the default lhs_indices (an identity arange over x's batch
    // dims). gather_qmm would otherwise regenerate this arange on every forward
    // pass via indices_or_default(); during decode the batch shape is constant,
    // so we compute it once per shape and reuse the already-evaluated array,
    // eliminating one arange kernel launch per expert projection per token.
    std::optional<mlx::core::array> lhs_indices_cache_;
    mlx::core::Shape lhs_indices_cache_shape_;

    const mlx::core::array& default_lhs_indices(const mlx::core::array& x);

public:
    SwitchLinear(int input_dims, int output_dims, int num_experts, bool bias = false);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& indices,
        bool sorted_indices = false);

    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Accessors used by SwitchGLU to build a fused gate+up expert projection.
    const mlx::core::array& weight() const { return weight_; }
    int output_dims() const { return output_dims_; }
    int num_experts() const { return num_experts_; }

    // Replace this layer's weight with a pre-built (already-evaluated) fused
    // quantized weight and register its quant metadata. Bit-width agnostic:
    // the caller supplies the packed weight, scales, optional biases, group_size
    // and bits (works for 4/5/6/8-bit). The weight_ address is stable (member of
    // a long-lived layer), so registry lookup by &weight_ stays valid.
    void adopt_fused_weight(mlx::core::array w,
                            mlx::core::array scales,
                            std::optional<mlx::core::array> biases,
                            int group_size, int bits,
                            const std::string& mode = "affine");

    // Free this layer's weight buffer and drop its quant metadata. Called after
    // its data has been folded into a fused projection so VRAM stays neutral
    // (the fused weight is the same total size as the originals combined).
    void release_weight();
};

// SwitchGLU — gated linear unit with expert routing.
// Applies gate_proj, up_proj, down_proj via SwitchLinear with silu activation.
class SwitchGLU {
    SwitchLinear gate_proj_;
    SwitchLinear up_proj_;
    SwitchLinear down_proj_;
    // Lazily-built fused gate+up projection ([E, 2*hidden, input]): one
    // gather_qmm instead of two, halving expert-projection launches and
    // doubling per-launch work (better occupancy / effective bandwidth on the
    // decode M=1 path). Built from gate_proj_/up_proj_ on first forward.
    SwitchLinear gate_up_proj_;
    bool gate_up_tried_ = false;
    bool gate_up_ready_ = false;

    int input_dims_;
    int hidden_dims_;
    int num_experts_;

    // Build (once) the fused gate+up expert weight by concatenating the packed
    // quantized gate/up weights, scales and (optional) biases along the output
    // axis. Bit-width agnostic. Returns false (and leaves the two-call path in
    // use) if either projection isn't registered-quantized or their group/bits
    // differ. Idempotent.
    bool ensure_gate_up_fused();

public:
    SwitchGLU(int input_dims, int hidden_dims, int num_experts, bool bias = false);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const mlx::core::array& indices);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
