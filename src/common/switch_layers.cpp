// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SwitchLayers.swift

#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/quantized_linear.h>
#include <cmath>
#include <iostream>

namespace mx = mlx::core;

namespace mlx_lm {

// --- gather_sort / scatter_unsort ---

std::tuple<mx::array, mx::array, mx::array>
gather_sort(const mx::array& x, const mx::array& indices) {
    int m = indices.shape(-1);
    auto flat_indices = mx::flatten(indices);
    auto order = mx::argsort(flat_indices);
    auto inverse_order = mx::argsort(order);

    auto x_flat = mx::flatten(x, 0, x.ndim() - 3);
    auto sorted_x = mx::contiguous(
        mx::take(x_flat, mx::floor_divide(order, mx::array(m)), 0));
    auto sorted_indices = mx::contiguous(mx::take(flat_indices, order));

    return {sorted_x, sorted_indices, inverse_order};
}

mx::array scatter_unsort(
    const mx::array& x,
    const mx::array& inv_order,
    const mx::Shape* shape)
{
    auto result = mx::contiguous(mx::take(x, inv_order, 0));
    if (shape && !shape->empty()) {
        mx::Shape new_shape(*shape);
        for (int i = 1; i < result.ndim(); ++i) {
            new_shape.push_back(result.shape(i));
        }
        result = mx::reshape(result, new_shape);
    }
    return result;
}

// --- SwitchLinear ---

SwitchLinear::SwitchLinear(int input_dims, int output_dims, int num_experts, bool bias)
    : weight_(mx::zeros({num_experts, output_dims, input_dims})),
      input_dims_(input_dims),
      output_dims_(output_dims),
      num_experts_(num_experts)
{
    if (bias) {
        bias_ = mx::zeros({num_experts, output_dims});
    }
}

mx::array SwitchLinear::operator()(
    const mx::array& x,
    const mx::array& indices,
    bool sorted_indices)
{
    auto* qi = QuantizedWeightRegistry::instance().find(&weight_);

    mx::array result(0.0f);
    if (qi) {
        result = mx::gather_qmm(
            x, weight_, qi->scales, qi->biases,
            /*lhs_indices=*/std::nullopt,
            /*rhs_indices=*/std::optional<mx::array>(indices),
            /*transpose=*/true,
            /*group_size=*/qi->group_size,
            /*bits=*/qi->bits,
            /*mode=*/"affine",
            /*sorted_indices=*/sorted_indices);
    } else {
        auto weight_t = mx::swapaxes(weight_, -1, -2);
        result = mx::gather_mm(x, weight_t, std::nullopt, indices, sorted_indices);
    }

    if (bias_.has_value()) {
        auto b = mx::take(bias_.value(), indices, 0);
        result = mx::add(result, mx::expand_dims(b, -2));
    }

    return result;
}

std::unordered_map<std::string, mx::array*> SwitchLinear::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["weight"] = &weight_;
    if (bias_.has_value()) map["bias"] = &bias_.value();
    return map;
}

// --- SwitchGLU ---

SwitchGLU::SwitchGLU(int input_dims, int hidden_dims, int num_experts, bool bias)
    : gate_proj_(input_dims, hidden_dims, num_experts, bias),
      up_proj_(input_dims, hidden_dims, num_experts, bias),
      down_proj_(hidden_dims, input_dims, num_experts, bias),
      input_dims_(input_dims),
      hidden_dims_(hidden_dims),
      num_experts_(num_experts)
{}

mx::array SwitchGLU::operator()(
    const mx::array& x,
    const mx::array& indices)
{
    // Expand dims for gather_mm: add [-2, -3]
    auto x_expanded = mx::expand_dims(mx::expand_dims(x, -2), -3);

    bool do_sort = (indices.size() >= 64);

    mx::array work_x = x_expanded;
    mx::array idx = indices;
    mx::array inverse_order(0.0f);

    if (do_sort) {
        auto [sx, si, io] = gather_sort(x_expanded, indices);
        work_x = sx;
        idx = si;
        inverse_order = io;
    }

    auto x_up = up_proj_(work_x, idx, do_sort);
    auto x_gate = gate_proj_(work_x, idx, do_sort);

    work_x = down_proj_(swiglu(x_gate, x_up), idx, do_sort);

    if (do_sort) {
        auto shape = indices.shape();
        work_x = scatter_unsort(work_x, inverse_order, &shape);
    }

    // Squeeze axis -2
    return mx::squeeze(work_x, -2);
}

std::unordered_map<std::string, mx::array*> SwitchGLU::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_proj_.weight_map()) map["gate_proj." + k] = v;
    for (auto& [k, v] : up_proj_.weight_map()) map["up_proj." + k] = v;
    for (auto& [k, v] : down_proj_.weight_map()) map["down_proj." + k] = v;
    return map;
}

} // namespace mlx_lm
