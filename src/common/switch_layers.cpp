// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of SwitchLayers.swift

#include <mlx-lm/common/switch_layers.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/quantized_linear.h>
#include <cmath>
#include <cstdlib>
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

// Build (and cache) the default lhs_indices that gather_qmm/gather_mm would
// otherwise create internally on every call: an identity arange over x's batch
// dimensions (x.shape minus the trailing two matmul dims). This matches MLX's
// indices_or_default() exactly, so results are bit-identical — we just avoid
// relaunching the arange kernel each decode step when the shape is unchanged.
const mx::array& SwitchLinear::default_lhs_indices(const mx::array& x) {
    mx::Shape batch_shape(x.shape().begin(), x.shape().end() - 2);
    if (!lhs_indices_cache_.has_value() ||
        lhs_indices_cache_shape_ != batch_shape) {
        int64_t total = 1;
        for (auto d : batch_shape) total *= d;
        lhs_indices_cache_ = mx::reshape(
            mx::arange(static_cast<int>(total), mx::uint32), batch_shape);
        lhs_indices_cache_shape_ = batch_shape;
    }
    return lhs_indices_cache_.value();
}

mx::array SwitchLinear::operator()(
    const mx::array& x,
    const mx::array& indices,
    bool sorted_indices)
{
    auto* qi = QuantizedWeightRegistry::instance().find(&weight_);

    mx::array result(0.0f);
    // Explicit lhs_indices forces right_sorted_=false in gather_qmm
    // (right_sorted_ = sorted_indices && !lhs_indices), which disables the
    // expert-batched prefill kernel. Pass nullopt when sorting so it's reachable.
    std::optional<mx::array> lhs =
        sorted_indices ? std::nullopt
                       : std::optional<mx::array>(default_lhs_indices(x));
    if (qi) {
        result = mx::gather_qmm(
            x, weight_, qi->scales, qi->biases,
            /*lhs_indices=*/lhs,
            /*rhs_indices=*/std::optional<mx::array>(indices),
            /*transpose=*/true,
            /*group_size=*/qi->group_size,
            /*bits=*/qi->bits,
            /*mode=*/"affine",
            /*sorted_indices=*/sorted_indices);
    } else {
        auto weight_t = mx::swapaxes(weight_, -1, -2);
        result = mx::gather_mm(
            x, weight_t,
            /*lhs_indices=*/std::optional<mx::array>(default_lhs_indices(x)),
            /*rhs_indices=*/std::optional<mx::array>(indices),
            sorted_indices);
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

void SwitchLinear::adopt_fused_weight(
    mx::array w, mx::array scales, std::optional<mx::array> biases,
    int group_size, int bits)
{
    weight_ = std::move(w);
    QuantizedWeightRegistry::instance().register_weight(
        &weight_, std::move(scales), std::move(biases), group_size, bits);
}

void SwitchLinear::release_weight()
{
    QuantizedWeightRegistry::instance().unregister(&weight_);
    weight_ = mx::array(0.0f);  // drop the big buffer
}

// --- SwitchGLU ---

SwitchGLU::SwitchGLU(int input_dims, int hidden_dims, int num_experts, bool bias)
    : gate_proj_(input_dims, hidden_dims, num_experts, bias),
      up_proj_(input_dims, hidden_dims, num_experts, bias),
      down_proj_(hidden_dims, input_dims, num_experts, bias),
      gate_up_proj_(input_dims, 2 * hidden_dims, num_experts, bias),
      input_dims_(input_dims),
      hidden_dims_(hidden_dims),
      num_experts_(num_experts)
{}

bool SwitchGLU::ensure_gate_up_fused() {
    if (gate_up_tried_) return gate_up_ready_;
    gate_up_tried_ = true;  // attempt exactly once
    // A/B escape hatch: MLX_NO_EXPERT_FUSION=1 keeps the two-call path.
    static const bool disabled = std::getenv("MLX_NO_EXPERT_FUSION") != nullptr;
    if (disabled) return false;

    auto& reg = QuantizedWeightRegistry::instance();
    auto* qg = reg.find(&gate_proj_.weight());
    auto* qu = reg.find(&up_proj_.weight());
    // Only fuse when both are registered-quantized with matching layout.
    // Bit-width agnostic: 4/5/6/8-bit all packed the same way, concat along the
    // output axis just stacks rows.
    if (!qg || !qu) return false;
    if (qg->group_size != qu->group_size || qg->bits != qu->bits) return false;
    // Linear (non-quant) biases are not supported on the fused path; the MoE
    // experts in these models are bias-free apart from the quant biases.
    if (gate_proj_.weight_map().count("bias") ||
        up_proj_.weight_map().count("bias"))
        return false;
    bool have_biases = qg->biases.has_value() && qu->biases.has_value();
    if (qg->biases.has_value() != qu->biases.has_value()) return false;

    // Expert weights are [E, N, K_packed]; scales/biases [E, N, K/group].
    // Concatenate gate then up along axis 1 (the output/N dim) so the fused
    // output rows are [gate(0:N) | up(N:2N)].
    auto w = mx::concatenate({gate_proj_.weight(), up_proj_.weight()}, 1);
    auto s = mx::concatenate({qg->scales, qu->scales}, 1);
    std::optional<mx::array> b;
    if (have_biases)
        b = mx::concatenate({qg->biases.value(), qu->biases.value()}, 1);
    mx::eval(w);
    mx::eval(s);
    if (b) mx::eval(*b);

    gate_up_proj_.adopt_fused_weight(std::move(w), std::move(s), std::move(b),
                                     qg->group_size, qg->bits);
    // w/s/b are materialized (eval'd above) and no longer depend on the gate/up
    // buffers, so release the originals — keeps VRAM neutral (fused == gate+up).
    gate_proj_.release_weight();
    up_proj_.release_weight();
    gate_up_ready_ = true;
    return true;
}

mx::array SwitchGLU::operator()(
    const mx::array& x,
    const mx::array& indices)
{
    // Expand dims for gather_mm: add [-2, -3]
    auto x_expanded = mx::expand_dims(mx::expand_dims(x, -2), -3);

    static const int sort_min = []() {
        const char* e = std::getenv("MLX_MOE_SORT_MIN");
        return e ? std::atoi(e) : 64;
    }();
    bool do_sort = (static_cast<int>(indices.size()) >= sort_min);

    mx::array work_x = x_expanded;
    mx::array idx = indices;
    mx::array inverse_order(0.0f);

    if (do_sort) {
        auto [sx, si, io] = gather_sort(x_expanded, indices);
        work_x = sx;
        idx = si;
        inverse_order = io;
    }

    mx::array x_gate(0.0f), x_up(0.0f);
    if (ensure_gate_up_fused()) {
        // One fused gather_qmm computes gate and up together ([.., 2*hidden]);
        // split back into the two halves. Same math, fewer/bigger launches.
        auto gu = gate_up_proj_(work_x, idx, do_sort);
        auto parts = mx::split(gu, 2, -1);
        x_gate = parts[0];
        x_up = parts[1];
    } else {
        x_up = up_proj_(work_x, idx, do_sort);
        x_gate = gate_proj_(work_x, idx, do_sort);
    }

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
