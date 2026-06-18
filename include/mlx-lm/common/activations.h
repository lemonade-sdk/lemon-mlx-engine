// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Compiled activation functions for kernel fusion.
#pragma once

#include <mlx/mlx.h>

namespace mlx_lm {

// Compiled SiLU (Sigmoid Linear Unit): x * sigmoid(x)
// Matches Python's @mx.compile(shapeless=True) on silu.
inline mlx::core::array silu(const mlx::core::array& x) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            return {mlx::core::multiply(inputs[0], mlx::core::sigmoid(inputs[0]))};
        },
        /*shapeless=*/true);
    return compiled({x})[0];
}

// Compiled SwiGLU: silu(gate) * up — fuses sigmoid + 2 multiplies into one kernel.
// Matches Python's @mx.compile(shapeless=True) on swiglu.
inline mlx::core::array swiglu(const mlx::core::array& gate, const mlx::core::array& up) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            auto silu_gate = mlx::core::multiply(inputs[0], mlx::core::sigmoid(inputs[0]));
            return {mlx::core::multiply(silu_gate, inputs[1])};
        },
        /*shapeless=*/true);
    return compiled({gate, up})[0];
}

// Compiled logit softcap: cap * tanh(x / cap)
// Matches Python's @partial(mx.compile, shapeless=True) on logit_softcap.
// Used by Gemma2, Gemma3n, Nanochat.
inline mlx::core::array logit_softcap(const mlx::core::array& x, float cap) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            return {mlx::core::multiply(inputs[1],
                    mlx::core::tanh(mlx::core::divide(inputs[0], inputs[1])))};
        },
        /*shapeless=*/true);
    return compiled({x, mlx::core::array(cap)})[0];
}

// Compiled squared ReLU: relu(x)^2
// Matches Python's @partial(mx.compile, shapeless=True) on relu_squared.
// Used by Nemotron.
inline mlx::core::array relu_squared(const mlx::core::array& x) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            auto r = mlx::core::maximum(inputs[0], mlx::core::array(0.0f));
            return {mlx::core::multiply(r, r)};
        },
        /*shapeless=*/true);
    return compiled({x})[0];
}

// Compiled GELU with tanh approximation (gelu_pytorch_tanh).
// Matches Python: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Used by Qwen3.5 vision encoder.
inline mlx::core::array gelu_tanh(const mlx::core::array& x) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            auto x3 = mlx::core::multiply(inputs[0], mlx::core::multiply(inputs[0], inputs[0]));
            auto coeff = mlx::core::array(0.7978845608028654f); // sqrt(2/pi)
            auto cubic_term = mlx::core::array(0.044715f);
            // sqrt(2/pi) * (x + 0.044715 * x^3)
            auto scaled = mlx::core::multiply(coeff, mlx::core::add(inputs[0], mlx::core::multiply(cubic_term, x3)));
            // 0.5 * x * (1 + tanh(scaled))
            auto tanh_val = mlx::core::tanh(scaled);
            return {mlx::core::multiply(mlx::core::multiply(mlx::core::array(0.5f), inputs[0]),
                                        mlx::core::add(mlx::core::array(1.0f), tanh_val))};
        },
        /*shapeless=*/true);
    return compiled({x})[0];
}

// Compiled residual clipping for float16 safety: x + cast(y, x.dtype)
// Matches Python's @partial(mx.compile, shapeless=True) on clip_residual (Gemma3).
inline mlx::core::array clip_residual(const mlx::core::array& x, const mlx::core::array& y) {
    static auto compiled = mlx::core::compile(
        [](const std::vector<mlx::core::array>& inputs) -> std::vector<mlx::core::array> {
            return {mlx::core::add(inputs[0], mlx::core::astype(inputs[1], inputs[0].dtype()))};
        },
        /*shapeless=*/true);
    return compiled({x, y})[0];
}

} // namespace mlx_lm
