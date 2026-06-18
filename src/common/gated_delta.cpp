// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// Port of GatedDelta.swift -- Gated Delta Net recurrence for linear attention.
//
// Optimized: fused HIP kernel for GDN recurrence, compiled kernels, T=1 fast paths.

#include <mlx-lm/common/gated_delta.h>
#include <mlx/fast.h>

namespace mx = mlx::core;

namespace mlx_lm {

// ---------------------------------------------------------------------------
// Fused HIP kernel for gated delta recurrence.
// Grid: (32, Dv, B*Hv), ThreadGroup: (32, 4, 1)
// ---------------------------------------------------------------------------
static const char* gdn_hip_source = R"(
    auto n = blockIdx.z * blockDim.z + threadIdx.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    int T_val = T[0];
    auto q_ = q + b_idx * T_val * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T_val * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T_val * Hv * Dv + hv_idx * Dv;
    y += b_idx * T_val * Hv * Dv + hv_idx * Dv;

    auto dk_idx = threadIdx.x;  // thread_position_in_threadgroup.x
    auto dv_idx = blockIdx.y * blockDim.y + threadIdx.y;  // thread_position_in_grid.y

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * T_val * Hv;
    auto beta_ = beta + b_idx * T_val * Hv;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    // Load state into registers (float32 accumulation)
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(i_state[s_idx]);
    }

    for (int t = 0; t < T_val; ++t) {
        // Compute kv_mem = sum(state * k) via warp reduction
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
        }
        // Wave32 reduction (RDNA warp size = 32)
        for (int offset = 16; offset > 0; offset >>= 1)
            kv_mem += __shfl_xor(kv_mem, offset);

        auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                     * static_cast<float>(beta_[hv_idx]);

        // Update state and compute output = sum(state * q)
        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            out += __shfl_xor(out, offset);

        // Lane 0 writes the output
        if (threadIdx.x == 0) {
            y[dv_idx] = static_cast<InT>(out);
        }

        // Advance to next timestep
        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    // Write state back
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        o_state[s_idx] = static_cast<InT>(state[i]);
    }
)";

// Speculative-decoding variant: like gdn_hip_source but also writes the
// per-token recurrent state into `state_seq` [B, T, Hv, Dv, Dk].
static const char* gdn_seq_hip_source = R"(
    auto n = blockIdx.z * blockDim.z + threadIdx.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    int T_val = T[0];
    auto q_ = q + b_idx * T_val * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T_val * Hk * Dk + hk_idx * Dk;

    auto v_ = v + b_idx * T_val * Hv * Dv + hv_idx * Dv;
    y += b_idx * T_val * Hv * Dv + hv_idx * Dv;

    auto dk_idx = threadIdx.x;
    auto dv_idx = blockIdx.y * blockDim.y + threadIdx.y;

    auto g_ = g + b_idx * T_val * Hv;
    auto beta_ = beta + b_idx * T_val * Hv;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(i_state[s_idx]);
    }

    for (int t = 0; t < T_val; ++t) {
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * static_cast<float>(g_[hv_idx]);
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            kv_mem += __shfl_xor(kv_mem, offset);

        auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                     * static_cast<float>(beta_[hv_idx]);

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
            out += state[i] * static_cast<float>(q_[s_idx]);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            out += __shfl_xor(out, offset);

        if (threadIdx.x == 0) {
            y[dv_idx] = static_cast<InT>(out);
        }

        // Per-token state snapshot: state AFTER processing token t.
        // state_seq layout [B, T, Hv, Dv, Dk].
        auto seq_base = (((b_idx * T_val + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state_seq[seq_base + s_idx] = static_cast<InT>(state[i]);
        }

        q_ += Hk * Dk;
        k_ += Hk * Dk;
        v_ += Hv * Dv;
        y += Hv * Dv;
        g_ += Hv;
        beta_ += Hv;
    }

    for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        o_state[s_idx] = static_cast<InT>(state[i]);
    }
)";

#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
static mx::fast::CustomKernelFunction make_gdn_kernel() {
    return mx::fast::hip_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        gdn_hip_source);
}

// In-place variant: state_out (output 1) aliases state_in (input 5).
static mx::fast::CustomKernelFunction make_gdn_kernel_inplace() {
    return mx::fast::hip_kernel(
        "gated_delta_step",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out"},
        gdn_hip_source,
        /*header=*/"", /*ensure_row_contiguous=*/true, /*shared_memory=*/0,
        /*output_input_aliases=*/{{1, 5}});
}

static mx::fast::CustomKernelFunction& get_gdn_kernel_inplace() {
    static auto kernel = make_gdn_kernel_inplace();
    return kernel;
}

static mx::fast::CustomKernelFunction make_gdn_seq_kernel() {
    return mx::fast::hip_kernel(
        "gated_delta_step_seq",
        {"q", "k", "v", "g", "beta", "state_in", "T"},
        {"y", "state_out", "state_seq"},
        gdn_seq_hip_source);
}

static mx::fast::CustomKernelFunction& get_gdn_seq_kernel() {
    static auto kernel = make_gdn_seq_kernel();
    return kernel;
}

static mx::fast::CustomKernelFunction& get_gdn_kernel() {
    static auto kernel = make_gdn_kernel();
    return kernel;
}

// In-place copy of `src` into `dst`'s buffer (output 0 aliases input 0).
static mx::fast::CustomKernelFunction& get_inplace_copy_kernel() {
    static auto kernel = mx::fast::hip_kernel(
        "inplace_copy",
        {"dst", "src"},
        {"out"},
        "  unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (idx < (unsigned long)src_shape[0]) { out[idx] = src[idx]; }\n",
        /*header=*/"", /*ensure_row_contiguous=*/true, /*shared_memory=*/0,
        /*output_input_aliases=*/{{0, 0}});
    return kernel;
}

// ---------------------------------------------------------------------------
// gatedDeltaKernel — dispatch the fused HIP kernel
// ---------------------------------------------------------------------------
static std::pair<mx::array, mx::array> gated_delta_kernel(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state,
    bool inplace_state = false)
{
    int B = k.shape(0), T = k.shape(1);
    int Hk = k.shape(2), Dk = k.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);
    auto input_type = q.dtype();

    auto& kern = inplace_state ? get_gdn_kernel_inplace() : get_gdn_kernel();
    auto results = kern(
        {q, k, v, g, beta, state, mx::array(T)},
        {{B, T, Hv, Dv}, state.shape()},
        {input_type, input_type},
        {32, Dv, B * Hv},      // grid
        {32, 4, 1},             // threadGroup
        {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
        std::nullopt,           // init_value
        true,                   // ensure_row_contiguous
        {});

    return {results[0], results[1]};
}
#else
// Non-ROCm fallback.
static std::pair<mx::array, mx::array> gated_delta_kernel(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state)
{
    // TODO: implement a CPU/Metal fallback using standard MLX ops.
    throw std::runtime_error(
        "GDN HIP kernel is only available on ROCm builds. "
        "Rebuild with -DMLX_BUILD_ROCM=ON for GPU acceleration.");
}
#endif

#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
// Dispatch the fused HIP kernel, returning the per-token state stack.
static std::tuple<mx::array, mx::array, mx::array> gated_delta_kernel_seq(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state)
{
    int B = k.shape(0), T = k.shape(1);
    int Hk = k.shape(2), Dk = k.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);
    auto input_type = q.dtype();

    auto results = get_gdn_seq_kernel()(
        {q, k, v, g, beta, state, mx::array(T)},
        {{B, T, Hv, Dv}, state.shape(), {B, T, Hv, Dv, Dk}},
        {input_type, input_type, input_type},
        {32, Dv, B * Hv},      // grid
        {32, 4, 1},             // threadGroup
        {{"InT", input_type}, {"Dk", Dk}, {"Dv", Dv}, {"Hk", Hk}, {"Hv", Hv}},
        std::nullopt,           // init_value
        true,                   // ensure_row_contiguous
        {});

    return {results[0], results[1], results[2]};
}
#else
static std::tuple<mx::array, mx::array, mx::array> gated_delta_kernel_seq(
    const mx::array&, const mx::array&, const mx::array&,
    const mx::array&, const mx::array&, const mx::array&)
{
    throw std::runtime_error(
        "GDN HIP kernel is only available on ROCm builds. "
        "Rebuild with -DMLX_BUILD_ROCM=ON for GPU acceleration.");
}
#endif

// ---------------------------------------------------------------------------
// computeGatedDeltaG — compiled to fuse element-wise ops
// ---------------------------------------------------------------------------
mx::array compute_gated_delta_g(
    const mx::array& a_log, const mx::array& a, const mx::array& dt_bias)
{
    static auto compiled = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            auto a_log_f32 = mx::astype(inputs[0], mx::float32);
            auto softplus_val = mx::log(mx::add(mx::exp(mx::add(inputs[1], inputs[2])), mx::array(1.0f)));
            auto decay = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), softplus_val)));
            return {mx::astype(decay, inputs[1].dtype())};
        },
        /*shapeless=*/true);
    return compiled({a_log, a, dt_bias})[0];
}

// ---------------------------------------------------------------------------
// Compiled beta + g: fuses sigmoid(b) and compute_gated_delta_g
// ---------------------------------------------------------------------------
static auto compiled_beta_and_g = mx::compile(
    [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto beta = mx::sigmoid(inputs[0]);
        auto a_log_f32 = mx::astype(inputs[1], mx::float32);
        auto softplus_val = mx::log(mx::add(mx::exp(mx::add(inputs[2], inputs[3])), mx::array(1.0f)));
        auto g = mx::exp(mx::negative(mx::multiply(mx::exp(a_log_f32), softplus_val)));
        g = mx::astype(g, inputs[1].dtype());
        return {beta, g};
    },
    /*shapeless=*/true);

// ---------------------------------------------------------------------------
// Compiled gated delta step (fallback when HIP kernel unavailable)
// ---------------------------------------------------------------------------
static auto compiled_gated_delta_step = mx::compile(
    [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
        auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
        auto& g = inputs[3]; auto& beta = inputs[4]; auto& state = inputs[5];

        auto decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
        auto s = mx::multiply(state, decay);
        auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
        auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
        s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
        auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);
        return {y, s};
    },
    /*shapeless=*/true);

// ---------------------------------------------------------------------------
// gatedDeltaStepOps — single timestep recurrence
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_step_ops(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta, const mx::array& state,
    const std::optional<mx::array>& mask)
{
    if (!mask.has_value() && g.ndim() == 2) {
        auto results = compiled_gated_delta_step({q, k, v, g, beta, state});
        return {results[0], results[1]};
    }

    // g.ndim()==2 with mask (prefill).
    if (g.ndim() == 2 && mask.has_value()) {
        static auto compiled_step_masked_2d = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
                auto& g = inputs[3]; auto& beta = inputs[4]; auto& state = inputs[5];
                auto& mask = inputs[6];

                auto decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
                auto s = mx::multiply(state, decay);
                auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
                auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
                auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);

                auto expanded_mask = mx::expand_dims(mx::expand_dims(mx::expand_dims(mask, -1), -1), -1);
                s = mx::where(expanded_mask, s, state);
                return {y, s};
            },
            /*shapeless=*/true);
        auto results = compiled_step_masked_2d({q, k, v, g, beta, state, *mask});
        return {results[0], results[1]};
    }

    // g.ndim()==3 with mask (prefill, 3d decay).
    if (g.ndim() == 3 && mask.has_value()) {
        static auto compiled_step_masked_3d = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
                auto& g = inputs[3]; auto& beta = inputs[4]; auto& state = inputs[5];
                auto& mask = inputs[6];

                auto decay = mx::expand_dims(g, -2);
                auto s = mx::multiply(state, decay);
                auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
                auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
                auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);

                auto expanded_mask = mx::expand_dims(mask, -1);
                s = mx::where(expanded_mask, s, state);
                return {y, s};
            },
            /*shapeless=*/true);
        auto results = compiled_step_masked_3d({q, k, v, g, beta, state, *mask});
        return {results[0], results[1]};
    }

    // g.ndim()==3 without mask.
    if (g.ndim() == 3 && !mask.has_value()) {
        static auto compiled_step_3d = mx::compile(
            [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                auto& q = inputs[0]; auto& k = inputs[1]; auto& v = inputs[2];
                auto& g = inputs[3]; auto& beta = inputs[4]; auto& state = inputs[5];

                auto decay = mx::expand_dims(g, -2);
                auto s = mx::multiply(state, decay);
                auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
                auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
                s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
                auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);
                return {y, s};
            },
            /*shapeless=*/true);
        auto results = compiled_step_3d({q, k, v, g, beta, state});
        return {results[0], results[1]};
    }

    // Fallback: g.ndim()==2, no mask.
    auto old_state = state;
    mx::array decay = mx::expand_dims(mx::expand_dims(g, -1), -1);
    auto s = mx::multiply(state, decay);
    auto kv_mem = mx::sum(mx::multiply(s, mx::expand_dims(k, -2)), -1);
    auto delta = mx::multiply(mx::subtract(v, kv_mem), mx::expand_dims(beta, -1));
    s = mx::add(s, mx::multiply(mx::expand_dims(k, -2), mx::expand_dims(delta, -1)));
    auto y = mx::sum(mx::multiply(s, mx::expand_dims(q, -2)), -1);
    return {y, s};
}

// ---------------------------------------------------------------------------
// Repeat q/k from Hk heads to Hv heads
// ---------------------------------------------------------------------------
static mx::array repeat_heads(const mx::array& x, int repeat_factor) {
    if (repeat_factor <= 1) return x;
    int B = x.shape(0), T = x.shape(1), Hk = x.shape(2), D = x.shape(3);
    auto expanded = mx::reshape(x, {B, T, Hk, 1, D});
    auto tiled = mx::broadcast_to(expanded, {B, T, Hk, repeat_factor, D});
    return mx::reshape(tiled, {B, T, Hk * repeat_factor, D});
}

// ---------------------------------------------------------------------------
// gatedDeltaOps — use fused HIP kernel when possible, fallback to loop
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_ops(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask,
    bool inplace_state)
{
    int B = q.shape(0), T = q.shape(1);
    int Hk = q.shape(2), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    int repeat_factor = Hv / Hk;
    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    // Fused HIP kernel (all T, no mask) — ROCm only.
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    if (!mask.has_value() && Dk % 32 == 0) {
        auto q_work = repeat_heads(q, repeat_factor);
        auto k_work = repeat_heads(k, repeat_factor);
        return gated_delta_kernel(q_work, k_work, v, g, beta, s, inplace_state);
    }
#endif

    // Fast path: T=1 decode without mask.
    if (T == 1 && !mask.has_value()) {
        auto q_t = mx::reshape(q, {B, Hk, Dk});
        auto k_t = mx::reshape(k, {B, Hk, Dk});
        auto v_t = mx::reshape(v, {B, Hv, Dv});
        auto g_t = mx::reshape(g, {B, g.shape(2)});
        auto beta_t = mx::reshape(beta, {B, beta.shape(2)});

        if (repeat_factor > 1) {
            q_t = mx::reshape(mx::broadcast_to(
                mx::reshape(q_t, {B, Hk, 1, Dk}), {B, Hk, repeat_factor, Dk}), {B, Hv, Dk});
            k_t = mx::reshape(mx::broadcast_to(
                mx::reshape(k_t, {B, Hk, 1, Dk}), {B, Hk, repeat_factor, Dk}), {B, Hv, Dk});
        }

        auto [y, new_s] = gated_delta_step_ops(q_t, k_t, v_t, g_t, beta_t, s, std::nullopt);
        return {mx::expand_dims(y, 1), new_s};
    }

    // General path: T>1 prefill with mask.
    auto q_work = repeat_heads(q, repeat_factor);
    auto k_work = repeat_heads(k, repeat_factor);

    std::vector<mx::array> ys;
    ys.reserve(T);

    for (int t = 0; t < T; ++t) {
        auto q_t = mx::squeeze(mx::slice(q_work, {0, t, 0, 0}, {B, t + 1, q_work.shape(2), Dk}), 1);
        auto k_t = mx::squeeze(mx::slice(k_work, {0, t, 0, 0}, {B, t + 1, k_work.shape(2), Dk}), 1);
        auto v_t = mx::squeeze(mx::slice(v,      {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
        auto g_t = mx::squeeze(mx::slice(g,      {0, t, 0},    {B, t + 1, g.shape(2)}), 1);
        auto beta_t = mx::squeeze(mx::slice(beta, {0, t, 0},   {B, t + 1, beta.shape(2)}), 1);

        std::optional<mx::array> mask_t;
        if (mask.has_value()) {
            mask_t = mx::squeeze(mx::slice(*mask, {0, t}, {B, t + 1}), 1);
        }

        auto [y, new_s] = gated_delta_step_ops(q_t, k_t, v_t, g_t, beta_t, s, mask_t);
        ys.push_back(mx::expand_dims(y, 1));
        s = new_s;
    }

    return {mx::concatenate(ys, 1), s};
}

// ---------------------------------------------------------------------------
// gatedDeltaUpdate — fused beta+g, then dispatch to gatedDeltaOps
// ---------------------------------------------------------------------------
std::pair<mx::array, mx::array> gated_delta_update(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& a,
    const mx::array& b, const mx::array& a_log,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask,
    bool inplace_state)
{
    auto bg = compiled_beta_and_g({b, a_log, a, dt_bias});
    auto& beta = bg[0];
    auto& g = bg[1];

    int B = q.shape(0), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    return gated_delta_ops(q, k, v, g, beta, s, mask, inplace_state);
}

// In-place write of src into dst's device buffer (same total element count).
mx::array inplace_write(const mx::array& dst, const mx::array& src) {
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    int n = static_cast<int>(src.size());
    auto dst1 = mx::reshape(dst, {n});
    auto src1 = mx::reshape(src, {n});
    auto res = get_inplace_copy_kernel()(
        {dst1, src1}, {{n}}, {dst.dtype()},
        {n, 1, 1}, {256, 1, 1}, {}, std::nullopt, true, {});
    return mx::reshape(res[0], dst.shape());
#else
    return src;
#endif
}

// ---------------------------------------------------------------------------
// Speculative variants — also return the per-token state stack [B,T,Hv,Dv,Dk].
// ---------------------------------------------------------------------------
std::tuple<mx::array, mx::array, mx::array> gated_delta_ops_seq(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& g,
    const mx::array& beta,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask)
{
    int B = q.shape(0), T = q.shape(1);
    int Hk = q.shape(2), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    int repeat_factor = Hv / Hk;
    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    // Fused HIP kernel (no mask) — ROCm only.
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    if (!mask.has_value() && Dk % 32 == 0) {
        auto q_work = repeat_heads(q, repeat_factor);
        auto k_work = repeat_heads(k, repeat_factor);
        return gated_delta_kernel_seq(q_work, k_work, v, g, beta, s);
    }
#endif

    // Loop fallback: collect the recurrent state after each timestep.
    auto q_work = repeat_heads(q, repeat_factor);
    auto k_work = repeat_heads(k, repeat_factor);

    std::vector<mx::array> ys;
    std::vector<mx::array> states;
    ys.reserve(T);
    states.reserve(T);

    for (int t = 0; t < T; ++t) {
        auto q_t = mx::squeeze(mx::slice(q_work, {0, t, 0, 0}, {B, t + 1, q_work.shape(2), Dk}), 1);
        auto k_t = mx::squeeze(mx::slice(k_work, {0, t, 0, 0}, {B, t + 1, k_work.shape(2), Dk}), 1);
        auto v_t = mx::squeeze(mx::slice(v,      {0, t, 0, 0}, {B, t + 1, Hv, Dv}), 1);
        auto g_t = mx::squeeze(mx::slice(g,      {0, t, 0},    {B, t + 1, g.shape(2)}), 1);
        auto beta_t = mx::squeeze(mx::slice(beta, {0, t, 0},   {B, t + 1, beta.shape(2)}), 1);

        std::optional<mx::array> mask_t;
        if (mask.has_value()) {
            mask_t = mx::squeeze(mx::slice(*mask, {0, t}, {B, t + 1}), 1);
        }

        auto [y, new_s] = gated_delta_step_ops(q_t, k_t, v_t, g_t, beta_t, s, mask_t);
        ys.push_back(mx::expand_dims(y, 1));
        s = new_s;
        states.push_back(mx::expand_dims(s, 1));  // state AFTER token t -> [B,1,Hv,Dv,Dk]
    }

    return {mx::concatenate(ys, 1), s, mx::concatenate(states, 1)};
}

std::tuple<mx::array, mx::array, mx::array> gated_delta_update_seq(
    const mx::array& q, const mx::array& k,
    const mx::array& v, const mx::array& a,
    const mx::array& b, const mx::array& a_log,
    const mx::array& dt_bias,
    const std::optional<mx::array>& state,
    const std::optional<mx::array>& mask)
{
    auto bg = compiled_beta_and_g({b, a_log, a, dt_bias});
    auto& beta = bg[0];
    auto& g = bg[1];

    int B = q.shape(0), Dk = q.shape(3);
    int Hv = v.shape(2), Dv = v.shape(3);

    auto s = state.value_or(mx::zeros({B, Hv, Dv, Dk}, q.dtype()));

    return gated_delta_ops_seq(q, k, v, g, beta, s, mask);
}

} // namespace mlx_lm
