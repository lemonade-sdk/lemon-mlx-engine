// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// MTP Decoder Layer with MoE (SwitchGLU) MLP implementation.

#include <mlx-lm/llm/models/mtp_moe.h>
#include <mlx-lm/common/quantized_linear.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

namespace mlx_lm {

namespace {

// Quant-aware: registry hit → quantized_matmul; else dense matmul (trunk parity).
inline mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w, nullptr);
}

// C11: optional draft MoE top-k override (MLX_MTP_DRAFT_TOPK=N).
// Qwen3.6-35B MTP head ships num_experts_per_tok=8 over 256 experts — each
// draft step pays 8× SwitchGLU gathers + shared expert. Speculative draft can
// often keep high accept with fewer experts (routing shortcut). Clamped to
// [1, num_experts]. Unset → trained top_k_. Logged once when active.
inline int effective_draft_top_k(int trained_top_k, int num_experts) {
    const char* env = std::getenv("MLX_MTP_DRAFT_TOPK");
    int k = trained_top_k;
    if (env && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) k = v;
    }
    if (k < 1) k = 1;
    if (num_experts > 0 && k > num_experts) k = num_experts;
    if (k != trained_top_k) {
        static bool logged = false;
        if (!logged) {
            std::cerr << "[MTP] C11 draft MoE top_k override: trained="
                      << trained_top_k << " effective=" << k
                      << " (MLX_MTP_DRAFT_TOPK)\n";
            logged = true;
        }
    }
    return k;
}

// C13: fuse packed quant projections along out-axis (matches trunk qwen35_moe
// fuse_quant_projections). Opt-in MLX_MTP_QKV_FUSE=1 (default off — gfx1150
// measured REGRESS vs C7). Contiguous packs only.
bool fuse_quant_projections_mtp(
    const std::vector<const mx::array*>& srcs,
    std::optional<mx::array>& dst)
{
    if (dst.has_value()) return true;
    // Default OFF after C13 measure (25.45 vs C7 27.34). Opt-in only.
    if (std::getenv("MLX_MTP_QKV_FUSE") == nullptr) return false;
    if (std::getenv("MLX_MTP_NO_QKV_FUSE") != nullptr) return false;
    auto& reg = QuantizedWeightRegistry::instance();
    std::vector<const QuantizationInfo*> qis;
    qis.reserve(srcs.size());
    for (auto* w : srcs) {
        auto* q = reg.find(w);
        if (!q) return false;
        qis.push_back(q);
    }
    for (size_t i = 1; i < qis.size(); ++i) {
        if (qis[i]->group_size != qis[0]->group_size ||
            qis[i]->bits != qis[0]->bits) {
            return false;
        }
    }
    bool have_biases = true;
    for (auto* q : qis) {
        if (!q->biases) have_biases = false;
    }
    std::vector<mx::array> ws, ss, bs;
    ws.reserve(srcs.size());
    ss.reserve(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i) {
        ws.push_back(*srcs[i]);
        ss.push_back(qis[i]->scales);
        if (have_biases) bs.push_back(*qis[i]->biases);
    }
    auto concat_axis0_ok = [](const std::vector<mx::array>& arrs) -> bool {
        if (arrs.empty()) return false;
        for (size_t i = 1; i < arrs.size(); ++i) {
            if (arrs[i].ndim() != arrs[0].ndim() || arrs[i].ndim() < 1) {
                return false;
            }
            for (int d = 1; d < arrs[0].ndim(); ++d) {
                if (arrs[i].shape(d) != arrs[0].shape(d)) return false;
            }
        }
        return true;
    };
    if (!concat_axis0_ok(ws) || !concat_axis0_ok(ss)) return false;
    if (have_biases && !concat_axis0_ok(bs)) return false;
    auto w = mx::contiguous(mx::concatenate(ws, 0));
    auto s = mx::contiguous(mx::concatenate(ss, 0));
    std::optional<mx::array> b;
    if (have_biases) b = mx::contiguous(mx::concatenate(bs, 0));
    mx::eval(w);
    mx::eval(s);
    if (b) mx::eval(*b);
    dst = std::move(w);
    reg.register_weight(
        &dst.value(), std::move(s), std::move(b), qis[0]->group_size, qis[0]->bits);
    return true;
}

}  // namespace

// --- MTPDecoderLayerMoE ---

MTPDecoderLayerMoE::MTPDecoderLayerMoE(const MTPHeadConfig& args, int num_experts, int top_k)
    : args_(args),
      num_experts_(num_experts),
      top_k_(top_k),
      // q_proj outputs 2x head_dim for the sigmoid gate (matches dense MTPDecoderLayer)
      q_proj_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim() * 2, args.hidden_size})),
      k_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      v_proj_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      o_proj_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      q_norm_weight_(mx::ones({args.resolved_head_dim()})),
      k_norm_weight_(mx::ones({args.resolved_head_dim()})),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      gate_weight_(mx::zeros({num_experts, args.hidden_size})),
      switch_mlp_(args.hidden_size,
                  args.moe_intermediate_size > 0 ? args.moe_intermediate_size
                                                 : args.intermediate_size,
                  num_experts),
      shared_expert_gate_weight_(mx::zeros({1, args.hidden_size})),
      shared_expert_(args.hidden_size, args.shared_expert_intermediate_size > 0
                         ? args.shared_expert_intermediate_size
                         : args.intermediate_size,
                     /*num_experts=*/1)
{
    assert(args_.shared_expert_intermediate_size > 0 || args_.intermediate_size > 0);
}

void MTPDecoderLayerMoE::ensure_qkv_proj_fused() {
    if (qkv_proj_fused_ready_) return;
    qkv_proj_fused_ready_ = true;  // attempt once
    if (fuse_quant_projections_mtp(
            {&q_proj_weight_, &k_proj_weight_, &v_proj_weight_},
            qkv_proj_fused_weight_)) {
        static bool logged = false;
        if (!logged) {
            std::cerr << "[MTP] C13 QKV fuse ON (draft attn 3→1 matmul; "
                         "MLX_MTP_QKV_FUSE=1). Escape: MLX_MTP_NO_QKV_FUSE=1\n";
            logged = true;
        }
    }
}

mx::array MTPDecoderLayerMoE::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0);
    int L = x.shape(1);
    int H = args_.hidden_size;
    int hd = args_.resolved_head_dim();
    int n_heads = args_.num_attention_heads;
    int n_kv_heads = args_.num_key_value_heads;
    float scale = std::pow(static_cast<float>(hd), -0.5f);

    // --- self-attention sub-block (same as dense MTPDecoderLayer) ---
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, args_.rms_norm_eps);

    // C13: one fused q|k|v matmul when quant-fuse packs are available.
    ensure_qkv_proj_fused();
    const int q_out = n_heads * hd * 2;
    const int k_out = n_kv_heads * hd;
    const int v_out = n_kv_heads * hd;
    mx::array q_proj_out(0.0f), k(0.0f), v(0.0f);
    if (qkv_proj_fused_weight_.has_value()) {
        auto fused = linear_no_bias(normed, *qkv_proj_fused_weight_);
        q_proj_out = mx::slice(fused, {0, 0, 0}, {B, L, q_out});
        k = mx::slice(fused, {0, 0, q_out}, {B, L, q_out + k_out});
        v = mx::slice(fused, {0, 0, q_out + k_out}, {B, L, q_out + k_out + v_out});
    } else {
        q_proj_out = linear_no_bias(normed, q_proj_weight_);
        k = linear_no_bias(normed, k_proj_weight_);
        v = linear_no_bias(normed, v_proj_weight_);
    }
    // Reshape to [B, L, num_heads, 2*head_dim] then split into queries + gate
    auto q_proj_reshaped = mx::reshape(q_proj_out, {B, L, n_heads, -1});
    auto queries = mx::slice(q_proj_reshaped, {0, 0, 0, 0}, {B, L, n_heads, hd});
    auto q_gate = mx::slice(q_proj_reshaped, {0, 0, 0, hd}, {B, L, n_heads, 2 * hd});

    auto q4 = mx::transpose(
        mx::fast::rms_norm(queries, q_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    auto k4 = mx::reshape(k, {B, L, n_kv_heads, hd});
    k4 = mx::transpose(
        mx::fast::rms_norm(k4, k_norm_weight_, args_.rms_norm_eps), {0, 2, 1, 3});
    auto v4 = mx::reshape(v, {B, L, n_kv_heads, hd});
    v4 = mx::transpose(v4, {0, 2, 1, 3});

    int rope_dims = args_.resolved_rope_dims();
    int offset = cache ? cache->offset() : 0;
    q4 = mx::fast::rope(q4, rope_dims, /*traditional=*/false, args_.rope_theta, 1.0f, offset);
    k4 = mx::fast::rope(k4, rope_dims, /*traditional=*/false, args_.rope_theta, 1.0f, offset);

    if (cache) {
        auto [kk, vv] = cache->update(k4, v4);
        k4 = kk;
        v4 = vv;
    }

    auto attn_out = sdpa(q4, k4, v4, scale, mask);
    attn_out = mx::reshape(mx::transpose(attn_out, {0, 2, 1, 3}), {B, L, n_heads * hd});
    // Apply sigmoid gate: reshape q_gate [B, L, n_heads, hd] -> [B, L, n_heads * hd]
    // to match attention output shape, matching dense MTPDecoderLayer.
    auto gate_sigmoid = mx::sigmoid(mx::reshape(q_gate, {B, L, -1}));
    attn_out = mx::multiply(attn_out, gate_sigmoid);
    attn_out = linear_no_bias(attn_out, o_proj_weight_);

    auto h = mx::add(x, attn_out);

    // --- MoE MLP sub-block ---
    auto post = mx::fast::rms_norm(h, post_attention_layernorm_weight_, args_.rms_norm_eps);

    // Routing: compute expert gates and select top-k experts.
    // C11: MLX_MTP_DRAFT_TOPK can shrink k for cheaper draft (see effective_draft_top_k).
    const int use_top_k = effective_draft_top_k(top_k_, num_experts_);
    auto gates = mx::softmax(linear_no_bias(post, gate_weight_), -1);
    int kth = gates.shape(-1) - use_top_k;
    auto inds = mx::argpartition(gates, kth, -1);
    inds = mx::slice(inds, {0, 0, kth}, {inds.shape(0), inds.shape(1), inds.shape(2)});
    auto scores = mx::take_along_axis(gates, inds, -1);

    // Normalize scores if needed (Qwen3.5 uses norm_topk_prob=true).
    static auto compiled_normalize = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::divide(inputs[0], mx::sum(inputs[0], -1, true))};
        },
        /*shapeless=*/true);
    scores = compiled_normalize({scores})[0];

    // SwitchGLU expert dispatch.
    auto expert_out = switch_mlp_(post, inds);
    static auto compiled_combine = mx::compile(
        [](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
            return {mx::sum(mx::multiply(inputs[0], mx::expand_dims(inputs[1], -1)), -2)};
        },
        /*shapeless=*/true);
    auto combined = compiled_combine({expert_out, scores})[0];

    // Shared expert path: sigmoid(gate) * shared_output + combined.
    auto shared_gate = mx::sigmoid(linear_no_bias(post, shared_expert_gate_weight_));
    // Shared expert uses single "expert" (num_experts=1), so indices = [[0]].
    auto shared_inds = mx::full({post.shape(0), post.shape(1), 1}, 0, mx::int32);
    // SwitchGLU with indices shape [B, L, 1] returns [B, L, 1, intermediate].
    // We need 3D [B, L, intermediate] to match the main MoE output from
    // compiled_combine, so squeeze at -2.
    auto shared_raw = shared_expert_(post, shared_inds);
    auto shared_out = mx::squeeze(shared_raw, -2);
    auto mlp_out = mx::add(combined, mx::multiply(shared_gate, shared_out));

    return mx::add(h, mlp_out);
}

std::unordered_map<std::string, mx::array*> MTPDecoderLayerMoE::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["self_attn.q_proj.weight"] = &q_proj_weight_;
    map["self_attn.k_proj.weight"] = &k_proj_weight_;
    map["self_attn.v_proj.weight"] = &v_proj_weight_;
    map["self_attn.o_proj.weight"] = &o_proj_weight_;
    map["self_attn.q_norm.weight"] = &q_norm_weight_;
    map["self_attn.k_norm.weight"] = &k_norm_weight_;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["mlp.gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["mlp.switch_mlp." + k] = v;
    map["mlp.shared_expert_gate.weight"] = &shared_expert_gate_weight_;
    for (auto& [k, v] : shared_expert_.weight_map()) map["mlp.shared_expert." + k] = v;
    return map;
}

}  // namespace mlx_lm
