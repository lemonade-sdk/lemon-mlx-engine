// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++
// MTP Decoder Layer with MoE (SwitchGLU) MLP implementation.

#include <mlx-lm/llm/models/mtp_moe.h>

#include <cassert>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

namespace {

mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    return mx::matmul(x, mx::transpose(w));
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
      switch_mlp_(args.hidden_size, args.intermediate_size, num_experts),
      shared_expert_gate_weight_(mx::zeros({1, args.hidden_size})),
      shared_expert_(args.hidden_size, args.shared_expert_intermediate_size > 0
                         ? args.shared_expert_intermediate_size
                         : args.intermediate_size,
                     /*num_experts=*/1)
{
    assert(args_.shared_expert_intermediate_size > 0 || args_.intermediate_size > 0);
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
    auto q_proj_out = linear_no_bias(normed, q_proj_weight_);
    // Reshape to [B, L, num_heads, 2*head_dim] then split into queries + gate
    auto q_proj_reshaped = mx::reshape(q_proj_out, {B, L, n_heads, -1});
    auto queries = mx::slice(q_proj_reshaped, {0, 0, 0, 0}, {B, L, n_heads, hd});
    auto q_gate = mx::slice(q_proj_reshaped, {0, 0, 0, hd}, {B, L, n_heads, 2 * hd});

    auto k = linear_no_bias(normed, k_proj_weight_);
    auto v = linear_no_bias(normed, v_proj_weight_);

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
    auto gates = mx::softmax(linear_no_bias(post, gate_weight_), -1);
    int kth = gates.shape(-1) - top_k_;
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
