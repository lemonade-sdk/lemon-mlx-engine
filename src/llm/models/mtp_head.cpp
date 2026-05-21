// Copyright (c) 2024-2026 Apple Inc. -- Ported to C++
// MTP head + decoder layer -- I7 sub-task 2 (scaffolding).
//
// Reference: mlx-lm-private qwen35_mtp branch, mlx_lm/models/qwen3_5.py
//   MTPDecoderLayer (lines 310-333): self_attn + MLP or SparseMoE + RMSNorms
//   MTPHead         (lines 336-360): pre_fc norms + Linear(2H -> H) + 1 layer

#include <mlx-lm/llm/models/mtp_head.h>

namespace mx = mlx::core;

namespace mlx_lm {

namespace {
mx::array linear_no_bias(const mx::array& x, const mx::array& w) {
    // Matches the linear_fwd helper used elsewhere when there is no bias.
    return mx::matmul(x, mx::transpose(w));
}
}  // namespace

// --- MTPDecoderLayer ---

MTPDecoderLayer::MTPDecoderLayer(const Qwen35Configuration& args, bool use_moe)
    : use_moe_(use_moe),
      self_attn_(args),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps) {
    if (use_moe_) {
        moe_mlp_.emplace(args);
    } else {
        dense_mlp_.emplace(args.hidden_size, args.intermediate_size);
    }
}

mx::array MTPDecoderLayer::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_);
    auto r = self_attn_(normed, mask, cache);
    auto h = mx::add(x, r);
    auto post = mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_);
    mx::array mlp_out = use_moe_ ? (*moe_mlp_)(post) : (*dense_mlp_)(post);
    return mx::add(h, mlp_out);
}

std::unordered_map<std::string, mx::array*> MTPDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) {
        map["self_attn." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    if (use_moe_) {
        for (auto& [k, v] : moe_mlp_->weight_map()) {
            map["mlp." + k] = v;
        }
    } else {
        for (auto& [k, v] : dense_mlp_->weight_map()) {
            map["mlp." + k] = v;
        }
    }
    return map;
}

// --- MTPHead ---

MTPHead::MTPHead(const Qwen35Configuration& args)
    : pre_fc_norm_hidden_weight_(mx::ones({args.hidden_size})),
      pre_fc_norm_embedding_weight_(mx::ones({args.hidden_size})),
      fc_weight_(mx::zeros({args.hidden_size, 2 * args.hidden_size})),
      layer_(args, /*use_moe=*/args.num_experts > 0),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps) {}

mx::array MTPHead::operator()(
    const mx::array& hidden_state,
    const mx::array& token_embedding,
    const AttentionMask& mask,
    KVCache* cache) {
    auto h_norm = mx::fast::rms_norm(
        hidden_state, pre_fc_norm_hidden_weight_, rms_norm_eps_);
    auto e_norm = mx::fast::rms_norm(
        token_embedding, pre_fc_norm_embedding_weight_, rms_norm_eps_);
    // Note: qwen3_5.py:357 concatenates [e_norm, h_norm] (embedding first).
    auto cat = mx::concatenate({e_norm, h_norm}, -1);
    auto h = linear_no_bias(cat, fc_weight_);
    return layer_(h, mask, cache);
}

mx::array MTPHead::apply_output_norm(const mx::array& h) const {
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

std::unordered_map<std::string, mx::array*> MTPHead::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["pre_fc_norm_hidden.weight"] = &pre_fc_norm_hidden_weight_;
    map["pre_fc_norm_embedding.weight"] = &pre_fc_norm_embedding_weight_;
    map["fc.weight"] = &fc_weight_;
    for (auto& [k, v] : layer_.weight_map()) {
        map["layers.0." + k] = v;
    }
    map["norm.weight"] = &norm_weight_;
    return map;
}

void MTPHead::load_mtp_weights(
    const std::unordered_map<std::string, mx::array>& mtp_weights) {
    // Keys come in as e.g. "model.mtp.fc.weight" or "mtp.layers.0...". We
    // accept both shapes by stripping any prefix up to and including the
    // "mtp." segment. This is best-effort scaffolding -- the bit-exact key
    // schema is finalised once we have a real safetensors to test against.
    auto wmap = weight_map();
    for (const auto& [raw_key, value] : mtp_weights) {
        std::string key = raw_key;
        auto pos = key.find("mtp.");
        if (pos != std::string::npos) {
            key = key.substr(pos + 4);  // strip up to and including "mtp."
        }
        auto it = wmap.find(key);
        if (it != wmap.end()) {
            *it->second = value;
        }
    }
}

}  // namespace mlx_lm
