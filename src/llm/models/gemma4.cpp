// Copyright © 2026 — Gemma 4 model implementation
// Port of https://huggingface.co/google/gemma-4

#include <mlx-lm/llm/models/gemma4.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>

namespace mx = mlx::core;
namespace mlx_lm {

// ── JSON deserialization ──────────────────────────────────────────────────

void from_json(const nlohmann::json& j, Gemma4Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.head_dim = j.value("head_dim", c.hidden_size / c.num_attention_heads);
    c.global_head_dim = j.value("global_head_dim", 512);
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.final_logit_softcapping = j.value("final_logit_softcapping", 30.0f);
    c.sliding_window = j.value("sliding_window", 512);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);
    c.attention_bias = j.value("attention_bias", false);
    c.hidden_act = j.value("hidden_act", "gelu_pytorch_tanh");
    c.num_kv_shared_layers = j.value("num_kv_shared_layers", 0);
    c.use_double_wide_mlp = j.value("use_double_wide_mlp", true);

    if (j.contains("layer_types"))
        for (auto& lt : j["layer_types"])
            c.layer_types.push_back(lt.get<std::string>());

    if (j.contains("rope_parameters")) {
        auto& rp = j["rope_parameters"];
        if (rp.contains("sliding_attention") && rp["sliding_attention"].contains("rope_theta"))
            c.rope_theta_sliding = rp["sliding_attention"]["rope_theta"].get<float>();
    }
}

// ── ROPE helper matching existing patterns ────────────────────────────────

// RoPE handled by LlamaDynamicNTKScalingRoPE member

// ── Gemma4Attention ──────────────────────────────────────────────────────

Gemma4Attention::Gemma4Attention(const Gemma4Configuration& args, bool is_full)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      global_head_dim_(args.global_head_dim),
      scale_(1.0f / std::sqrt(static_cast<float>(args.head_dim))),
      sliding_scale_(1.0f / std::sqrt(static_cast<float>(args.head_dim))),
      is_full_attention_(is_full),
      wq_weight_(mx::zeros({(is_full ? args.num_attention_heads * args.head_dim + args.hidden_size
                                     : args.num_attention_heads * args.head_dim),
                             args.hidden_size})),
      wk_weight_(mx::zeros({(is_full ? 2 * args.num_key_value_heads * args.head_dim
                                     : args.num_key_value_heads * args.head_dim),
                             args.hidden_size})),
      wv_weight_(mx::zeros({(is_full ? 2 * args.num_key_value_heads * args.head_dim
                                     : args.num_key_value_heads * args.head_dim),
                             args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size,
                            (is_full ? args.num_attention_heads * args.head_dim + args.hidden_size
                                     : args.num_attention_heads * args.head_dim)})),
      q_norm_weight_(mx::ones({args.head_dim})),
      k_norm_weight_(mx::ones({args.head_dim})),
      rms_norm_eps_(args.rms_norm_eps),
      rope_theta_(is_full ? args.rope_theta : args.rope_theta_sliding),
      sliding_window_(args.sliding_window),
      rope_(args.head_dim, std::nullopt, false, is_full ? args.rope_theta : args.rope_theta_sliding, 1.0f, "default", std::nullopt)
{
    if (args.attention_bias) {
        wq_bias_ = mx::zeros({args.num_attention_heads * args.head_dim});
        wk_bias_ = mx::zeros({args.num_key_value_heads * args.head_dim});
        wv_bias_ = mx::zeros({args.num_key_value_heads * args.head_dim});
        wo_bias_ = mx::zeros({args.hidden_size});
    }
}

mx::array Gemma4Attention::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);
    int kv_h = num_kv_heads_, hd = head_dim_;

    auto q_all = linear_forward(x, wq_weight_, wq_bias_ ? &*wq_bias_ : nullptr);
    auto k_all = linear_forward(x, wk_weight_, wk_bias_ ? &*wk_bias_ : nullptr);
    auto v_all = linear_forward(x, wv_weight_, wv_bias_ ? &*wv_bias_ : nullptr);

    int q_dim = q_all.shape(-1);
    int k_dim = k_all.shape(-1);
    bool is_full = (is_full_attention_ && q_dim > num_heads_ * hd);
    int norm_dim = is_full ? global_head_dim_ : hd;
    int g_heads = is_full ? (q_dim / norm_dim) : 0;

    mx::array q(0.0f), k(0.0f), v(0.0f), q_global(0.0f);
    if (is_full) {
        // Full attention: slice to regular head dims for SDPA
        // Q/K norms from checkpoint are [global_head_dim]; slice first hd for regular
        mx::eval(q_all); mx::eval(k_all); mx::eval(v_all);
        int regular_qd = num_heads_ * hd;
        int regular_kd = kv_h * hd;
        q = mx::transpose(mx::reshape(mx::slice(q_all, {0,0,0}, {B,L,regular_qd}),
                                       {B, L, num_heads_, hd}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(mx::slice(k_all, {0,0,0}, {B,L,regular_kd}),
                                       {B, L, kv_h, hd}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(mx::slice(v_all, {0,0,0}, {B,L,regular_kd}),
                                       {B, L, kv_h, hd}), {0, 2, 1, 3});
        // Use first hd elements of norms (checkpoint norms are [global_head_dim])
        auto qn = mx::slice(q_norm_weight_, {0}, {hd});
        auto kn = mx::slice(k_norm_weight_, {0}, {hd});
        q = mx::fast::rms_norm(q, qn, rms_norm_eps_);
        k = mx::fast::rms_norm(k, kn, rms_norm_eps_);
    } else {
        q = mx::transpose(mx::reshape(q_all, {B, L, num_heads_, hd}), {0, 2, 1, 3});
        k = mx::transpose(mx::reshape(k_all, {B, L, kv_h, hd}), {0, 2, 1, 3});
        v = mx::transpose(mx::reshape(v_all, {B, L, kv_h, hd}), {0, 2, 1, 3});
        q = mx::fast::rms_norm(q, q_norm_weight_, rms_norm_eps_);
        k = mx::fast::rms_norm(k, k_norm_weight_, rms_norm_eps_);
    }

    // RoPE (regular portion only)
    int offset = cache ? cache->offset() : 0;
    q = rope_(q, offset);
    k = rope_(k, offset);

    // KV cache
    if (cache) {
        auto [ck, cv] = cache->update(k, v);
        k = ck; v = cv;
    }

    auto out = sdpa(q, k, v, scale_, mask);
    out = mx::reshape(mx::transpose(out, {0, 2, 1, 3}), {B, L, -1});

    // For full_attention layers with larger o_proj, pad to expected input dim.
    // The o_proj expects 4096 input features (regular + global). We only have
    // regular (2048), so pad with zeros for the global portion.
    if (is_full && out.shape(-1) == num_heads_ * hd) {
        mx::eval(out);
        int target = num_heads_ * hd + q_dim - num_heads_ * hd; // = q_dim = 4096
        int pad = target - out.shape(-1);
        if (pad > 0) {
            auto zeros = mx::zeros({B, L, pad}, out.dtype());
            out = mx::concatenate({out, zeros}, -1);
        }
    }
    out = linear_forward(out, wo_weight_, wo_bias_ ? &*wo_bias_ : nullptr);
    return out;
}

std::unordered_map<std::string, mx::array*> Gemma4Attention::weight_map() {
    auto m = std::unordered_map<std::string, mx::array*>{
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
    if (wq_bias_) {
        m["q_proj.bias"] = &*wq_bias_;
        m["k_proj.bias"] = &*wk_bias_;
        m["v_proj.bias"] = &*wv_bias_;
        m["o_proj.bias"] = &*wo_bias_;
    }
    // Full attention: q_proj and o_proj are larger (regular + global),
    // handled via slicing in operator(). No separate weight entries.
    return m;
}

// ── Gemma4MLP ─────────────────────────────────────────────────────────────

Gemma4MLP::Gemma4MLP(const Gemma4Configuration& args)
    : gate_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      down_weight_(mx::zeros({args.hidden_size, args.intermediate_size})),
      up_weight_(mx::zeros({args.intermediate_size, args.hidden_size}))
{
    if (args.attention_bias) {
        gate_bias_ = mx::zeros({args.intermediate_size});
        down_bias_ = mx::zeros({args.hidden_size});
        up_bias_ = mx::zeros({args.intermediate_size});
    }
}

mx::array Gemma4MLP::operator()(const mx::array& x) {
    auto gate = linear_forward(x, gate_weight_, gate_bias_ ? &*gate_bias_ : nullptr);
    auto up = linear_forward(x, up_weight_, up_bias_ ? &*up_bias_ : nullptr);
    return linear_forward(mx::multiply(gelu_tanh(gate), up), down_weight_, down_bias_ ? &*down_bias_ : nullptr);
}

std::unordered_map<std::string, mx::array*> Gemma4MLP::weight_map() {
    auto m = std::unordered_map<std::string, mx::array*>{
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
    if (gate_bias_) {
        m["gate_proj.bias"] = &*gate_bias_;
        m["down_proj.bias"] = &*down_bias_;
        m["up_proj.bias"] = &*up_bias_;
    }
    return m;
}

// ── Gemma4TransformerBlock ────────────────────────────────────────────────

Gemma4TransformerBlock::Gemma4TransformerBlock(const Gemma4Configuration& args, int layer_idx)
    : attention_(args, layer_idx < (int)args.layer_types.size() &&
                        args.layer_types[layer_idx] == "full_attention"),
      mlp_(args),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      pre_feedforward_layernorm_weight_(mx::ones({args.hidden_size})),
      post_feedforward_layernorm_weight_(mx::ones({args.hidden_size})),
      per_layer_input_gate_weight_(mx::zeros({args.hidden_size})),
      per_layer_projection_weight_(mx::zeros({args.hidden_size, args.hidden_size})),
      post_per_layer_input_norm_weight_(mx::ones({args.hidden_size})),
      layer_scalar_(mx::ones({1})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Gemma4TransformerBlock::operator()(
    const mx::array& x, const AttentionMask& mask, KVCache* cache)
{
    // Per-layer input: compress to 256 dims, expand back, residual
    auto gated = linear_forward(x, per_layer_input_gate_weight_);
    auto projected = linear_forward(gated, per_layer_projection_weight_);
    auto h = mx::add(x, mx::fast::rms_norm(projected, post_per_layer_input_norm_weight_, rms_norm_eps_));

    // Self-attention
    auto attn_out = attention_(mx::fast::rms_norm(h, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    h = mx::add(h, attn_out);
    h = mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_);

    // FFN with extra norms
    auto ffn_out = mlp_(mx::fast::rms_norm(h, pre_feedforward_layernorm_weight_, rms_norm_eps_));
    h = mx::add(h, ffn_out);
    h = mx::fast::rms_norm(h, post_feedforward_layernorm_weight_, rms_norm_eps_);

    return h;
}

std::unordered_map<std::string, mx::array*> Gemma4TransformerBlock::weight_map() {
    auto m = std::unordered_map<std::string, mx::array*>{};
    for (auto& [k, v] : attention_.weight_map()) m["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) m["mlp." + k] = v;
    m["input_layernorm.weight"] = &input_layernorm_weight_;
    m["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    m["pre_feedforward_layernorm.weight"] = &pre_feedforward_layernorm_weight_;
    m["post_feedforward_layernorm.weight"] = &post_feedforward_layernorm_weight_;
    m["per_layer_input_gate.weight"] = &per_layer_input_gate_weight_;
    m["per_layer_projection.weight"] = &per_layer_projection_weight_;
    m["post_per_layer_input_norm.weight"] = &post_per_layer_input_norm_weight_;
    m["layer_scalar"] = &layer_scalar_;
    return m;
}

// ── Gemma4ModelInner ──────────────────────────────────────────────────────

Gemma4ModelInner::Gemma4ModelInner(const Gemma4Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      embed_tokens_per_layer_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      hidden_size_(args.hidden_size)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args, i);
}

mx::array Gemma4ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Gemma4ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma4ModelInner::weight_map() {
    auto m = std::unordered_map<std::string, mx::array*>{
        {"embed_tokens.weight", &embed_tokens_weight_},
        {"embed_tokens_per_layer.weight", &embed_tokens_per_layer_weight_},
        {"norm.weight", &norm_weight_},
    };
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) m[prefix + k] = v;
    }
    return m;
}

// ── Gemma4Model ──────────────────────────────────────────────────────────

Gemma4Model::Gemma4Model(const Gemma4Configuration& args)
    : config_(args),
      model_(config_),
      per_layer_model_projection_weight_(mx::zeros({args.num_hidden_layers * 256, args.hidden_size})),
      per_layer_projection_norm_weight_(mx::ones({256}))
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings)
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
}

PrepareResult Gemma4Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Gemma4Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Gemma4Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value())
        return linear_forward(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Gemma4Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");
    std::vector<std::string> to_remove;
    for (auto& [k, _] : weights) {
        if (k.find("__metadata__") != std::string::npos || k.find("inv_freq") != std::string::npos)
            to_remove.push_back(k);
    }
    for (auto& k : to_remove) weights.erase(k);
    return weights;
}

void Gemma4Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Gemma4Model::weight_map() {
    auto m = std::unordered_map<std::string, mx::array*>{};
    for (auto& [k, v] : model_.weight_map()) m["model." + k] = v;
    if (lm_head_weight_.has_value()) m["lm_head.weight"] = &lm_head_weight_.value();
    m["model.per_layer_model_projection.weight"] = &per_layer_model_projection_weight_;
    m["model.per_layer_projection_norm.weight"] = &per_layer_projection_norm_weight_;
    return m;
}

} // namespace mlx_lm
