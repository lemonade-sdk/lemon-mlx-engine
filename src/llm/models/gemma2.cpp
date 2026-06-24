// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/gemma2.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, Gemma2Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.attn_logit_softcapping = j.at("attn_logit_softcapping").get<float>();
    c.final_logit_softcapping = j.at("final_logit_softcapping").get<float>();
    c.query_pre_attn_scalar = j.at("query_pre_attn_scalar").get<float>();
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Gemma2Attention ---

Gemma2Attention::Gemma2Attention(const Gemma2Configuration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      repeats_(args.num_attention_heads / args.num_key_value_heads),
      scale_(1.0f / std::pow(args.query_pre_attn_scalar, 0.5f)),
      logit_soft_cap_(args.attn_logit_softcapping),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim, args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim})),
      rope_theta_(args.rope_theta),
      rope_traditional_(args.rope_traditional)
{}

mx::array Gemma2Attention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    // Scale queries
    queries = mx::multiply(queries, mx::array(scale_));

    // Grouped-query attention with repeats
    if (repeats_ > 1) {
        queries = mx::reshape(queries, {B, num_kv_heads_, repeats_, L, head_dim_});
        keys = mx::expand_dims(keys, {2});
        values = mx::expand_dims(values, {2});
    }

    // Manual attention with logit soft-capping (compiled)
    auto scores = mx::matmul(queries, mx::swapaxes(keys, -1, -2));
    scores = logit_softcap(scores, logit_soft_cap_);

    if (mask.is_causal()) {
        // Manual attention needs the actual causal mask array
        int S = keys.shape(-2);
        auto causal_arr = mx::astype(
            create_causal_mask(L, S - L, std::nullopt), scores.dtype());
        scores = mx::add(scores, causal_arr);
    } else if (mask.has_array()) {
        scores = mx::add(scores, mask.as_array());
    }

    scores = mx::softmax(scores, -1);
    auto output = mx::matmul(scores, values);

    if (repeats_ > 1) {
        output = mx::reshape(output, {B, num_heads_, L, head_dim_});
    }

    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma2Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- Gemma2MLP ---

Gemma2MLP::Gemma2MLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Gemma2MLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    // GELU activation
    auto activation = mx::multiply(g,
        mx::multiply(mx::array(0.5f),
            mx::add(mx::array(1.0f), mx::erf(mx::divide(g, mx::array(std::sqrt(2.0f)))))));
    auto up = linear_fwd(x, up_weight_);
    return linear_fwd(mx::multiply(activation, up), down_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma2MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Gemma2TransformerBlock ---

Gemma2TransformerBlock::Gemma2TransformerBlock(const Gemma2Configuration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_(args.hidden_size, args.rms_norm_eps),
      pre_feedforward_layernorm_(args.hidden_size, args.rms_norm_eps),
      post_feedforward_layernorm_(args.hidden_size, args.rms_norm_eps),
      post_attention_layernorm_(args.hidden_size, args.rms_norm_eps)
{}

mx::array Gemma2TransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = attention_(input_layernorm_(x), mask, cache);
    auto h = mx::add(x, post_attention_layernorm_(r));
    r = mlp_(pre_feedforward_layernorm_(h));
    return mx::add(h, post_feedforward_layernorm_(r));
}

std::unordered_map<std::string, mx::array*> Gemma2TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = input_layernorm_.weight_ptr();
    map["pre_feedforward_layernorm.weight"] = pre_feedforward_layernorm_.weight_ptr();
    map["post_feedforward_layernorm.weight"] = post_feedforward_layernorm_.weight_ptr();
    map["post_attention_layernorm.weight"] = post_attention_layernorm_.weight_ptr();
    return map;
}

// --- Gemma2ModelInner ---

Gemma2ModelInner::Gemma2ModelInner(const Gemma2Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_(args.hidden_size, args.rms_norm_eps),
      hidden_scale_(std::pow(static_cast<float>(args.hidden_size), 0.5f))
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Gemma2ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    h = mx::multiply(h, mx::array(hidden_scale_));

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return norm_(h);
}

mx::array Gemma2ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Gemma2ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = norm_.weight_ptr();
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Gemma2Model ---

Gemma2Model::Gemma2Model(const Gemma2Configuration& args)
    : config_(args), model_(config_), logit_soft_cap_(args.final_logit_softcapping)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

PrepareResult Gemma2Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Gemma2Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Gemma2Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    out = model_.embed_as_linear(out);
    // Final logit soft-capping (compiled)
    out = logit_softcap(out, logit_soft_cap_);
    return out;
}

std::unordered_map<std::string, mx::array>
Gemma2Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void Gemma2Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Gemma2Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

} // namespace mlx_lm
