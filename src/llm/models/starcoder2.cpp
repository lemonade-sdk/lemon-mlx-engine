// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/starcoder2.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, Starcoder2Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.norm_epsilon = j.value("norm_epsilon", 1e-5f);
    c.vocab_size = j.value("vocab_size", 49152);
    c.rope_theta = j.value("rope_theta", 100000.0f);
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// GELU activation
static mx::array gelu(const mx::array& x) {
    return mx::multiply(x,
        mx::multiply(mx::array(0.5f),
            mx::add(mx::array(1.0f), mx::erf(mx::divide(x, mx::array(std::sqrt(2.0f)))))));
}

// --- Starcoder2Attention ---

Starcoder2Attention::Starcoder2Attention(const Starcoder2Configuration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.hidden_size / args.num_attention_heads),
      scale_(std::pow(static_cast<float>(args.hidden_size / args.num_attention_heads), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wq_bias_(mx::zeros({args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      wk_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wk_bias_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads)})),
      wv_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wv_bias_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads)})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      wo_bias_(mx::zeros({args.hidden_size})),
      rope_theta_(args.rope_theta)
{}

mx::array Starcoder2Attention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values = linear_fwd(x, wv_weight_, &wv_bias_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> Starcoder2Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"o_proj.weight", &wo_weight_}, {"o_proj.bias", &wo_bias_},
    };
}

// --- Starcoder2MLP ---

Starcoder2MLP::Starcoder2MLP(int dimensions, int hidden_dimensions)
    : c_fc_weight_(mx::zeros({hidden_dimensions, dimensions})),
      c_fc_bias_(mx::zeros({hidden_dimensions})),
      c_proj_weight_(mx::zeros({dimensions, hidden_dimensions})),
      c_proj_bias_(mx::zeros({dimensions}))
{}

mx::array Starcoder2MLP::operator()(const mx::array& x) {
    return linear_fwd(gelu(linear_fwd(x, c_fc_weight_, &c_fc_bias_)), c_proj_weight_, &c_proj_bias_);
}

std::unordered_map<std::string, mx::array*> Starcoder2MLP::weight_map() {
    return {
        {"c_fc.weight", &c_fc_weight_}, {"c_fc.bias", &c_fc_bias_},
        {"c_proj.weight", &c_proj_weight_}, {"c_proj.bias", &c_proj_bias_},
    };
}

// --- Starcoder2TransformerBlock ---

Starcoder2TransformerBlock::Starcoder2TransformerBlock(const Starcoder2Configuration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      input_layernorm_bias_(mx::zeros({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_bias_(mx::zeros({args.hidden_size})),
      norm_eps_(args.norm_epsilon)
{}

mx::array Starcoder2TransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = attention_(mx::fast::layer_norm(x, input_layernorm_weight_, input_layernorm_bias_, norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::layer_norm(h, post_attention_layernorm_weight_, post_attention_layernorm_bias_, norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Starcoder2TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["input_layernorm.bias"] = &input_layernorm_bias_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    map["post_attention_layernorm.bias"] = &post_attention_layernorm_bias_;
    return map;
}

// --- Starcoder2ModelInner ---

Starcoder2ModelInner::Starcoder2ModelInner(const Starcoder2Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      norm_bias_(mx::zeros({args.hidden_size})),
      norm_eps_(args.norm_epsilon)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Starcoder2ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::layer_norm(h, norm_weight_, norm_bias_, norm_eps_);
}

mx::array Starcoder2ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Starcoder2ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    map["norm.bias"] = &norm_bias_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Starcoder2Model ---

Starcoder2Model::Starcoder2Model(const Starcoder2Configuration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Starcoder2Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Starcoder2Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Starcoder2Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    }
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Starcoder2Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void Starcoder2Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Starcoder2Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
