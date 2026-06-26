// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/phi.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, PhiConfiguration& c) {
    c.max_position_embeddings = j.value("max_position_embeddings", 2048);
    c.vocab_size = j.value("vocab_size", 51200);
    c.hidden_size = j.value("hidden_size", 2560);
    c.num_attention_heads = j.value("num_attention_heads", 32);
    c.num_hidden_layers = j.value("num_hidden_layers", 32);
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.partial_rotary_factor = j.value("partial_rotary_factor", 0.4f);
    c.intermediate_size = j.value("intermediate_size", 10240);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-5f);
    c.rope_theta = j.value("rope_theta", 10000.0f);
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

// --- PhiAttention ---

PhiAttention::PhiAttention(const PhiConfiguration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.hidden_size / args.num_attention_heads),
      rope_dim_(static_cast<int>(args.partial_rotary_factor * (args.hidden_size / args.num_attention_heads))),
      wq_weight_(mx::zeros({args.num_attention_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wq_bias_(mx::zeros({args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      wk_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wk_bias_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads)})),
      wv_weight_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads), args.hidden_size})),
      wv_bias_(mx::zeros({args.num_key_value_heads * (args.hidden_size / args.num_attention_heads)})),
      dense_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * (args.hidden_size / args.num_attention_heads)})),
      dense_bias_(mx::zeros({args.hidden_size})),
      rope_theta_(args.rope_theta)
{}

mx::array PhiAttention::operator()(const mx::array& x,
                                     const AttentionMask& mask,
                                     KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values = linear_fwd(x, wv_weight_, &wv_bias_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, rope_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, rope_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    float scale = std::sqrt(1.0f / static_cast<float>(head_dim_));
    // Cast queries to float32 for attention stability (like Swift code)
    auto q_f32 = mx::astype(queries, mx::float32);
    auto output = sdpa(q_f32, keys, values, scale, mask);
    output = mx::astype(output, values.dtype());
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, dense_weight_, &dense_bias_);
}

std::unordered_map<std::string, mx::array*> PhiAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"dense.weight", &dense_weight_}, {"dense.bias", &dense_bias_},
    };
}

// --- PhiMLP ---

PhiMLP::PhiMLP(const PhiConfiguration& config)
    : fc1_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      fc1_bias_(mx::zeros({config.intermediate_size})),
      fc2_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      fc2_bias_(mx::zeros({config.hidden_size}))
{}

mx::array PhiMLP::operator()(const mx::array& x) {
    return linear_fwd(gelu(linear_fwd(x, fc1_weight_, &fc1_bias_)), fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> PhiMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// --- PhiDecoderLayer ---
// Phi uses parallel attention + MLP from the same normed input: out = attn(h) + mlp(h) + x

PhiDecoderLayer::PhiDecoderLayer(const PhiConfiguration& config)
    : attention_(config), mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      input_layernorm_bias_(mx::zeros({config.hidden_size})),
      norm_eps_(config.layer_norm_eps)
{}

mx::array PhiDecoderLayer::operator()(const mx::array& x,
                                        const AttentionMask& mask,
                                        KVCache* cache) {
    auto h = mx::fast::layer_norm(x, input_layernorm_weight_, input_layernorm_bias_, norm_eps_);
    auto attn_out = attention_(h, mask, cache);
    auto ff_out = mlp_(h);
    return mx::add(mx::add(attn_out, ff_out), x);
}

std::unordered_map<std::string, mx::array*> PhiDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["input_layernorm.bias"] = &input_layernorm_bias_;
    return map;
}

// --- PhiModelInner ---

PhiModelInner::PhiModelInner(const PhiConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      final_layernorm_weight_(mx::ones({args.hidden_size})),
      final_layernorm_bias_(mx::zeros({args.hidden_size})),
      norm_eps_(args.layer_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array PhiModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::layer_norm(h, final_layernorm_weight_, final_layernorm_bias_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> PhiModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["final_layernorm.weight"] = &final_layernorm_weight_;
    map["final_layernorm.bias"] = &final_layernorm_bias_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- PhiModel ---

PhiModel::PhiModel(const PhiConfiguration& args)
    : config_(args), model_(config_),
      lm_head_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      lm_head_bias_(mx::zeros({args.vocab_size}))
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

PrepareResult PhiModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput PhiModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array PhiModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return linear_fwd(out, lm_head_weight_, &lm_head_bias_);
}

std::unordered_map<std::string, mx::array>
PhiModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    return weights;
}

void PhiModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> PhiModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    map["lm_head.bias"] = &lm_head_bias_;
    return map;
}

} // namespace mlx_lm
