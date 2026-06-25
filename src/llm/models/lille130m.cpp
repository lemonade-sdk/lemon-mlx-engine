// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Lille130m.swift

#include <mlx-lm/llm/models/lille130m.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, Lille130mConfiguration& c) {
    c.block_size = j.at("block_size").get<int>();
    c.layer_norm_eps = j.at("layer_norm_eps").get<float>();
    c.hidden_size = j.at("n_embd").get<int>();
    c.num_attention_heads = j.at("n_head").get<int>();
    c.num_key_value_heads = j.at("n_kv_heads").get<int>();
    c.num_hidden_layers = j.at("n_layer").get<int>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", true);

    // Read quantization parameters if present
    if (j.contains("quantization")) {
        const auto& q = j["quantization"];
        c.quant_bits = q.value("bits", 0);
        c.quant_group_size = q.value("group_size", 0);
    }
}

// --- Helpers ---

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Lille130mAttention ---

Lille130mAttention::Lille130mAttention(const Lille130mConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(config.resolved_head_dim()), -0.5f)),
      qkv_proj_weight_(mx::zeros({(config.num_attention_heads + 2 * config.num_key_value_heads) * config.resolved_head_dim(), config.hidden_size})),
      out_proj_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.resolved_head_dim()})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.layer_norm_eps),
      rope_theta_(config.rope_theta)
{}

mx::array Lille130mAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    // Apply attention norm to input first
    auto normed = mx::fast::rms_norm(x, norm_weight_, norm_eps_);

    // Combined QKV projection
    auto qkv = linear_fwd(normed, qkv_proj_weight_);

    // Split into Q, K, V parts
    int q_size = num_heads_ * head_dim_;
    int kv_size = num_kv_heads_ * head_dim_;

    auto queries = mx::slice(qkv, {0, 0, 0}, {B, L, q_size});
    auto keys = mx::slice(qkv, {0, 0, q_size}, {B, L, q_size + kv_size});
    auto values = mx::slice(qkv, {0, 0, q_size + kv_size}, {B, L, q_size + 2 * kv_size});

    // Reshape and transpose: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    // Apply RoPE (traditional=true for Lille130m)
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, true, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, true, rope_theta_, 1.0f, offset);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    // Scaled dot-product attention
    auto output = sdpa(
        queries, keys, values, scale_, mask);

    // Reshape back: [B, heads, L, head_dim] -> [B, L, heads*head_dim]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    return linear_fwd(output, out_proj_weight_);
}

std::unordered_map<std::string, mx::array*> Lille130mAttention::weight_map() {
    return {
        {"norm.weight", &norm_weight_},
        {"qkv_proj.weight", &qkv_proj_weight_},
        {"out_proj.weight", &out_proj_weight_},
    };
}

// --- Lille130mMLP ---

static int lille_hidden_dim(int hidden_size) {
    int numerator = (8 * hidden_size) / 3;
    int rounded = static_cast<int>(std::round(static_cast<float>(numerator) / 256.0f));
    return std::max(256 * rounded, 1);
}

Lille130mMLP::Lille130mMLP(const Lille130mConfiguration& config)
    : norm_weight_(mx::ones({config.hidden_size})),
      gate_weight_(mx::zeros({lille_hidden_dim(config.hidden_size), config.hidden_size})),
      up_weight_(mx::zeros({lille_hidden_dim(config.hidden_size), config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, lille_hidden_dim(config.hidden_size)})),
      norm_eps_(config.layer_norm_eps)
{}

mx::array Lille130mMLP::operator()(const mx::array& x) {
    // Internal norm first
    auto h = mx::fast::rms_norm(x, norm_weight_, norm_eps_);

    // SwiGLU: down(swiglu(gate(h), up(h)))
    return linear_fwd(swiglu(linear_fwd(h, gate_weight_), linear_fwd(h, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Lille130mMLP::weight_map() {
    return {
        {"norm.weight", &norm_weight_},
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- Lille130mBlock ---

Lille130mBlock::Lille130mBlock(const Lille130mConfiguration& config)
    : attention_(config),
      feed_forward_(config)
{}

mx::array Lille130mBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // No block-level norms — norms are inside attention and MLP
    auto h = mx::add(x, attention_(x, mask, cache));
    return mx::add(h, feed_forward_(h));
}

std::unordered_map<std::string, mx::array*> Lille130mBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["attention." + k] = v;
    for (auto& [k, v] : feed_forward_.weight_map()) map["feed_forward." + k] = v;
    return map;
}

// --- Lille130mModelInner ---

Lille130mModelInner::Lille130mModelInner(const Lille130mConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.layer_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array Lille130mModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto tokens = inputs;
    if (tokens.ndim() < 2) {
        tokens = mx::reshape(tokens, {1, static_cast<int>(tokens.size())});
    }

    // Embedding lookup — no scaling
    auto h = mx::take(embed_tokens_weight_, tokens, 0);

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    // Final RMS norm
    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array Lille130mModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> Lille130mModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["tok_embeddings.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }
    return map;
}

// --- Lille130mModel ---

Lille130mModel::Lille130mModel(const Lille130mConfiguration& config)
    : config_(config),
      transformer_(config)
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult Lille130mModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput Lille130mModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Lille130mModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = transformer_(inputs, cache);
    // Lille130m uses tied embeddings (embed_as_linear)
    return transformer_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Lille130mModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Dequantize all affine-quantized weights at load time using the bits and
    // group_size from the model config. Lille-130m is tiny (130M params), so
    // dequantizing to float32 (~520MB) is fine. This bypasses quantized_matmul
    // entirely, avoiding potential issues with the ROCm quantized kernel path
    // for this particular model.
    if (config_.quant_bits > 0 && config_.quant_group_size > 0) {
        int bits = config_.quant_bits;
        int group_size = config_.quant_group_size;
        std::vector<std::string> to_remove;
        std::vector<std::pair<std::string, mx::array>> to_add;
        const std::string scales_suffix = ".scales";

        for (const auto& [key, scales] : weights) {
            if (key.size() <= scales_suffix.size() ||
                key.compare(key.size() - scales_suffix.size(), scales_suffix.size(), scales_suffix) != 0)
                continue;

            auto prefix = key.substr(0, key.size() - scales_suffix.size());
            auto weight_key = prefix + ".weight";
            auto weight_it = weights.find(weight_key);
            if (weight_it == weights.end()) continue;

            std::optional<mx::array> biases;
            auto biases_key = prefix + ".biases";
            auto biases_it = weights.find(biases_key);
            if (biases_it != weights.end()) {
                biases = biases_it->second;
                to_remove.push_back(biases_key);
            }

            to_add.emplace_back(weight_key,
                mx::dequantize(weight_it->second, scales, biases, group_size, bits));
            to_remove.push_back(key);
        }

        for (auto& [k, v] : to_add) {
            weights.insert_or_assign(k, std::move(v));
        }
        for (const auto& k : to_remove) {
            weights.erase(k);
        }
    }

    // Remove unused precomputed rotary frequencies.
    std::vector<std::string> to_remove;
    for (auto& [k, v] : weights) {
        if (k.find("rotary_emb") != std::string::npos) {
            to_remove.push_back(k);
        }
    }
    for (const auto& k : to_remove) {
        weights.erase(k);
    }
    return weights;
}

void Lille130mModel::load_weights(
    const std::unordered_map<std::string, mx::array>& weights)
{
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
        }
    }
}

std::unordered_map<std::string, mx::array*> Lille130mModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    // Lille130m checkpoints store the inner model under the transformer.*
    // prefix. Keeping the keys aligned here lets sanitization and loading bind
    // checkpoint weights (including .scales/.biases companions) to members.
    for (auto& [k, v] : transformer_.weight_map()) {
        map["transformer." + k] = v;
    }
    return map;
}

} // namespace mlx_lm
