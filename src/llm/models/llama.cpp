// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/llama.py

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <mlx-lm/llm/models/llama.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/bitnet_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// --- JSON deserialization ---

void from_json(const nlohmann::json& j, LlamaConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    if (j.contains("head_dim") && !j["head_dim"].is_null())
        c.head_dim = j["head_dim"].get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);

    if (j.contains("max_position_embeddings"))
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();
    if (j.contains("rope_theta"))
        c.rope_theta = j["rope_theta"].get<float>();
    if (j.contains("rope_traditional"))
        c.rope_traditional = j["rope_traditional"].get<bool>();
    if (j.contains("tie_word_embeddings"))
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    if (j.contains("attention_bias"))
        c.attention_bias = j["attention_bias"].get<bool>();
    if (j.contains("mlp_bias"))
        c.mlp_bias = j["mlp_bias"].get<bool>();
    if (j.contains("hidden_act"))
        c.hidden_act = j["hidden_act"].get<std::string>();

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }
}

// --- compute_base_frequency ---

float compute_base_frequency(
    float base, int dims, const std::string& rope_type,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling)
{
    if (rope_type != "llama3") return base;
    if (!rope_scaling.has_value()) return base;

    const auto& rs = rope_scaling.value();
    auto get_float = [&](const std::string& key, float default_val) -> float {
        auto it = rs.find(key);
        if (it != rs.end() && it->second.is_float()) return it->second.as_float();
        return default_val;
    };

    float factor = get_float("factor", 1.0f);
    float low_freq_factor = get_float("low_freq_factor", 1.0f);
    float high_freq_factor = get_float("high_freq_factor", 4.0f);
    float old_context_len = get_float("original_max_position_embeddings", 8192.0f);

    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;

    std::vector<float> freqs;
    for (int i = 0; i < dims; i += 2) {
        freqs.push_back(std::pow(base, static_cast<float>(i) / static_cast<float>(dims)));
    }

    std::vector<float> new_base_freqs;
    for (float freq : freqs) {
        float wavelen = 2.0f * static_cast<float>(M_PI) / freq;
        float smooth = std::max(0.0f,
            std::min(1.0f,
                (wavelen - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)));
        new_base_freqs.push_back(freq * ((1.0f - smooth) * factor + smooth));
    }

    float sum = 0;
    for (float f : new_base_freqs) sum += f;
    return sum / static_cast<float>(new_base_freqs.size());
}

// --- LlamaDynamicNTKScalingRoPE ---

LlamaDynamicNTKScalingRoPE::LlamaDynamicNTKScalingRoPE(
    int dims,
    std::optional<int> max_position_embeddings,
    bool traditional,
    float base,
    float scale,
    const std::string& rope_type,
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling)
    : dims_(dims),
      max_position_embeddings_(max_position_embeddings.value_or(2048)),
      traditional_(traditional),
      base_(base),
      scale_(scale),
      rope_type_(rope_type)
{
    compute_freqs(rope_scaling);
}

void LlamaDynamicNTKScalingRoPE::compute_freqs(
    const std::optional<std::unordered_map<std::string, StringOrNumber>>& rope_scaling)
{
    if (rope_type_ != "llama3") {
        freqs_ = std::nullopt;
        return;
    }

    if (!rope_scaling.has_value() || !base_.has_value()) {
        freqs_ = std::nullopt;
        return;
    }

    const auto& rs = rope_scaling.value();
    auto get_float = [&](const std::string& key, float default_val) -> float {
        auto it = rs.find(key);
        if (it != rs.end() && it->second.is_float()) return it->second.as_float();
        return default_val;
    };

    float factor = get_float("factor", 1.0f);
    float low_freq_factor = get_float("low_freq_factor", 1.0f);
    float high_freq_factor = get_float("high_freq_factor", 4.0f);
    float old_context_len = get_float("original_max_position_embeddings", 8192.0f);

    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;
    float b = base_.value();

    auto indices = mx::arange(0, dims_, 2);
    auto frequencies = mx::power(mx::array(b), mx::divide(indices, mx::array(static_cast<float>(dims_))));
    auto wavelens = mx::multiply(mx::array(2.0f * static_cast<float>(M_PI)), frequencies);

    auto low_wl = mx::array(low_freq_wavelen);
    auto high_wl = mx::array(high_freq_wavelen);
    auto factor_arr = mx::array(factor);

    frequencies = mx::where(
        mx::greater(wavelens, low_wl),
        mx::multiply(frequencies, factor_arr),
        frequencies);

    auto is_medium_freq = mx::logical_and(
        mx::greater(wavelens, high_wl),
        mx::less(wavelens, low_wl));

    auto smooth_factors = mx::divide(
        mx::subtract(mx::divide(mx::array(old_context_len), wavelens), mx::array(low_freq_factor)),
        mx::array(high_freq_factor - low_freq_factor));

    auto smooth_freqs = mx::divide(frequencies,
        mx::add(
            mx::divide(mx::subtract(mx::array(1.0f), smooth_factors), factor_arr),
            smooth_factors));

    freqs_ = mx::where(is_medium_freq, smooth_freqs, frequencies);
    base_ = std::nullopt;
}

mx::array LlamaDynamicNTKScalingRoPE::operator()(const mx::array& x, int offset) {
    return mx::fast::rope(
        x,
        dims_,
        traditional_,
        base_,
        scale_,
        offset,
        freqs_);
}

// --- Linear helper ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight,
    const std::optional<mx::array>& bias)
{
    return linear_forward(x, weight, bias.has_value() ? &bias.value() : nullptr);
}

// --- LlamaAttention ---

LlamaAttention::LlamaAttention(const LlamaConfiguration& args)
    : args_(args),
      scale_(std::pow(static_cast<float>(args.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      rope_(args.resolved_head_dim(),
            args.max_position_embeddings,
            args.rope_traditional,
            args.rope_theta,
            1.0f,
            [&]() -> std::string {
                if (args.rope_scaling.has_value()) {
                    auto it = args.rope_scaling->find("type");
                    if (it == args.rope_scaling->end())
                        it = args.rope_scaling->find("rope_type");
                    if (it != args.rope_scaling->end() && it->second.is_string())
                        return it->second.as_string();
                }
                return "default";
            }(),
            args.rope_scaling)
{
    if (args.attention_bias) {
        wq_bias_ = mx::zeros({args.num_attention_heads * args.resolved_head_dim()});
        wk_bias_ = mx::zeros({args.num_key_value_heads * args.resolved_head_dim()});
        wv_bias_ = mx::zeros({args.num_key_value_heads * args.resolved_head_dim()});
        wo_bias_ = mx::zeros({args.hidden_size});
    }
}

mx::array LlamaAttention::linear(const mx::array& x, const mx::array& weight,
                                  const std::optional<mx::array>& bias) const {
    return linear_fwd(x, weight, bias);
}

mx::array LlamaAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);
    int head_dim = args_.resolved_head_dim();

    auto queries = linear(x, wq_weight_, wq_bias_);
    auto keys = linear(x, wk_weight_, wk_bias_);
    auto values = linear(x, wv_weight_, wv_bias_);

    // Reshape: [B, L, heads*head_dim] -> [B, heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, args_.num_attention_heads, head_dim}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys = rope_(keys, offset);

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

    return linear(output, wo_weight_, wo_bias_);
}

std::unordered_map<std::string, mx::array*> LlamaAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_) map["q_proj.bias"] = &wq_bias_.value();
    if (wk_bias_) map["k_proj.bias"] = &wk_bias_.value();
    if (wv_bias_) map["v_proj.bias"] = &wv_bias_.value();
    if (wo_bias_) map["o_proj.bias"] = &wo_bias_.value();
    return map;
}

// --- LlamaMLP ---

LlamaMLP::LlamaMLP(const LlamaConfiguration& args)
    : gate_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      down_weight_(mx::zeros({args.hidden_size, args.intermediate_size})),
      up_weight_(mx::zeros({args.intermediate_size, args.hidden_size}))
{
    if (args.mlp_bias) {
        gate_bias_ = mx::zeros({args.intermediate_size});
        down_bias_ = mx::zeros({args.hidden_size});
        up_bias_ = mx::zeros({args.intermediate_size});
    }
}

mx::array LlamaMLP::linear(const mx::array& x, const mx::array& weight,
                            const std::optional<mx::array>& bias) const {
    return linear_fwd(x, weight, bias);
}

mx::array LlamaMLP::operator()(const mx::array& x) {
    // swiglu(gate(x), up(x)) -> down
    auto up_out = linear(x, up_weight_, up_bias_);
    return linear(swiglu(linear(x, gate_weight_, gate_bias_), up_out), down_weight_, down_bias_);
}

std::unordered_map<std::string, mx::array*> LlamaMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
    if (gate_bias_) map["gate_proj.bias"] = &gate_bias_.value();
    if (down_bias_) map["down_proj.bias"] = &down_bias_.value();
    if (up_bias_) map["up_proj.bias"] = &up_bias_.value();
    return map;
}

// --- LlamaTransformerBlock ---

LlamaTransformerBlock::LlamaTransformerBlock(const LlamaConfiguration& args)
    : attention_(args),
      mlp_(args),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array LlamaTransformerBlock::rms_norm(const mx::array& x, const mx::array& weight) const {
    return mx::fast::rms_norm(x, weight, rms_norm_eps_);
}

mx::array LlamaTransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    auto r = attention_(rms_norm(x, input_layernorm_weight_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(rms_norm(h, post_attention_layernorm_weight_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> LlamaTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    // Attention weights
    for (auto& [k, v] : attention_.weight_map()) {
        map["self_attn." + k] = v;
    }

    // MLP weights
    for (auto& [k, v] : mlp_.weight_map()) {
        map["mlp." + k] = v;
    }

    // LayerNorm weights
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;

    return map;
}

// --- LlamaModelInner ---

LlamaModelInner::LlamaModelInner(const LlamaConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        layers_.emplace_back(args);
    }
}

mx::array LlamaModelInner::rms_norm(const mx::array& x, const mx::array& weight) const {
    return mx::fast::rms_norm(x, weight, rms_norm_eps_);
}

mx::array LlamaModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    // Embedding lookup
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Create attention mask
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    // Forward through layers
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return rms_norm(h, norm_weight_);
}

mx::array LlamaModelInner::embed_as_linear(const mx::array& x) const {
    // Use embedding weights as a linear layer (for tied embeddings)
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> LlamaModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;

    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) {
            map[prefix + k] = v;
        }
    }

    return map;
}

// --- LlamaModel ---

LlamaModel::LlamaModel(const LlamaConfiguration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);

    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult LlamaModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput LlamaModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array LlamaModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return mx::matmul(out, mx::transpose(lm_head_weight_.value()));
    } else {
        return model_.embed_as_linear(out);
    }
}

std::unordered_map<std::string, mx::array>
LlamaModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Dequantize BitNet-style uint8 packed ternary weights at load time.
    // Each *.weight (uint8, shape [out/4, in]) is paired with a *.weight_scale.
    // Normal Llama weights do not have this pair and are left unchanged.
    std::vector<std::string> to_remove;
    std::vector<std::pair<std::string, mx::array>> to_add;

    const std::string scale_suffix = ".weight_scale";

    for (auto& [key, val] : weights) {
        if (key.size() > scale_suffix.size() &&
            key.compare(key.size() - scale_suffix.size(), scale_suffix.size(), scale_suffix) == 0) {

            auto prefix = key.substr(0, key.size() - scale_suffix.size());
            auto weight_key = prefix + ".weight";

            auto w_it = weights.find(weight_key);
            if (w_it != weights.end() && w_it->second.dtype() == mx::uint8) {
                int packed_rows = w_it->second.shape(0);
                int out_features = packed_rows * 4;

                to_add.emplace_back(weight_key,
                    dequantize_bitnet_weight(w_it->second, val, out_features));
                to_remove.push_back(key);
            }
        }
    }

    for (auto& [k, v] : to_add) {
        weights.insert_or_assign(k, std::move(v));
    }

    // Remove unused precomputed rotary frequencies
    for (auto& [k, v] : weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            to_remove.push_back(k);
        }
    }
    for (const auto& k : to_remove) {
        weights.erase(k);
    }
    return weights;
}

void LlamaModel::load_weights(
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

std::unordered_map<std::string, mx::array*> LlamaModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : model_.weight_map()) {
        map["model." + k] = v;
    }

    if (lm_head_weight_.has_value()) {
        map["lm_head.weight"] = &lm_head_weight_.value();
    }

    return map;
}

} // namespace mlx_lm
