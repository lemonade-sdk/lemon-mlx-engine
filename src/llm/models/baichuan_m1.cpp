// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of BaichuanM1.swift

#include <mlx-lm/llm/models/baichuan_m1.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Config ---

void from_json(const nlohmann::json& j, BaichuanM1Configuration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.at("rope_theta").get<float>();
    c.sliding_window = j.at("sliding_window").get<int>();
    c.sliding_window_layers = j.at("sliding_window_layers").get<std::vector<int>>();
    c.conv_window = j.at("conv_window").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();

    if (j.contains("num_swa_attention_heads") && !j["num_swa_attention_heads"].is_null()) {
        c.num_swa_attention_heads = j["num_swa_attention_heads"].get<int>();
    }
    if (j.contains("num_swa_key_value_heads") && !j["num_swa_key_value_heads"].is_null()) {
        c.num_swa_key_value_heads = j["num_swa_key_value_heads"].get<int>();
    }
    if (j.contains("tie_word_embeddings")) {
        c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- BaichuanM1Attention ---

BaichuanM1Attention::BaichuanM1Attention(const BaichuanM1Configuration& config, int layer_idx)
    : layer_idx_(layer_idx),
      rope_theta_(config.rope_theta),
      conv_window_(config.conv_window),
      // W_pack: combined QKV = hidden_size + 2 * num_kv_heads * head_dim
      w_pack_weight_(mx::zeros({0})),  // placeholder, set below
      o_proj_weight_(mx::zeros({0})),  // placeholder
      conv_k_(mx::zeros({1, 1, 0, 1, config.conv_window})),
      conv_v_(mx::zeros({1, 1, 0, 1, config.conv_window}))
{
    // Determine if this is a sliding window layer
    std::set<int> swa_set(config.sliding_window_layers.begin(), config.sliding_window_layers.end());
    is_swa_ = swa_set.count(layer_idx) > 0;

    // SWA layers may have different head counts
    num_heads_ = (is_swa_ && config.num_swa_attention_heads.has_value())
        ? config.num_swa_attention_heads.value() : config.num_attention_heads;
    num_kv_heads_ = (is_swa_ && config.num_swa_key_value_heads.has_value())
        ? config.num_swa_key_value_heads.value() : config.num_key_value_heads;

    head_dim_ = config.hidden_size / num_heads_;
    scale_ = std::pow(static_cast<float>(head_dim_), -0.5f);

    int qkv_dim = config.hidden_size + 2 * num_kv_heads_ * head_dim_;
    w_pack_weight_ = mx::zeros({qkv_dim, config.hidden_size});
    o_proj_weight_ = mx::zeros({config.hidden_size, num_heads_ * head_dim_});
    conv_k_ = mx::zeros({1, 1, num_kv_heads_, 1, conv_window_});
    conv_v_ = mx::zeros({1, 1, num_kv_heads_, 1, conv_window_});
}

mx::array BaichuanM1Attention::custom_convolution(
    const mx::array& u, const mx::array& weights,
    const std::optional<mx::array>& state)
{
    // u: [B, H, L, D], weights: [1, 1, H, 1, conv_window]
    // 2-tap FIR: w0 * u_prev + w1 * u
    int L = u.shape(2);
    auto reshaped_w = mx::reshape(weights, {1, weights.shape(2), conv_window_, 1, 1});
    auto w0 = mx::slice(reshaped_w, {0, 0, 0, 0, 0}, {1, reshaped_w.shape(1), 1, 1, 1});
    auto w1 = mx::slice(reshaped_w, {0, 0, 1, 0, 0}, {1, reshaped_w.shape(1), 2, 1, 1});
    // Squeeze conv_window dim: w0, w1 are [1, H, 1, 1]
    w0 = mx::squeeze(w0, 2);
    w1 = mx::squeeze(w1, 2);

    // State: [B, H, 1, D] (previous last frame) or zeros
    auto prev_state = state.has_value()
        ? state.value()
        : mx::zeros({u.shape(0), u.shape(1), 1, u.shape(3)}, u.dtype());

    // u_prev: concat(state, u[:,:,:L-1,:])
    auto u_prev = L > 1
        ? mx::concatenate({prev_state, mx::slice(u, {0, 0, 0, 0}, {u.shape(0), u.shape(1), L - 1, u.shape(3)})}, 2)
        : prev_state;

    return mx::add(mx::multiply(u_prev, w0), mx::multiply(u, w1));
}

mx::array BaichuanM1Attention::operator()(const mx::array& x,
                                           const AttentionMask& mask,
                                           KVCache* cache) {
    int B = x.shape(0), L = x.shape(1), D = x.shape(2);

    auto proj = linear_fwd(x, w_pack_weight_);

    // Split into Q, K, V
    int kv_dim = num_kv_heads_ * head_dim_;
    auto queries = mx::slice(proj, {0, 0, 0}, {B, L, D});
    auto keys = mx::slice(proj, {0, 0, D}, {B, L, D + kv_dim});
    auto values = mx::slice(proj, {0, 0, D + kv_dim}, {B, L, D + 2 * kv_dim});

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Get conv state from MambaCache (sub-part of CompoundCache)
    int offset = 0;
    std::optional<mx::array> last_k, last_v;

    if (cache) {
        offset = cache->offset();
        auto* mc = cache->as_mamba();
        if (mc) {
            if ((*mc)[0].has_value()) last_k = (*mc)[0].value();
            if ((*mc)[1].has_value()) last_v = (*mc)[1].value();
        }
    }

    // Save pre-conv K/V for state update
    auto k_init = keys;
    auto v_init = values;

    // Apply 2-tap conv on K and V
    keys = custom_convolution(keys, conv_k_, last_k);
    values = custom_convolution(values, conv_v_, last_v);

    // RoPE
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    // Update KV cache (delegates to the KVCacheSimple/RotatingKVCache inside CompoundCache)
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;

        // Store conv state (last frame of pre-conv K/V)
        auto* mc = cache->as_mamba();
        if (mc && L > 0) {
            (*mc)[0] = mx::slice(k_init, {0, 0, L - 1, 0}, {k_init.shape(0), k_init.shape(1), L, k_init.shape(3)});
            (*mc)[1] = mx::slice(v_init, {0, 0, L - 1, 0}, {v_init.shape(0), v_init.shape(1), L, v_init.shape(3)});
        }
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, o_proj_weight_);
}

std::unordered_map<std::string, mx::array*> BaichuanM1Attention::weight_map() {
    return {
        {"W_pack.weight", &w_pack_weight_},
        {"o_proj.weight", &o_proj_weight_},
        {"conv_k", &conv_k_},
        {"conv_v", &conv_v_},
    };
}

// --- BaichuanM1MLP ---

BaichuanM1MLP::BaichuanM1MLP(const BaichuanM1Configuration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size}))
{}

mx::array BaichuanM1MLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> BaichuanM1MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- BaichuanM1DecoderLayer ---

BaichuanM1DecoderLayer::BaichuanM1DecoderLayer(const BaichuanM1Configuration& config, int layer_idx)
    : attention_(config, layer_idx),
      mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array BaichuanM1DecoderLayer::operator()(const mx::array& x,
                                              const AttentionMask& mask,
                                              KVCache* cache) {
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_);
    auto h = attention_(normed, mask, cache);
    auto r = mx::add(x, h);
    auto ff_normed = mx::fast::rms_norm(r, post_attention_layernorm_weight_, norm_eps_);
    return mx::add(r, mlp_(ff_normed));
}

std::unordered_map<std::string, mx::array*> BaichuanM1DecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- BaichuanM1ModelInner ---

BaichuanM1ModelInner::BaichuanM1ModelInner(const BaichuanM1Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }
}

mx::array BaichuanM1ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    auto mask = create_attention_mask(h,
        (cache && !cache->empty()) ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

mx::array BaichuanM1ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> BaichuanM1ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- BaichuanM1Model ---

BaichuanM1Model::BaichuanM1Model(const BaichuanM1Configuration& config)
    : config_(config), model_(config_)
{
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult BaichuanM1Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput BaichuanM1Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array BaichuanM1Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (config_.tie_word_embeddings) {
        return model_.embed_as_linear(out);
    }
    return linear_fwd(out, lm_head_weight_.value());
}

std::vector<KVCache> BaichuanM1Model::new_cache_impl(const GenerateParameters& params) {
    std::set<int> swa_set(config_.sliding_window_layers.begin(), config_.sliding_window_layers.end());
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        MambaCache conv_cache;
        if (swa_set.count(i)) {
            caches.emplace_back(CompoundCache(std::move(conv_cache),
                                               RotatingKVCache(config_.sliding_window)));
        } else {
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(CompoundCache(std::move(conv_cache),
                                                   RotatingKVCache(params.max_kv_size.value(), 4)));
            } else {
                caches.emplace_back(CompoundCache(std::move(conv_cache), KVCacheSimple{}));
            }
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
BaichuanM1Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Normalize lm_head weights (L2 normalization per output row)
    auto lm_it = weights.find("lm_head.weight");
    bool is_quantized = weights.find("lm_head.scales") != weights.end();
    if (!is_quantized && lm_it != weights.end()) {
        auto w = mx::astype(lm_it->second, mx::float32);
        auto norm = mx::sqrt(mx::sum(mx::multiply(w, w), {-1}, true));
        w = mx::divide(w, mx::add(norm, mx::array(1e-7f)));
        lm_it->second = mx::astype(w, lm_it->second.dtype());
    }

    if (config_.tie_word_embeddings) {
        weights.erase("lm_head.weight");
    }

    return weights;
}

void BaichuanM1Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> BaichuanM1Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
