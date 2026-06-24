// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2.swift (non-MoE variant)

#include <mlx-lm/llm/models/lfm2.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, LFM2Configuration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.norm_eps = j.at("norm_eps").get<float>();
    c.conv_bias = j.value("conv_bias", false);
    c.conv_l_cache = j.contains("conv_L_cache") ? j["conv_L_cache"].get<int>() : 3;

    // block_dim defaults to hidden_size
    c.block_dim = j.contains("block_dim") ? j["block_dim"].get<int>() : c.hidden_size;
    // block_ff_dim defaults to hidden_size
    c.block_ff_dim = j.contains("block_ff_dim") ? j["block_ff_dim"].get<int>() : c.hidden_size;
    c.block_multiple_of = j.value("block_multiple_of", 256);
    c.block_ffn_dim_multiplier = j.value("block_ffn_dim_multiplier", 1.0f);
    c.block_auto_adjust_ff_dim = j.value("block_auto_adjust_ff_dim", true);
    c.rope_theta = j.value("rope_theta", 1000000.0f);

    // Compute full_attn_idxs from explicit list or layer_types
    if (j.contains("full_attn_idxs") && !j["full_attn_idxs"].is_null()) {
        c.full_attn_idxs = j["full_attn_idxs"].get<std::vector<int>>();
    } else if (j.contains("layer_types") && !j["layer_types"].is_null()) {
        auto lt = j["layer_types"].get<std::vector<std::string>>();
        for (int i = 0; i < static_cast<int>(lt.size()); ++i) {
            if (lt[i] == "full_attention") c.full_attn_idxs.push_back(i);
        }
    } else {
        // Default: all layers are attention layers
        for (int i = 0; i < c.num_hidden_layers; ++i) {
            c.full_attn_idxs.push_back(i);
        }
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// Compute adjusted FF dim from config
static int compute_ff_dim(const LFM2Configuration& config) {
    int ff_dim = config.block_ff_dim;
    if (config.block_auto_adjust_ff_dim) {
        ff_dim = static_cast<int>(2.0f * ff_dim / 3.0f * config.block_ffn_dim_multiplier);
        ff_dim = ((ff_dim + config.block_multiple_of - 1) / config.block_multiple_of) * config.block_multiple_of;
    }
    return ff_dim;
}

// --- LFM2Attention ---

LFM2Attention::LFM2Attention(const LFM2Configuration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim()),
      scale_(std::pow(static_cast<float>(config.head_dim()), -0.5f)),
      rope_theta_(config.rope_theta),
      norm_eps_(config.norm_eps),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim()})),
      q_norm_weight_(mx::ones({config.head_dim()})),
      k_norm_weight_(mx::ones({config.head_dim()}))
{}

mx::array LFM2Attention::operator()(const mx::array& x,
                                      const AttentionMask& mask,
                                      KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Q/K norms before reshape->transpose (applied on last dim which is nHeads*headDim)
    queries = mx::fast::rms_norm(mx::reshape(queries, {B, L, num_heads_, -1}), q_norm_weight_, norm_eps_);
    keys = mx::fast::rms_norm(mx::reshape(keys, {B, L, num_kv_heads_, -1}), k_norm_weight_, norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3});
    keys = mx::transpose(keys, {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"out_proj.weight", &wo_weight_},
        {"q_layernorm.weight", &q_norm_weight_}, {"k_layernorm.weight", &k_norm_weight_},
    };
}

// --- LFM2ShortConv ---

LFM2ShortConv::LFM2ShortConv(const LFM2Configuration& config)
    : hidden_size_(config.hidden_size),
      l_cache_(config.conv_l_cache),
      bias_(config.conv_bias),
      conv_weight_(mx::zeros({config.hidden_size, config.conv_l_cache, 1})),
      in_proj_weight_(mx::zeros({3 * config.hidden_size, config.hidden_size})),
      out_proj_weight_(mx::zeros({config.hidden_size, config.hidden_size}))
{
    if (bias_) {
        conv_bias_ = mx::zeros({config.hidden_size});
        in_proj_bias_ = mx::zeros({3 * config.hidden_size});
        out_proj_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array LFM2ShortConv::operator()(const mx::array& x,
                                      KVCache* cache) {
    // in_proj: x -> [B, L, 3*hidden] (split into 3)
    auto bcx = linear_fwd(x, in_proj_weight_,
        in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);

    int third = hidden_size_;
    auto b_part = mx::slice(bcx, {0, 0, 0}, {bcx.shape(0), bcx.shape(1), third});
    auto c_part = mx::slice(bcx, {0, 0, third}, {bcx.shape(0), bcx.shape(1), 2 * third});
    auto x_comp = mx::slice(bcx, {0, 0, 2 * third}, {bcx.shape(0), bcx.shape(1), 3 * third});

    auto bx = mx::multiply(b_part, x_comp);

    // Get/create conv state from MambaCache
    mx::array state = mx::zeros({bx.shape(0), l_cache_ - 1, hidden_size_}, bx.dtype());
    if (cache) {
        auto* mc = cache->as_mamba();
        if (mc && (*mc)[0].has_value()) {
            state = (*mc)[0].value();
        }
    }

    bx = mx::concatenate({state, bx}, 1);

    // Store updated conv state
    if (cache) {
        auto* mc = cache->as_mamba();
        if (mc) {
            int start = bx.shape(1) - (l_cache_ - 1);
            (*mc)[0] = mx::slice(bx, {0, start, 0}, {bx.shape(0), bx.shape(1), bx.shape(2)});
        }
    }

    // Depthwise conv1d
    auto conv_out = mx::conv1d(bx, conv_weight_, /*stride=*/1, /*padding=*/0, /*dilation=*/1, hidden_size_);
    if (conv_bias_.has_value()) {
        conv_out = mx::add(conv_out, conv_bias_.value());
    }

    auto y = mx::multiply(c_part, conv_out);
    return linear_fwd(y, out_proj_weight_,
        out_proj_bias_.has_value() ? &out_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> LFM2ShortConv::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"conv.weight", &conv_weight_},
        {"in_proj.weight", &in_proj_weight_},
        {"out_proj.weight", &out_proj_weight_},
    };
    if (conv_bias_.has_value()) map["conv.bias"] = &conv_bias_.value();
    if (in_proj_bias_.has_value()) map["in_proj.bias"] = &in_proj_bias_.value();
    if (out_proj_bias_.has_value()) map["out_proj.bias"] = &out_proj_bias_.value();
    return map;
}

// --- LFM2MLP ---

LFM2MLP::LFM2MLP(int hidden_size, int ff_size)
    : gate_weight_(mx::zeros({ff_size, hidden_size})),
      up_weight_(mx::zeros({ff_size, hidden_size})),
      down_weight_(mx::zeros({hidden_size, ff_size}))
{}

mx::array LFM2MLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- LFM2DecoderLayer ---

LFM2DecoderLayer::LFM2DecoderLayer(const LFM2Configuration& config, int layer_idx)
    : is_attention_layer_(false),
      mlp_(config.block_dim, compute_ff_dim(config)),
      operator_norm_weight_(mx::ones({config.hidden_size})),
      ffn_norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.norm_eps)
{
    // Check if this layer is an attention layer
    std::set<int> attn_set(config.full_attn_idxs.begin(), config.full_attn_idxs.end());
    is_attention_layer_ = attn_set.count(layer_idx) > 0;

    if (is_attention_layer_) {
        attention_.emplace(config);
    } else {
        conv_.emplace(config);
    }
}

mx::array LFM2DecoderLayer::operator()(const mx::array& x,
                                         const AttentionMask& attn_mask,
                                         KVCache* cache) {
    auto residual = mx::fast::rms_norm(x, operator_norm_weight_, norm_eps_);

    mx::array r = is_attention_layer_
        ? (*attention_)(residual, attn_mask, cache)
        : (*conv_)(residual, cache);

    auto h = mx::add(x, r);
    auto normed = mx::fast::rms_norm(h, ffn_norm_weight_, norm_eps_);
    r = mlp_(normed);
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> LFM2DecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (is_attention_layer_) {
        for (auto& [k, v] : attention_->weight_map()) map["self_attn." + k] = v;
    } else {
        for (auto& [k, v] : conv_->weight_map()) map["conv." + k] = v;
    }
    for (auto& [k, v] : mlp_.weight_map()) map["feed_forward." + k] = v;
    map["operator_norm.weight"] = &operator_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;
    return map;
}

// --- LFM2ModelInner ---

LFM2ModelInner::LFM2ModelInner(const LFM2Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      embedding_norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.norm_eps),
      first_attn_idx_(-1)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }

    // Find first attention layer index
    if (!config.full_attn_idxs.empty()) first_attn_idx_ = config.full_attn_idxs[0];
}

mx::array LFM2ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Attention mask
    AttentionMask attn_mask;
    if (first_attn_idx_ >= 0) {
        attn_mask = create_attention_mask(h,
            cache && first_attn_idx_ < static_cast<int>(cache->size())
                ? &(*cache)[first_attn_idx_] : nullptr);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, attn_mask, lc);
    }

    return mx::fast::rms_norm(h, embedding_norm_weight_, norm_eps_);
}

mx::array LFM2ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> LFM2ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["embedding_norm.weight"] = &embedding_norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- LFM2Model ---

LFM2Model::LFM2Model(const LFM2Configuration& config)
    : config_(config), model_(config_)
{}

PrepareResult LFM2Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput LFM2Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array LFM2Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return model_.embed_as_linear(out);
}

std::vector<KVCache> LFM2Model::new_cache_impl(const GenerateParameters& params) {
    std::set<int> attn_set(config_.full_attn_idxs.begin(), config_.full_attn_idxs.end());
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        if (attn_set.count(i)) {
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            } else {
                caches.emplace_back(KVCacheSimple{});
            }
        } else {
            caches.emplace_back(MambaCache());
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
LFM2Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Rename w1/w2/w3 to gate_proj/down_proj/up_proj and transpose conv weights
    std::unordered_map<std::string, mx::array> sanitized;
    for (auto& [name, param] : weights) {
        auto tensor = param;
        if (name.find("conv.weight") != std::string::npos) {
            if (tensor.shape(-1) > tensor.shape(1)) {
                tensor = mx::transpose(tensor, {0, 2, 1});
            }
        }

        std::string updated = name;
        if (updated.find("w1.weight") != std::string::npos) {
            auto pos = updated.find("w1.weight");
            updated.replace(pos, 9, "gate_proj.weight");
        } else if (updated.find("w2.weight") != std::string::npos) {
            auto pos = updated.find("w2.weight");
            updated.replace(pos, 9, "down_proj.weight");
        } else if (updated.find("w3.weight") != std::string::npos) {
            auto pos = updated.find("w3.weight");
            updated.replace(pos, 9, "up_proj.weight");
        }
        sanitized.insert_or_assign(updated, std::move(tensor));
    }
    return sanitized;
}

void LFM2Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> LFM2Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

} // namespace mlx_lm
