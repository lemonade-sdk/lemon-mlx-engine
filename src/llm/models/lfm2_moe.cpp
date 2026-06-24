// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2MoE.swift

#include <mlx-lm/llm/models/lfm2_moe.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, LFM2MoEConfiguration& c) {
    c.vocab_size = j.at("vocab_size").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.moe_intermediate_size = j.at("moe_intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_experts = j.at("num_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.norm_topk_prob = j.at("norm_topk_prob").get<bool>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.use_expert_bias = j.at("use_expert_bias").get<bool>();
    c.num_dense_layers = j.at("num_dense_layers").get<int>();
    c.norm_eps = j.at("norm_eps").get<float>();
    c.conv_bias = j.at("conv_bias").get<bool>();
    c.conv_l_cache = j.at("conv_L_cache").get<int>();
    c.rope_theta = j.at("rope_theta").get<float>();

    // Compute full_attn_idxs from explicit list or layer_types
    if (j.contains("full_attn_idxs") && !j["full_attn_idxs"].is_null()) {
        c.full_attn_idxs = j["full_attn_idxs"].get<std::vector<int>>();
    } else if (j.contains("layer_types") && !j["layer_types"].is_null()) {
        auto lt = j["layer_types"].get<std::vector<std::string>>();
        for (int i = 0; i < static_cast<int>(lt.size()); ++i) {
            if (lt[i] == "full_attention") c.full_attn_idxs.push_back(i);
        }
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- LFM2MoEAttention ---

LFM2MoEAttention::LFM2MoEAttention(const LFM2MoEConfiguration& config)
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

mx::array LFM2MoEAttention::operator()(const mx::array& x,
                                          const AttentionMask& mask,
                                          KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    // Q/K norms before reshape→transpose (applied on last dim which is nHeads*headDim)
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

std::unordered_map<std::string, mx::array*> LFM2MoEAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"out_proj.weight", &wo_weight_},
        {"q_layernorm.weight", &q_norm_weight_}, {"k_layernorm.weight", &k_norm_weight_},
    };
}

// --- LFM2MoEShortConv ---

LFM2MoEShortConv::LFM2MoEShortConv(const LFM2MoEConfiguration& config, int /*layer_idx*/)
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

mx::array LFM2MoEShortConv::operator()(const mx::array& x,
                                          const std::optional<mx::array>& ssm_mask,
                                          KVCache* cache) {
    // in_proj: x → [B, C, x_comp] (split into 3)
    auto bcx = linear_fwd(x, in_proj_weight_,
        in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);

    int third = hidden_size_;
    auto b_part = mx::slice(bcx, {0, 0, 0}, {bcx.shape(0), bcx.shape(1), third});
    auto c_part = mx::slice(bcx, {0, 0, third}, {bcx.shape(0), bcx.shape(1), 2 * third});
    auto x_comp = mx::slice(bcx, {0, 0, 2 * third}, {bcx.shape(0), bcx.shape(1), 3 * third});

    auto bx = mx::multiply(b_part, x_comp);

    if (ssm_mask.has_value()) {
        auto expanded_mask = mx::expand_dims(ssm_mask.value(), -1);
        bx = mx::where(expanded_mask, bx, mx::zeros_like(bx));
    }

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

std::unordered_map<std::string, mx::array*> LFM2MoEShortConv::weight_map() {
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

// --- LFM2MoEMLP ---

LFM2MoEMLP::LFM2MoEMLP(int hidden_size, int intermediate_size)
    : gate_weight_(mx::zeros({intermediate_size, hidden_size})),
      up_weight_(mx::zeros({intermediate_size, hidden_size})),
      down_weight_(mx::zeros({hidden_size, intermediate_size}))
{}

mx::array LFM2MoEMLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2MoEMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- LFM2MoESparseMoeBlock ---

LFM2MoESparseMoeBlock::LFM2MoESparseMoeBlock(const LFM2MoEConfiguration& config)
    : num_experts_(config.num_experts),
      top_k_(config.num_experts_per_tok),
      norm_topk_prob_(config.norm_topk_prob),
      use_expert_bias_(config.use_expert_bias),
      gate_weight_(mx::zeros({config.num_experts, config.hidden_size})),
      switch_mlp_(config.hidden_size, config.moe_intermediate_size, config.num_experts)
{
    if (use_expert_bias_) {
        expert_bias_ = mx::zeros({config.num_experts});
    }
}

mx::array LFM2MoESparseMoeBlock::operator()(const mx::array& x) {
    auto gates = mx::softmax(mx::astype(linear_fwd(x, gate_weight_), mx::float32), -1);

    if (use_expert_bias_ && expert_bias_.has_value()) {
        gates = mx::add(gates, expert_bias_.value());
    }

    int k = top_k_;
    auto inds = mx::argpartition(mx::negative(gates), k - 1, -1);
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto scores = mx::take_along_axis(gates, inds, -1);

    if (norm_topk_prob_) {
        scores = mx::divide(scores, mx::add(mx::sum(scores, -1, true), mx::array(1e-20f)));
    }
    scores = mx::astype(scores, x.dtype());

    auto expert_out = switch_mlp_(x, inds);
    auto weighted = mx::multiply(expert_out, mx::expand_dims(scores, -1));
    return mx::sum(weighted, -2);
}

std::unordered_map<std::string, mx::array*> LFM2MoESparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["gate.weight"] = &gate_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    if (expert_bias_.has_value()) map["expert_bias"] = &expert_bias_.value();
    return map;
}

// --- LFM2MoEDecoderLayer ---

LFM2MoEDecoderLayer::LFM2MoEDecoderLayer(const LFM2MoEConfiguration& config, int layer_idx)
    : is_attention_layer_(false),
      uses_dense_ff_(layer_idx < config.num_dense_layers),
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
        conv_.emplace(config, layer_idx);
    }

    if (uses_dense_ff_) {
        dense_ff_.emplace(config.hidden_size, config.intermediate_size);
    } else {
        sparse_ff_.emplace(config);
    }
}

mx::array LFM2MoEDecoderLayer::operator()(const mx::array& x,
                                             const AttentionMask& attn_mask,
                                             const std::optional<mx::array>& ssm_mask,
                                             KVCache* cache) {
    auto residual = mx::fast::rms_norm(x, operator_norm_weight_, norm_eps_);

    mx::array r = is_attention_layer_
        ? (*attention_)(residual, attn_mask, cache)
        : (*conv_)(residual, ssm_mask, cache);

    auto h = mx::add(x, r);
    auto normed = mx::fast::rms_norm(h, ffn_norm_weight_, norm_eps_);

    if (uses_dense_ff_) {
        r = (*dense_ff_)(normed);
    } else {
        r = (*sparse_ff_)(normed);
    }
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> LFM2MoEDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (is_attention_layer_) {
        for (auto& [k, v] : attention_->weight_map()) map["self_attn." + k] = v;
    } else {
        for (auto& [k, v] : conv_->weight_map()) map["conv." + k] = v;
    }
    if (uses_dense_ff_) {
        for (auto& [k, v] : dense_ff_->weight_map()) map["feed_forward." + k] = v;
    } else {
        for (auto& [k, v] : sparse_ff_->weight_map()) map["feed_forward." + k] = v;
    }
    map["operator_norm.weight"] = &operator_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;
    return map;
}

// --- LFM2MoEModelInner ---

LFM2MoEModelInner::LFM2MoEModelInner(const LFM2MoEConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      embedding_norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.norm_eps),
      first_attn_idx_(-1), first_conv_idx_(-1)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }

    // Find first attention and first conv layer indices
    if (!config.full_attn_idxs.empty()) first_attn_idx_ = config.full_attn_idxs[0];
    for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
        if (!layers_[i].is_attention()) { first_conv_idx_ = i; break; }
    }
}

mx::array LFM2MoEModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Attention mask
    AttentionMask attn_mask;
    if (first_attn_idx_ >= 0) {
        attn_mask = create_attention_mask(h,
            cache && first_attn_idx_ < static_cast<int>(cache->size())
                ? &(*cache)[first_attn_idx_] : nullptr);
    }

    // SSM mask (for conv layers) - nullopt for single-sequence inference
    std::optional<mx::array> ssm_mask;

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, attn_mask, ssm_mask, lc);
    }

    return mx::fast::rms_norm(h, embedding_norm_weight_, norm_eps_);
}

mx::array LFM2MoEModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

std::unordered_map<std::string, mx::array*> LFM2MoEModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["embedding_norm.weight"] = &embedding_norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- LFM2MoEModel ---

LFM2MoEModel::LFM2MoEModel(const LFM2MoEConfiguration& config)
    : config_(config), model_(config_)
{
    std::set<int> attn_set(config.full_attn_idxs.begin(), config.full_attn_idxs.end());
    kv_heads_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        kv_heads_[i] = attn_set.count(i) ? config.num_key_value_heads : 0;
    }
}

PrepareResult LFM2MoEModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput LFM2MoEModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array LFM2MoEModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return model_.embed_as_linear(out);
}

std::vector<KVCache> LFM2MoEModel::new_cache_impl(const GenerateParameters& params) {
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
LFM2MoEModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
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

    // Stack per-expert weights
    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".feed_forward.";
        std::string expert_prefix = prefix + "experts.";
        std::string key0 = expert_prefix + "0.gate_proj.weight";
        if (sanitized.find(key0) == sanitized.end()) continue;

        for (const auto& n : {"gate_proj", "down_proj", "up_proj"}) {
            std::string check = expert_prefix + "0." + n + ".weight";
            if (sanitized.find(check) == sanitized.end()) continue;
            std::vector<mx::array> to_join;
            to_join.reserve(config_.num_experts);
            for (int e = 0; e < config_.num_experts; ++e) {
                std::string ek = expert_prefix + std::to_string(e) + "." + n + ".weight";
                auto it = sanitized.find(ek);
                to_join.push_back(std::move(it->second));
                sanitized.erase(it);
            }
            sanitized.insert_or_assign(prefix + "switch_mlp." + n + ".weight", mx::stack(to_join));
        }
    }
    return sanitized;
}

void LFM2MoEModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> LFM2MoEModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

} // namespace mlx_lm
