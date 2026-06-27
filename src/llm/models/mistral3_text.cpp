// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Mistral3Text.swift

#include <mlx-lm/llm/models/mistral3_text.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <algorithm>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Configuration ---

void from_json(const nlohmann::json& j, Mistral3TextConfiguration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.value("num_key_value_heads", c.num_attention_heads);
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);

    if (j.contains("head_dim") && !j["head_dim"].is_null()) {
        c.head_dim = j["head_dim"].get<int>();
    }
    if (j.contains("max_position_embeddings") && !j["max_position_embeddings"].is_null()) {
        c.max_position_embeddings = j["max_position_embeddings"].get<int>();
    }
    if (j.contains("sliding_window") && !j["sliding_window"].is_null()) {
        c.sliding_window = j["sliding_window"].get<int>();
    }

    if (j.contains("rope_parameters") && !j["rope_parameters"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> params;
        for (auto& [key, val] : j["rope_parameters"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            params[key] = sn;
        }
        c.rope_parameters = params;
    }

    if (j.contains("layer_types") && !j["layer_types"].is_null()) {
        c.layer_types = j["layer_types"].get<std::vector<std::string>>();
    } else {
        c.layer_types.assign(c.num_hidden_layers, "full_attention");
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Llama4 Attention Scaling ---

mx::array get_llama4_attention_scale(
    int start, int stop, float beta, int max_position_embeddings, mx::Dtype dtype)
{
    // positions = arange(start, stop)
    std::vector<int32_t> pos_data(stop - start);
    for (int i = 0; i < stop - start; ++i) pos_data[i] = start + i;
    auto positions = mx::array(pos_data.data(), {stop - start}, mx::int32);
    auto pos_f = mx::astype(positions, mx::float32);

    // scaling = 1 + beta * log(1 + floor(positions / max_pos_embed))
    auto floored = mx::floor(mx::divide(pos_f, mx::array(static_cast<float>(max_position_embeddings))));
    auto scaling = mx::add(mx::array(1.0f), mx::multiply(mx::array(beta), mx::log(mx::add(mx::array(1.0f), floored))));

    // Return shape [seq_len, 1] cast to dtype
    return mx::astype(mx::expand_dims(scaling, -1), dtype);
}

// --- Mistral3Attention ---

Mistral3Attention::Mistral3Attention(const Mistral3TextConfiguration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.resolved_head_dim()),
      scale_(std::pow(static_cast<float>(args.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      rope_theta_(10000.0f)
{
    // Get rope_theta from rope_parameters
    if (args.rope_parameters.has_value()) {
        auto& params = args.rope_parameters.value();
        auto it = params.find("rope_theta");
        if (it != params.end() && it->second.is_float()) {
            rope_theta_ = it->second.as_float();
        }
    }
}

mx::array Mistral3Attention::operator()(
    const mx::array& x, const mx::array& attn_scale,
    const AttentionMask& mask, KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = mx::transpose(mx::reshape(linear_fwd(x, wq_weight_), {B, L, num_heads_, -1}), {0, 2, 1, 3});
    auto keys = mx::transpose(mx::reshape(linear_fwd(x, wk_weight_), {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    auto values = mx::transpose(mx::reshape(linear_fwd(x, wv_weight_), {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, 1.0f, offset);

    // Apply attention scaling: queries = queries * attn_scale
    queries = mx::multiply(queries, attn_scale);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Mistral3Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- Mistral3MLP ---

Mistral3MLP::Mistral3MLP(int dim, int hidden_dim)
    : gate_weight_(mx::zeros({hidden_dim, dim})),
      down_weight_(mx::zeros({dim, hidden_dim})),
      up_weight_(mx::zeros({hidden_dim, dim}))
{}

mx::array Mistral3MLP::operator()(const mx::array& x) {
    return linear_fwd(swiglu(linear_fwd(x, gate_weight_), linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> Mistral3MLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// --- Mistral3TextTransformerBlock ---

Mistral3TextTransformerBlock::Mistral3TextTransformerBlock(const Mistral3TextConfiguration& args, bool use_sliding)
    : use_sliding_(use_sliding),
      attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array Mistral3TextTransformerBlock::operator()(
    const mx::array& x, const mx::array& attn_scale,
    const AttentionMask& mask, KVCache* cache)
{
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), attn_scale, mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Mistral3TextTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Mistral3TextModelInner ---

Mistral3TextModelInner::Mistral3TextModelInner(const Mistral3TextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      layer_types_(args.layer_types),
      sliding_window_(args.sliding_window),
      fa_idx_(0),
      swa_idx_(-1),
      llama4_scaling_beta_(1.0f),
      original_max_pos_embed_(4096)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        bool use_sliding = (i < static_cast<int>(layer_types_.size()) && layer_types_[i] == "sliding_attention");
        layers_.emplace_back(args, use_sliding);
    }

    // Find first full attention and sliding window indices
    for (int i = 0; i < static_cast<int>(layer_types_.size()); ++i) {
        if (layer_types_[i] == "full_attention") { fa_idx_ = i; break; }
    }
    for (int i = 0; i < static_cast<int>(layer_types_.size()); ++i) {
        if (layer_types_[i] == "sliding_attention") { swa_idx_ = i; break; }
    }

    // Extract rope scaling parameters
    if (args.rope_parameters.has_value()) {
        auto& params = args.rope_parameters.value();
        auto beta_it = params.find("llama_4_scaling_beta");
        if (beta_it != params.end() && beta_it->second.is_float()) {
            llama4_scaling_beta_ = beta_it->second.as_float();
        }
        auto orig_it = params.find("original_max_position_embeddings");
        if (orig_it != params.end() && orig_it->second.is_float()) {
            original_max_pos_embed_ = static_cast<int>(orig_it->second.as_float());
        }
    }
}

mx::array Mistral3TextModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    int offset = (cache && !cache->empty()) ? (*cache)[0].offset() : 0;
    int seq_len = inputs.shape(1);

    auto fa_mask = create_attention_mask(h, cache && fa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[fa_idx_] : nullptr);

    AttentionMask swa_mask;
    if (swa_idx_ >= 0 && sliding_window_.has_value()) {
        swa_mask = create_attention_mask(h, cache && swa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[swa_idx_] : nullptr, sliding_window_.value());
    }

    // Compute attention scale
    auto attn_scale = get_llama4_attention_scale(
        offset, offset + seq_len, llama4_scaling_beta_, original_max_pos_embed_, h.dtype());

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        auto& mask = layers_[i].use_sliding() && !swa_mask.is_none() ? swa_mask : fa_mask;
        h = layers_[i](h, attn_scale, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Mistral3TextModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Mistral3TextModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Mistral3TextModel ---

Mistral3TextModel::Mistral3TextModel(const Mistral3TextConfiguration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Mistral3TextModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Mistral3TextModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Mistral3TextModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return linear_forward(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::vector<KVCache> Mistral3TextModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (const auto& layer : model_.get_layers()) {
        if (layer.use_sliding() && config_.sliding_window.has_value()) {
            caches.emplace_back(RotatingKVCache(config_.sliding_window.value()));
        } else {
            caches.emplace_back(KVCacheSimple{});
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
Mistral3TextModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Handle language_model prefix (VLM compatibility)
    std::unordered_map<std::string, mx::array> processed;
    bool has_lm_prefix = false;
    for (auto& [key, val] : weights) {
        if (key.find("language_model.") == 0) {
            has_lm_prefix = true;
            processed.insert_or_assign(key.substr(15), std::move(val));
        }
    }
    if (has_lm_prefix) weights = std::move(processed);

    // Remove rotary_emb.inv_freq
    for (auto it = weights.begin(); it != weights.end(); ) {
        if (it->first.find("rotary_emb.inv_freq") != std::string::npos) {
            it = weights.erase(it);
        } else {
            ++it;
        }
    }

    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");

    // Handle weight_scale_inv for quantized weights
    std::unordered_map<std::string, mx::array> new_weights;
    bool has_scale_inv = false;
    for (auto& [key, value] : weights) {
        if (key.find("weight_scale_inv") != std::string::npos) {
            has_scale_inv = true;
            auto weight_key = key;
            auto pos = weight_key.find("_scale_inv");
            weight_key.erase(pos, 10);
            auto wit = weights.find(weight_key);
            if (wit != weights.end()) {
                new_weights.insert_or_assign(weight_key, mx::multiply(wit->second, value));
            }
        } else if (key.find("activation_scale") != std::string::npos) {
            continue;
        } else if (new_weights.find(key) == new_weights.end()) {
            new_weights.insert_or_assign(key, value);
        }
    }

    return has_scale_inv ? new_weights : weights;
}

void Mistral3TextModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Mistral3TextModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
