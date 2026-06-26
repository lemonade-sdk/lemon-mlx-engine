// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/llm/models/qwen3.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx-lm/common/bitnet_utils.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, Qwen3Configuration& c) {
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.head_dim = j.at("head_dim").get<int>();
    c.tie_word_embeddings = j.value("tie_word_embeddings", false);
    c.has_pre_norms = j.value("has_pre_norms", false);
    // Detect BitNet inverse weight scales: bitlinear class = 1/scale
    {
        std::string linear_class;
        if (j.contains("quantization_config") && j["quantization_config"].contains("linear_class")) {
            linear_class = j["quantization_config"]["linear_class"].get<std::string>();
        }
        c.bitnet_invert_weight_scales = (linear_class == "bitlinear");
    }

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

static mx::array linear_fwd(const mx::array& x, const mx::array& w) {
    return linear_forward(x, w);
}

// --- Qwen3Attention ---

Qwen3Attention::Qwen3Attention(const Qwen3Configuration& args)
    : num_heads_(args.num_attention_heads),
      num_kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      scale_(std::pow(static_cast<float>(args.head_dim), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim, args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim})),
      q_norm_weight_(mx::ones({args.head_dim})),
      k_norm_weight_(mx::ones({args.head_dim})),
      wq_pre_norm_(mx::ones({args.hidden_size})),
      wk_pre_norm_(mx::ones({args.hidden_size})),
      wv_pre_norm_(mx::ones({args.hidden_size})),
      wo_pre_norm_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps),
      rope_theta_(args.rope_theta),
      rope_scale_(1.0f)
{
    if (args.rope_scaling.has_value()) {
        auto& scaling = args.rope_scaling.value();
        auto type_it = scaling.find("type");
        if (type_it == scaling.end())
            type_it = scaling.find("rope_type");
        if (type_it != scaling.end() && type_it->second.is_string()) {
            auto rope_type = type_it->second.as_string();
            if (rope_type == "linear" || rope_type == "yarn") {
                auto factor_it = scaling.find("factor");
                if (factor_it != scaling.end() && factor_it->second.is_float()) {
                    rope_scale_ = 1.0f / factor_it->second.as_float();
                }
            }
        }
    }
}

mx::array Qwen3Attention::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    // Apply per-projection pre-norms if present (BitNet variants)
    auto xq = has_pre_norms_ ? mx::fast::rms_norm(x, wq_pre_norm_, rms_norm_eps_) : x;
    auto xk = has_pre_norms_ ? mx::fast::rms_norm(x, wk_pre_norm_, rms_norm_eps_) : x;
    auto xv = has_pre_norms_ ? mx::fast::rms_norm(x, wv_pre_norm_, rms_norm_eps_) : x;

    auto queries = linear_fwd(xq, wq_weight_);
    auto keys = linear_fwd(xk, wk_weight_);
    auto values = linear_fwd(xv, wv_weight_);

    // Reshape and apply Q/K norms before transpose
    queries = mx::reshape(queries, {B, L, num_heads_, -1});
    queries = mx::fast::rms_norm(queries, q_norm_weight_, rms_norm_eps_);
    queries = mx::transpose(queries, {0, 2, 1, 3});

    keys = mx::reshape(keys, {B, L, num_kv_heads_, -1});
    keys = mx::fast::rms_norm(keys, k_norm_weight_, rms_norm_eps_);
    keys = mx::transpose(keys, {0, 2, 1, 3});

    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, false, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3Attention::weight_map() {
    auto map = std::unordered_map<std::string, mx::array*>{
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
        {"q_norm.weight", &q_norm_weight_},
        {"k_norm.weight", &k_norm_weight_},
    };
    // Pre-projection norms (BitNet)
    if (has_pre_norms_) {
        map["q_proj.rms_norm.weight"] = &wq_pre_norm_;
        map["k_proj.rms_norm.weight"] = &wk_pre_norm_;
        map["v_proj.rms_norm.weight"] = &wv_pre_norm_;
        map["o_proj.rms_norm.weight"] = &wo_pre_norm_;
    }
    return map;
}

// --- Qwen3MLP ---

Qwen3MLP::Qwen3MLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions})),
      gate_pre_norm_(mx::ones({dimensions})),
      up_pre_norm_(mx::ones({dimensions})),
      down_pre_norm_(mx::ones({hidden_dimensions}))
{}

mx::array Qwen3MLP::operator()(const mx::array& x) {
    auto xg = has_pre_norms_ ? mx::fast::rms_norm(x, gate_pre_norm_, 1e-6f) : x;
    auto xu = has_pre_norms_ ? mx::fast::rms_norm(x, up_pre_norm_, 1e-6f) : x;
    auto g = linear_fwd(xg, gate_weight_);
    auto up = linear_fwd(xu, up_weight_);
    auto hidden = swiglu(g, up);
    auto xd = has_pre_norms_ ? mx::fast::rms_norm(hidden, down_pre_norm_, 1e-6f) : hidden;
    return linear_fwd(xd, down_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3MLP::weight_map() {
    auto map = std::unordered_map<std::string, mx::array*>{
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
    if (has_pre_norms_) {
        map["gate_proj.rms_norm.weight"] = &gate_pre_norm_;
        map["up_proj.rms_norm.weight"] = &up_pre_norm_;
        map["down_proj.rms_norm.weight"] = &down_pre_norm_;
    }
    return map;
}

// --- Qwen3TransformerBlock ---

Qwen3TransformerBlock::Qwen3TransformerBlock(const Qwen3Configuration& args)
    : attention_(args),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    if (args.has_pre_norms) {
        attention_.enable_pre_norms();
        mlp_.enable_pre_norms();
    }
}

mx::array Qwen3TransformerBlock::operator()(const mx::array& x, const AttentionMask& mask, KVCache* cache) {
    auto r = attention_(mx::fast::rms_norm(x, input_layernorm_weight_, rms_norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, rms_norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> Qwen3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- Qwen3ModelInner ---

Qwen3ModelInner::Qwen3ModelInner(const Qwen3Configuration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args);
}

mx::array Qwen3ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);
    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);
    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }
    return mx::fast::rms_norm(h, norm_weight_, rms_norm_eps_);
}

mx::array Qwen3ModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> Qwen3ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- Qwen3Model ---

Qwen3Model::Qwen3Model(const Qwen3Configuration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult Qwen3Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput Qwen3Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array Qwen3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) return linear_fwd(out, lm_head_weight_.value());
    return model_.embed_as_linear(out);
}

std::unordered_map<std::string, mx::array>
Qwen3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    if (config_.tie_word_embeddings) weights.erase("lm_head.weight");
    
    // Remove metadata and config entries
    std::vector<std::string> to_remove;
    for (auto& [k, _] : weights) {
        if (k.find("inv_freq") != std::string::npos ||
            k.find("__metadata__") != std::string::npos ||
            k.find("quantization") != std::string::npos)
            to_remove.push_back(k);
    }
    for (auto& k : to_remove) weights.erase(k);

    // If this is a BitNet model (has_pre_norms), unpack U8 ternary weights to fp16.
    // Iterate over *_weight_scale entries (not U8 weights) to correctly derive
    // the matching weight key. A key like "...q_proj.weight_scale" pairs with
    // "...q_proj.weight" (U8 packed ternary).
    if (config_.has_pre_norms) {
        std::vector<std::string> scale_keys_to_remove;
        std::vector<std::pair<std::string, mx::array>> weights_to_replace;

        const std::string scale_suffix = ".weight_scale";
        for (auto& [key, val] : weights) {
            // Only process *.weight_scale entries
            if (key.size() <= scale_suffix.size() ||
                key.compare(key.size() - scale_suffix.size(), scale_suffix.size(), scale_suffix) != 0) {
                continue;
            }

            // Derive the matching weight key
            std::string prefix = key.substr(0, key.size() - scale_suffix.size());
            std::string weight_key = prefix + ".weight";

            auto w_it = weights.find(weight_key);
            if (w_it == weights.end() || w_it->second.dtype() != mx::uint8) {
                continue;
            }

            // Dequantize: unpack U8 with 4 ternary values per byte to fp16
            // U8 shape: [out/4, in] -> unpacked: [out, in]
            auto shape = w_it->second.shape();
            int packed_rows = shape[0];  // out/4
            int in_features = shape[1];
            int out_features = packed_rows * 4;

            // Extract 4 ternary codes per byte: bits [1:0], [3:2], [5:4], [7:6]
            auto codes = mx::astype(w_it->second, mx::int32);
            auto v0 = mx::bitwise_and(codes, mx::array(0x03));
            auto v1 = mx::bitwise_and(mx::right_shift(codes, mx::array(2)), mx::array(0x03));
            auto v2 = mx::bitwise_and(mx::right_shift(codes, mx::array(4)), mx::array(0x03));
            auto v3 = mx::bitwise_and(mx::right_shift(codes, mx::array(6)), mx::array(0x03));

            // Concatenate along output dimension: [packed_rows, in] x4 -> [out, in]
            auto unpacked = mx::concatenate({v0, v1, v2, v3}, 0);

            // Map codes: 0->-1, 1->0, 2->+1, then scale
            auto ternary = mx::subtract(mx::astype(unpacked, mx::float16), mx::array(1.0f));

            // Read scale from weight_scale entry
            // bitlinear models use inverse scaling (actual_scale = 1/weight_scale)
            mx::eval(val);
            float scale_val = val.data<float>()[0];
            if (config_.bitnet_invert_weight_scales) {
                scale_val = 1.0f / scale_val;
            }
            auto scaled = mx::multiply(ternary, mx::array(scale_val));
            mx::eval(scaled);

            weights_to_replace.emplace_back(weight_key, std::move(scaled));
            scale_keys_to_remove.push_back(key);
        }

        for (auto& [k, v] : weights_to_replace) {
            auto it = weights.find(k);
            if (it != weights.end()) it->second = std::move(v);
        }
        for (auto& k : scale_keys_to_remove) weights.erase(k);
    }
    
    return weights;
}

void Qwen3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Qwen3Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
