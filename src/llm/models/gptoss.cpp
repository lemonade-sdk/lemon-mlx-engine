// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of GPTOSS.swift

#include <mlx-lm/llm/models/gptoss.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

void from_json(const nlohmann::json& j, GPTOSSConfiguration& c) {
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_local_experts = j.at("num_local_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.head_dim = j.at("head_dim").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.sliding_window = j.at("sliding_window").get<int>();
    c.rope_theta = j.value("rope_theta", 150000.0f);
    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }
    if (j.contains("layer_types") && !j["layer_types"].is_null()) {
        c.layer_types = j["layer_types"].get<std::vector<std::string>>();
    } else {
        // Default: alternating sliding/full
        c.layer_types.clear();
        for (int i = 0; i < c.num_hidden_layers; ++i) {
            c.layer_types.push_back(i % 2 == 0 ? "sliding_attention" : "full_attention");
        }
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// SwiGLU activation: clipped gating
static mx::array swiglu(const mx::array& x_linear, const mx::array& x_glu,
                          float alpha = 1.702f, float limit = 7.0f) {
    auto xlin = mx::clip(x_linear, mx::array(-limit), mx::array(limit));
    auto xg = mx::clip(x_glu, std::nullopt, mx::array(limit));
    auto glu_scaled = mx::multiply(mx::array(alpha), xg);
    auto sig = mx::sigmoid(glu_scaled);
    auto out_glu = mx::multiply(xg, sig);
    return mx::multiply(out_glu, mx::add(xlin, mx::array(1.0f)));
}

// SwiGLU is already compiled in activations.h — no need to double-compile.

// --- GPTOSSAttention ---

GPTOSSAttention::GPTOSSAttention(const GPTOSSConfiguration& config)
    : head_dim_(config.head_dim),
      num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      scale_(1.0f / std::sqrt(static_cast<float>(config.head_dim))),
      rope_theta_(config.rope_theta),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim})),
      wq_bias_(mx::zeros({config.num_attention_heads * config.head_dim})),
      wk_bias_(mx::zeros({config.num_key_value_heads * config.head_dim})),
      wv_bias_(mx::zeros({config.num_key_value_heads * config.head_dim})),
      wo_bias_(mx::zeros({config.hidden_size}))
{}

mx::array GPTOSSAttention::operator()(const mx::array& x,
                                        const AttentionMask& mask,
                                        KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto q = linear_fwd(x, wq_weight_, &wq_bias_);
    auto k = linear_fwd(x, wk_weight_, &wk_bias_);
    auto v = linear_fwd(x, wv_weight_, &wv_bias_);

    q = mx::transpose(mx::reshape(q, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    k = mx::transpose(mx::reshape(k, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});
    v = mx::transpose(mx::reshape(v, {B, L, num_kv_heads_, head_dim_}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    q = mx::fast::rope(q, head_dim_, false, rope_theta_, 1.0f, offset);
    k = mx::fast::rope(k, head_dim_, false, rope_theta_, 1.0f, offset);

    if (cache) {
        auto [ck, cv] = cache->update(k, v);
        k = ck; v = cv;
    }

    auto output = sdpa(q, k, v, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> GPTOSSAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
        {"q_proj.bias", &wq_bias_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.bias", &wv_bias_}, {"o_proj.bias", &wo_bias_},
    };
}

// --- GPTOSSMLP ---

GPTOSSMLP::GPTOSSMLP(const GPTOSSConfiguration& config)
    : num_local_experts_(config.num_local_experts),
      num_experts_per_tok_(config.num_experts_per_tok),
      gate_proj_(config.hidden_size, config.intermediate_size, config.num_local_experts, true),
      up_proj_(config.hidden_size, config.intermediate_size, config.num_local_experts, true),
      down_proj_(config.intermediate_size, config.hidden_size, config.num_local_experts, true),
      router_weight_(mx::zeros({config.num_local_experts, config.hidden_size})),
      router_bias_(mx::zeros({config.num_local_experts}))
{}

mx::array GPTOSSMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, router_weight_, &router_bias_);

    // topk selection: negate, argpartition for top-k
    int k = num_experts_per_tok_;
    auto neg_g = mx::negative(g);
    auto part_inds = mx::argpartition(neg_g, k - 1, -1);
    auto inds = mx::slice(part_inds, {0, 0, 0}, {part_inds.shape(0), part_inds.shape(1), k});
    auto expert_vals = mx::take_along_axis(g, inds, -1);

    auto stop_inds = mx::stop_gradient(inds);
    auto expert_weights = mx::softmax(expert_vals, -1);

    // Expand dims for SwitchLinear
    auto x_exp = mx::expand_dims(mx::expand_dims(x, {-2}), {-3});

    bool do_sort = (inds.size() >= 64);
    mx::array idx = stop_inds;
    mx::array inverse_order = mx::array(0.0f);

    if (do_sort) {
        auto [sorted_x, sorted_idx, inv] = gather_sort(x_exp, stop_inds);
        x_exp = sorted_x;
        idx = sorted_idx;
        inverse_order = inv;
    }

    auto x_up = up_proj_(x_exp, idx, do_sort);
    auto x_gate = gate_proj_(x_exp, idx, do_sort);
    auto activated = swiglu(x_gate, x_up);
    auto result = down_proj_(activated, idx, do_sort);

    if (do_sort) {
        auto shape = stop_inds.shape();
        result = scatter_unsort(result, inverse_order, &shape);
    }

    result = mx::squeeze(result, -2);
    result = mx::multiply(result, mx::expand_dims(expert_weights, -1));
    return mx::sum(result, -2);
}

std::unordered_map<std::string, mx::array*> GPTOSSMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : gate_proj_.weight_map()) map["experts.gate_proj." + k] = v;
    for (auto& [k, v] : up_proj_.weight_map()) map["experts.up_proj." + k] = v;
    for (auto& [k, v] : down_proj_.weight_map()) map["experts.down_proj." + k] = v;
    map["router.weight"] = &router_weight_;
    map["router.bias"] = &router_bias_;
    return map;
}

// --- GPTOSSBlock ---

GPTOSSBlock::GPTOSSBlock(const GPTOSSConfiguration& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array GPTOSSBlock::operator()(const mx::array& x,
                                    const AttentionMask& mask,
                                    KVCache* cache) {
    auto r = self_attn_(mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::rms_norm(h, post_attention_layernorm_weight_, norm_eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> GPTOSSBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : self_attn_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;
    return map;
}

// --- GPTOSSModelInner ---

GPTOSSModelInner::GPTOSSModelInner(const GPTOSSConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps),
      layer_types_(config.layer_types),
      sliding_window_(config.sliding_window),
      swa_idx_(0), fa_idx_(0)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);

    for (int i = 0; i < static_cast<int>(layer_types_.size()); ++i) {
        if (layer_types_[i] == "sliding_attention") { swa_idx_ = i; break; }
    }
    for (int i = 0; i < static_cast<int>(layer_types_.size()); ++i) {
        if (layer_types_[i] == "full_attention") { fa_idx_ = i; break; }
    }
}

mx::array GPTOSSModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    AttentionMask fa_mask = create_attention_mask(h,
        cache && fa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[fa_idx_] : nullptr);

    AttentionMask swa_mask;
    if (swa_idx_ >= 0) {
        swa_mask = create_attention_mask(h,
            cache && swa_idx_ < static_cast<int>(cache->size()) ? &(*cache)[swa_idx_] : nullptr,
            sliding_window_);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        const auto& mask = (i < layer_types_.size() && layer_types_[i] == "sliding_attention") ? swa_mask : fa_mask;
        h = layers_[i](h, mask, lc);
    }

    return mx::fast::rms_norm(h, norm_weight_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> GPTOSSModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = &norm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- GPTOSSModel ---

GPTOSSModel::GPTOSSModel(const GPTOSSConfiguration& config)
    : config_(config), model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size}))
{
    kv_heads_.resize(config.num_hidden_layers, config.num_key_value_heads);
}

PrepareResult GPTOSSModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput GPTOSSModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array GPTOSSModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return linear_forward(out, lm_head_weight_);
}

std::vector<KVCache> GPTOSSModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    auto& lt = model_.layer_types();
    caches.reserve(lt.size());
    for (const auto& t : lt) {
        if (t == "full_attention") {
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            } else {
                caches.emplace_back(KVCacheSimple{});
            }
        } else {
            caches.emplace_back(RotatingKVCache(config_.sliding_window));
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
GPTOSSModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // If already in gate_proj.weight form, nothing to do
    bool has_gate_proj = false;
    for (auto& [k, v] : weights) {
        if (k.find("gate_proj.weight") != std::string::npos) { has_gate_proj = true; break; }
    }
    if (has_gate_proj) return weights;

    // Handle gate_up_proj split into gate_proj.weight and up_proj.weight
    std::unordered_map<std::string, mx::array> result;
    for (auto& [k, v] : weights) {
        if (k.find("gate_up_proj") != std::string::npos && k.find("bias") == std::string::npos) {
            // Split interleaved: even indices → gate_proj, odd → up_proj
            auto gate_key = k;
            auto up_key = k;
            auto pos = gate_key.find("gate_up_proj");
            gate_key.replace(pos, 12, "gate_proj.weight");
            up_key.replace(pos, 12, "up_proj.weight");

            // v shape: [..., interleaved, hidden] — stride(by: 2) for gate, stride(from: 1, by: 2) for up
            auto gate_w = mx::slice(v, {0, 0, 0}, {v.shape(0), v.shape(1), v.shape(2)}, {1, 2, 1});
            auto up_w = mx::slice(v, {0, 1, 0}, {v.shape(0), v.shape(1), v.shape(2)}, {1, 2, 1});
            result.insert_or_assign(gate_key, mx::contiguous(gate_w));
            result.insert_or_assign(up_key, mx::contiguous(up_w));
        } else if (k.find("gate_up_proj_bias") != std::string::npos) {
            auto gate_key = k;
            auto up_key = k;
            auto pos = gate_key.find("gate_up_proj_bias");
            gate_key.replace(pos, 17, "gate_proj.bias");
            up_key.replace(pos, 17, "up_proj.bias");

            auto gate_b = mx::slice(v, {0, 0}, {v.shape(0), v.shape(1)}, {1, 2});
            auto up_b = mx::slice(v, {0, 1}, {v.shape(0), v.shape(1)}, {1, 2});
            result.insert_or_assign(gate_key, mx::contiguous(gate_b));
            result.insert_or_assign(up_key, mx::contiguous(up_b));
        } else if (k.find("down_proj") != std::string::npos && k.find("bias") == std::string::npos) {
            auto new_key = k;
            auto pos = new_key.find("down_proj");
            new_key.replace(pos, 9, "down_proj.weight");
            result.insert_or_assign(new_key, mx::contiguous(v));
        } else if (k.find("down_proj_bias") != std::string::npos) {
            auto new_key = k;
            auto pos = new_key.find("down_proj_bias");
            new_key.replace(pos, 14, "down_proj.bias");
            result.insert_or_assign(new_key, mx::contiguous(v));
        } else {
            result.insert_or_assign(k, std::move(v));
        }
    }
    return result;
}

void GPTOSSModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> GPTOSSModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

} // namespace mlx_lm
