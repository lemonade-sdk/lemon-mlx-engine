// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of FalconH1.swift

#include <mlx-lm/llm/models/falcon_h1.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/ssm_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <algorithm>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Config ---

void from_json(const nlohmann::json& j, FalconH1Configuration& c) {
    auto get = [&](const char* key, auto& dst, auto def) {
        if (j.contains(key) && !j[key].is_null()) dst = j[key].get<std::decay_t<decltype(dst)>>();
        else dst = def;
    };
    auto get_bool = [&](const char* key, bool& dst, bool def) {
        if (j.contains(key)) dst = j[key].get<bool>(); else dst = def;
    };

    get_bool("attention_bias", c.attention_bias, false);
    get("attention_in_multiplier", c.attention_in_multiplier, 1.0f);
    get("attention_out_multiplier", c.attention_out_multiplier, 1.0f);
    get("embedding_multiplier", c.embedding_multiplier, 1.0f);
    get("head_dim", c.head_dim, 64);
    get("hidden_size", c.hidden_size, 4096);
    get("key_multiplier", c.key_multiplier, 1.0f);
    get("lm_head_multiplier", c.lm_head_multiplier, 1.0f);
    get_bool("mamba_conv_bias", c.mamba_conv_bias, true);
    get("mamba_d_conv", c.mamba_d_conv, 4);
    get("mamba_d_head", c.mamba_d_head, 64);
    get("mamba_d_ssm", c.mamba_d_ssm, 1536);
    get("mamba_d_state", c.mamba_d_state, 256);
    get("mamba_expand", c.mamba_expand, 2);
    get("mamba_n_groups", c.mamba_n_groups, 1);
    get("mamba_n_heads", c.mamba_n_heads, 128);
    get_bool("mamba_norm_before_gate", c.mamba_norm_before_gate, true);
    get_bool("mamba_proj_bias", c.mamba_proj_bias, false);
    get_bool("mamba_rms_norm", c.mamba_rms_norm, false);
    get_bool("mamba_use_mlp", c.mamba_use_mlp, true);
    get("max_position_embeddings", c.max_position_embeddings, 8192);
    get_bool("mlp_bias", c.mlp_bias, false);
    get("mlp_expansion_factor", c.mlp_expansion_factor, 8);
    get("num_attention_heads", c.num_attention_heads, 32);
    get("num_hidden_layers", c.num_hidden_layers, 32);
    get("num_key_value_heads", c.num_key_value_heads, 8);
    get_bool("projectors_bias", c.projectors_bias, false);
    get("rms_norm_eps", c.rms_norm_eps, 1e-5f);
    get_bool("rope_traditional", c.rope_traditional, false);
    get("rope_theta", c.rope_theta, 100000.0f);
    get("ssm_in_multiplier", c.ssm_in_multiplier, 1.0f);
    get("ssm_out_multiplier", c.ssm_out_multiplier, 1.0f);
    get_bool("tie_word_embeddings", c.tie_word_embeddings, false);
    get("vocab_size", c.vocab_size, 128000);

    if (j.contains("intermediate_size") && !j["intermediate_size"].is_null())
        c.intermediate_size = j["intermediate_size"].get<int>();

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null())
        c.rope_scaling = j["rope_scaling"].get<float>();

    if (j.contains("mlp_multipliers"))
        c.mlp_multipliers = j["mlp_multipliers"].get<std::vector<float>>();

    if (j.contains("ssm_multipliers"))
        c.ssm_multipliers = j["ssm_multipliers"].get<std::vector<float>>();
}

mx::array compute_mup_vector(const FalconH1Configuration& config) {
    int intermediate_size = config.mamba_d_ssm;
    int groups_time_state_size = config.mamba_n_groups * config.mamba_d_state;
    int num_heads = config.mamba_n_heads;

    std::vector<int> sizes = {
        intermediate_size, intermediate_size,
        groups_time_state_size, groups_time_state_size,
        num_heads
    };

    std::vector<mx::array> segments;
    for (size_t i = 0; i < sizes.size(); ++i) {
        float mult = (i < config.ssm_multipliers.size()) ? config.ssm_multipliers[i] : 1.0f;
        segments.push_back(mx::broadcast_to(mx::array(mult), {sizes[i]}));
    }

    return mx::concatenate(segments, 0);
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- FalconH1Attention ---

FalconH1Attention::FalconH1Attention(const FalconH1Configuration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      scale_(std::pow(static_cast<float>(config.head_dim), -0.5f)),
      rope_theta_(config.rope_theta),
      rope_scale_(config.rope_scaling.has_value() ? 1.0f / config.rope_scaling.value() : 1.0f),
      rope_traditional_(config.rope_traditional),
      use_bias_(config.attention_bias),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim, config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim, config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim}))
{
    if (use_bias_) {
        wq_bias_ = mx::zeros({config.num_attention_heads * config.head_dim});
        wk_bias_ = mx::zeros({config.num_key_value_heads * config.head_dim});
        wv_bias_ = mx::zeros({config.num_key_value_heads * config.head_dim});
        wo_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array FalconH1Attention::operator()(const mx::array& x,
                                         const AttentionMask& mask,
                                         KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, wq_bias_.has_value() ? &wq_bias_.value() : nullptr);
    auto keys = linear_fwd(x, wk_weight_, wk_bias_.has_value() ? &wk_bias_.value() : nullptr);
    auto values = linear_fwd(x, wv_weight_, wv_bias_.has_value() ? &wv_bias_.value() : nullptr);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);
    keys = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, rope_scale_, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k; values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, wo_bias_.has_value() ? &wo_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> FalconH1Attention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_}, {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_}, {"o_proj.weight", &wo_weight_},
    };
    if (wq_bias_.has_value()) { map["q_proj.bias"] = &wq_bias_.value(); map["k_proj.bias"] = &wk_bias_.value(); }
    if (wv_bias_.has_value()) { map["v_proj.bias"] = &wv_bias_.value(); map["o_proj.bias"] = &wo_bias_.value(); }
    return map;
}

// --- RMSNormGated ---

RMSNormGated::RMSNormGated(int hidden_size, float eps, bool norm_before_gate)
    : weight_(mx::ones({hidden_size})), eps_(eps), norm_before_gate_(norm_before_gate)
{}

mx::array RMSNormGated::operator()(const mx::array& x, const std::optional<mx::array>& gate) {
    auto h = x;
    if (!norm_before_gate_ && gate.has_value()) {
        h = swiglu(gate.value(), h); // silu(gate) * h
    }
    h = mx::fast::rms_norm(h, weight_, eps_);
    if (norm_before_gate_ && gate.has_value()) {
        h = swiglu(gate.value(), h); // silu(gate) * h
    }
    return h;
}

std::unordered_map<std::string, mx::array*> RMSNormGated::weight_map() {
    return {{"weight", &weight_}};
}

// --- FalconH1Mixer ---

static int compute_conv_dim(const FalconH1Configuration& config) {
    return config.mamba_d_ssm + 2 * config.mamba_n_groups * config.mamba_d_state;
}

FalconH1Mixer::FalconH1Mixer(const FalconH1Configuration& config)
    : num_heads_(config.mamba_n_heads),
      hidden_size_(config.hidden_size),
      ssm_state_size_(config.mamba_d_state),
      conv_kernel_size_(config.mamba_d_conv),
      intermediate_size_(config.mamba_d_ssm),
      use_conv_bias_(config.mamba_conv_bias),
      use_bias_(config.mamba_proj_bias),
      groups_time_state_size_(config.mamba_n_groups * config.mamba_d_state),
      n_groups_(config.mamba_n_groups),
      head_dim_(config.mamba_d_head),
      conv_dim_(compute_conv_dim(config)),
      mamba_rms_norm_(config.mamba_rms_norm),
      in_proj_weight_(mx::zeros({config.mamba_d_ssm + compute_conv_dim(config) + config.mamba_n_heads, config.hidden_size})),
      conv1d_weight_(mx::zeros({compute_conv_dim(config), config.mamba_d_conv, 1})),
      dt_bias_(mx::ones({config.mamba_n_heads})),
      A_log_(mx::zeros({config.mamba_n_heads})),
      D_(mx::ones({config.mamba_n_heads})),
      out_proj_weight_(mx::zeros({config.hidden_size, config.mamba_d_ssm})),
      ssm_in_multiplier_(config.ssm_in_multiplier),
      rms_norm_eps_(config.rms_norm_eps)
{
    if (use_bias_) in_proj_bias_ = mx::zeros({intermediate_size_ + conv_dim_ + num_heads_});
    if (use_conv_bias_) conv1d_bias_ = mx::zeros({conv_dim_});
    if (config.projectors_bias) out_proj_bias_ = mx::zeros({hidden_size_});

    if (mamba_rms_norm_) {
        norm_.emplace(intermediate_size_, rms_norm_eps_, config.mamba_norm_before_gate);
    }
}

mx::array FalconH1Mixer::apply_conv(const mx::array& conv_input, MambaCache* mc) {
    auto conv_state = (mc && (*mc)[0].has_value())
        ? (*mc)[0].value()
        : mx::zeros({conv_input.shape(0), conv_kernel_size_ - 1, conv_dim_}, conv_input.dtype());

    auto padded = mx::concatenate({conv_state, conv_input}, 1);

    if (mc) {
        int start = padded.shape(1) - (conv_kernel_size_ - 1);
        (*mc)[0] = mx::slice(padded, {0, start, 0}, {padded.shape(0), padded.shape(1), padded.shape(2)});
    }

    auto conv_out = mx::conv1d(padded, conv1d_weight_, 1, 0, 1, conv_dim_);
    if (conv1d_bias_.has_value()) conv_out = mx::add(conv_out, conv1d_bias_.value());

    return silu(conv_out);
}

std::pair<mx::array, mx::array>
FalconH1Mixer::ssm(const mx::array& hidden_states,
                     const mx::array& B_ssm,
                     const mx::array& C_ssm,
                     const mx::array& dt,
                     const std::optional<mx::array>& state) {
    int b = hidden_states.shape(0), l = hidden_states.shape(1);

    auto h_reshaped = mx::reshape(hidden_states, {b, l, num_heads_, head_dim_});
    auto B_reshaped = mx::reshape(B_ssm, {b, l, n_groups_, ssm_state_size_});
    auto C_reshaped = mx::reshape(C_ssm, {b, l, n_groups_, ssm_state_size_});

    auto [y, new_state] = ssm_update(h_reshaped, A_log_, B_reshaped, C_reshaped, D_, dt, dt_bias_,
                                      state, 0.0f, std::numeric_limits<float>::max());

    return {mx::reshape(y, {b, l, intermediate_size_}), new_state};
}

mx::array FalconH1Mixer::operator()(const mx::array& x, KVCache* cache) {
    auto projected = linear_fwd(x, in_proj_weight_, in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);

    // Split: gate | conv_input | dt
    auto gate = mx::slice(projected, {0, 0, 0}, {projected.shape(0), projected.shape(1), intermediate_size_});
    auto conv_input = mx::slice(projected, {0, 0, intermediate_size_},
                                {projected.shape(0), projected.shape(1), intermediate_size_ + conv_dim_});
    auto dt = mx::slice(projected, {0, 0, intermediate_size_ + conv_dim_},
                        {projected.shape(0), projected.shape(1), projected.shape(2)});

    auto* mc = cache ? cache->as_mamba() : nullptr;
    auto conv_output = apply_conv(conv_input, mc);

    // Split conv output: hidden_states | B | C
    auto hidden_ssm = mx::slice(conv_output, {0, 0, 0},
                                {conv_output.shape(0), conv_output.shape(1), intermediate_size_});
    auto B_ssm = mx::slice(conv_output, {0, 0, intermediate_size_},
                           {conv_output.shape(0), conv_output.shape(1), intermediate_size_ + n_groups_ * ssm_state_size_});
    auto C_ssm = mx::slice(conv_output, {0, 0, intermediate_size_ + n_groups_ * ssm_state_size_},
                           {conv_output.shape(0), conv_output.shape(1), conv_output.shape(2)});

    std::optional<mx::array> ssm_state;
    if (mc && (*mc)[1].has_value()) ssm_state = (*mc)[1].value();

    auto [y, new_ssm_state] = ssm(hidden_ssm, B_ssm, C_ssm, dt, ssm_state);

    if (mc) (*mc)[1] = new_ssm_state;

    // Apply norm or gate
    if (norm_.has_value()) {
        y = (*norm_)(y, gate);
    } else {
        y = swiglu(gate, y); // silu(gate) * y
    }

    return linear_fwd(y, out_proj_weight_, out_proj_bias_.has_value() ? &out_proj_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> FalconH1Mixer::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"in_proj.weight", &in_proj_weight_},
        {"conv1d.weight", &conv1d_weight_},
        {"dt_bias", &dt_bias_},
        {"A_log", &A_log_},
        {"D", &D_},
        {"out_proj.weight", &out_proj_weight_},
    };
    if (in_proj_bias_.has_value()) map["in_proj.bias"] = &in_proj_bias_.value();
    if (conv1d_bias_.has_value()) map["conv1d.bias"] = &conv1d_bias_.value();
    if (out_proj_bias_.has_value()) map["out_proj.bias"] = &out_proj_bias_.value();
    if (norm_.has_value()) {
        for (auto& [k, v] : norm_->weight_map()) map["norm." + k] = v;
    }
    return map;
}

// --- FalconH1MLP ---

static int falcon_h1_intermediate(const FalconH1Configuration& config) {
    return config.intermediate_size.value_or(4 * config.hidden_size);
}

FalconH1MLP::FalconH1MLP(const FalconH1Configuration& config)
    : gate_weight_(mx::zeros({falcon_h1_intermediate(config), config.hidden_size})),
      up_weight_(mx::zeros({falcon_h1_intermediate(config), config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, falcon_h1_intermediate(config)}))
{
    if (config.mlp_bias) {
        int intermediate = falcon_h1_intermediate(config);
        gate_bias_ = mx::zeros({intermediate});
        up_bias_ = mx::zeros({intermediate});
        down_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array FalconH1MLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_, gate_bias_.has_value() ? &gate_bias_.value() : nullptr);
    auto u = linear_fwd(x, up_weight_, up_bias_.has_value() ? &up_bias_.value() : nullptr);
    auto activated = swiglu(g, u); // silu(gate) * up
    return linear_fwd(activated, down_weight_, down_bias_.has_value() ? &down_bias_.value() : nullptr);
}

std::unordered_map<std::string, mx::array*> FalconH1MLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
    if (gate_bias_.has_value()) map["gate_proj.bias"] = &gate_bias_.value();
    if (up_bias_.has_value()) map["up_proj.bias"] = &up_bias_.value();
    if (down_bias_.has_value()) map["down_proj.bias"] = &down_bias_.value();
    return map;
}

// --- FalconH1DecoderLayer ---

FalconH1DecoderLayer::FalconH1DecoderLayer(const FalconH1Configuration& config)
    : feed_forward_(config),
      mamba_(config),
      attention_(config),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      pre_ff_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{}

mx::array FalconH1DecoderLayer::operator()(const mx::array& h,
                                            const AttentionMask& attn_mask,
                                            KVCache* cache) {
    auto normed = mx::fast::rms_norm(h, input_layernorm_weight_, norm_eps_);

    // Parallel mamba + attention
    auto mamba_h = mamba_(normed, cache);
    auto attn_h = attention_(normed, attn_mask, cache);

    auto residual = mx::add(mx::add(h, mamba_h), attn_h);

    auto ff_normed = mx::fast::rms_norm(residual, pre_ff_layernorm_weight_, norm_eps_);
    return mx::add(residual, feed_forward_(ff_normed));
}

std::unordered_map<std::string, mx::array*> FalconH1DecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : mamba_.weight_map()) map["mamba." + k] = v;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : feed_forward_.weight_map()) map["feed_forward." + k] = v;
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["pre_ff_layernorm.weight"] = &pre_ff_layernorm_weight_;
    return map;
}

// --- FalconH1ModelInner ---

FalconH1ModelInner::FalconH1ModelInner(const FalconH1Configuration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      final_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array FalconH1ModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    auto attn_mask = create_attention_mask(h,
        (cache && !cache->empty()) ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, attn_mask, lc);
    }

    return mx::fast::rms_norm(h, final_layernorm_weight_, norm_eps_);
}

std::unordered_map<std::string, mx::array*> FalconH1ModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["final_layernorm.weight"] = &final_layernorm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- FalconH1Model ---

FalconH1Model::FalconH1Model(const FalconH1Configuration& config)
    : config_(config),
      model_(config_),
      lm_head_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      mup_vector_(compute_mup_vector(config))
{}

PrepareResult FalconH1Model::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput FalconH1Model::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array FalconH1Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);
    return linear_fwd(out, lm_head_weight_);
}

std::vector<KVCache> FalconH1Model::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        MambaCache mc;
        if (params.max_kv_size.has_value()) {
            caches.emplace_back(CompoundCache(std::move(mc), RotatingKVCache(params.max_kv_size.value(), 4)));
        } else {
            caches.emplace_back(CompoundCache(std::move(mc), KVCacheSimple{}));
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array>
FalconH1Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Check if we need to apply muP scaling
    auto conv_it = weights.find("model.layers.0.mamba.conv1d.weight");
    if (conv_it == weights.end() || conv_it->second.shape(-1) <= conv_it->second.shape(1)) {
        return weights;
    }

    std::unordered_map<std::string, mx::array> sanitized;
    auto mup_vec = mup_vector_;

    for (auto& [name, param] : weights) {
        auto tensor = std::move(param);

        if (name.find("embed_tokens.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.embedding_multiplier));
        } else if (name.find("lm_head.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.lm_head_multiplier));
        } else if (name.find("q_proj.weight") != std::string::npos ||
                   name.find("k_proj.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.attention_in_multiplier));
        } else if (name.find("o_proj.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.attention_out_multiplier));
        } else if (name.find("out_proj.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.ssm_out_multiplier));
        } else if (name.find("gate_proj.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.mlp_multipliers[0]));
        } else if (name.find("down_proj.weight") != std::string::npos) {
            tensor = mx::multiply(tensor, mx::array(config_.mlp_multipliers.size() > 1 ? config_.mlp_multipliers[1] : 1.0f));
        } else if (name.find("in_proj.weight") != std::string::npos) {
            auto scale = mx::multiply(mx::array(config_.ssm_in_multiplier),
                                       mx::expand_dims(mx::astype(mup_vec, tensor.dtype()), -1));
            tensor = mx::multiply(tensor, scale);
        } else if (name.find("conv1d.weight") != std::string::npos) {
            tensor = mx::transpose(tensor, {0, 2, 1});
        }

        sanitized.insert_or_assign(name, std::move(tensor));
    }

    return sanitized;
}

void FalconH1Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> FalconH1Model::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

} // namespace mlx_lm
