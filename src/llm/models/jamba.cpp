// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Jamba.swift

#include <mlx-lm/llm/models/jamba.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/activations.h>
#include <cmath>
#include <algorithm>
#include <mlx-lm/common/quantized_linear.h>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Config ---

void from_json(const nlohmann::json& j, JambaConfiguration& c) {
    c.model_type = j.at("model_type").get<std::string>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.attn_layer_offset = j.at("attn_layer_offset").get<int>();
    c.attn_layer_period = j.at("attn_layer_period").get<int>();
    c.expert_layer_offset = j.at("expert_layer_offset").get<int>();
    c.expert_layer_period = j.at("expert_layer_period").get<int>();
    c.mamba_d_conv = j.at("mamba_d_conv").get<int>();
    c.mamba_d_state = j.at("mamba_d_state").get<int>();
    c.mamba_expand = j.at("mamba_expand").get<int>();
    c.num_experts = j.at("num_experts").get<int>();
    c.num_experts_per_tok = j.at("num_experts_per_tok").get<int>();
    c.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    c.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();

    // Optional fields
    if (j.contains("mamba_proj_bias")) c.mamba_proj_bias = j["mamba_proj_bias"].get<bool>();
    if (j.contains("mamba_conv_bias")) c.mamba_conv_bias = j["mamba_conv_bias"].get<bool>();
    if (j.contains("tie_word_embeddings")) c.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();

    // mamba_dt_rank: compute if absent
    if (j.contains("mamba_dt_rank") && !j["mamba_dt_rank"].is_null()) {
        c.mamba_dt_rank = j["mamba_dt_rank"].get<int>();
    } else {
        c.mamba_dt_rank = static_cast<int>(std::ceil(static_cast<double>(c.hidden_size) / 16.0));
    }

    // layers_block_type: generate if absent
    if (j.contains("layers_block_type") && !j["layers_block_type"].is_null()) {
        c.layers_block_type = j["layers_block_type"].get<std::vector<std::string>>();
    } else {
        c.layers_block_type.resize(c.num_hidden_layers);
        for (int i = 0; i < c.num_hidden_layers; ++i) {
            c.layers_block_type[i] =
                (i % c.attn_layer_period == c.attn_layer_offset) ? "attention" : "mamba";
        }
    }
}

static mx::array linear_fwd(const mx::array& x, const mx::array& w, const mx::array* bias = nullptr) {
    return linear_forward(x, w, bias);
}

// --- JambaMLP ---

JambaMLP::JambaMLP(const JambaConfiguration& config)
    : gate_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      up_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      down_weight_(mx::zeros({config.hidden_size, config.intermediate_size}))
{}

mx::array JambaMLP::operator()(const mx::array& x) {
    auto g = linear_fwd(x, gate_weight_);
    return linear_fwd(swiglu(g, linear_fwd(x, up_weight_)), down_weight_);
}

std::unordered_map<std::string, mx::array*> JambaMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"up_proj.weight", &up_weight_},
        {"down_proj.weight", &down_weight_},
    };
}

// --- JambaAttention ---

JambaAttention::JambaAttention(const JambaConfiguration& config)
    : num_heads_(config.num_attention_heads),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim()),
      scale_(std::pow(static_cast<float>(config.head_dim()), -0.5f)),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim()}))
{}

mx::array JambaAttention::operator()(const mx::array& x,
                                      const AttentionMask& mask,
                                      KVCache* cache) {
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys = linear_fwd(x, wk_weight_);
    auto values = linear_fwd(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, -1}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, num_kv_heads_, -1}), {0, 2, 1, 3});

    auto output = attention_with_cache_update(queries, keys, values, cache, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> JambaAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
}

// --- JambaMambaMixer ---

JambaMambaMixer::JambaMambaMixer(const JambaConfiguration& config)
    : hidden_size_(config.hidden_size),
      ssm_state_size_(config.mamba_d_state),
      conv_kernel_size_(config.mamba_d_conv),
      intermediate_size_(config.mamba_expand * config.hidden_size),
      time_step_rank_(config.mamba_dt_rank),
      use_conv_bias_(config.mamba_conv_bias),
      use_bias_(config.mamba_proj_bias),
      in_proj_weight_(mx::zeros({config.mamba_expand * config.hidden_size * 2, config.hidden_size})),
      conv1d_weight_(mx::zeros({config.mamba_expand * config.hidden_size, config.mamba_d_conv, 1})),
      x_proj_weight_(mx::zeros({config.mamba_dt_rank + config.mamba_d_state * 2, config.mamba_expand * config.hidden_size})),
      dt_proj_weight_(mx::zeros({config.mamba_expand * config.hidden_size, config.mamba_dt_rank})),
      dt_proj_bias_(mx::zeros({config.mamba_expand * config.hidden_size})),
      A_log_(mx::zeros({config.mamba_expand * config.hidden_size, config.mamba_d_state})),
      D_(mx::ones({config.mamba_expand * config.hidden_size})),
      out_proj_weight_(mx::zeros({config.hidden_size, config.mamba_expand * config.hidden_size})),
      dt_layernorm_weight_(mx::ones({config.mamba_dt_rank})),
      b_layernorm_weight_(mx::ones({config.mamba_d_state})),
      c_layernorm_weight_(mx::ones({config.mamba_d_state})),
      norm_eps_(config.rms_norm_eps)
{
    if (use_bias_) {
        in_proj_bias_ = mx::zeros({intermediate_size_ * 2});
        out_proj_bias_ = mx::zeros({hidden_size_});
    }
    if (use_conv_bias_) {
        conv1d_bias_ = mx::zeros({intermediate_size_});
    }
}

std::pair<mx::array, mx::array>
JambaMambaMixer::ssm_step(const mx::array& x, const mx::array& A,
                           const std::optional<mx::array>& state) {
    int T = x.shape(1);

    // x_proj: x → [delta, B, C]
    auto delta_bc = linear_fwd(x, x_proj_weight_);
    auto delta = mx::slice(delta_bc, {0, 0, 0},
                           {delta_bc.shape(0), delta_bc.shape(1), time_step_rank_});
    auto B_part = mx::slice(delta_bc, {0, 0, time_step_rank_},
                            {delta_bc.shape(0), delta_bc.shape(1), time_step_rank_ + ssm_state_size_});
    auto C_part = mx::slice(delta_bc, {0, 0, time_step_rank_ + ssm_state_size_},
                            {delta_bc.shape(0), delta_bc.shape(1), time_step_rank_ + ssm_state_size_ * 2});

    // Layer norms
    delta = mx::fast::rms_norm(delta, dt_layernorm_weight_, norm_eps_);
    B_part = mx::fast::rms_norm(B_part, b_layernorm_weight_, norm_eps_);
    C_part = mx::fast::rms_norm(C_part, c_layernorm_weight_, norm_eps_);

    // dt_proj + softplus
    delta = mx::log(mx::add(mx::exp(linear_fwd(delta, dt_proj_weight_, &dt_proj_bias_)), mx::array(1.0f)));

    // new_state = expand_dims(delta * x, -1) * expand_dims(B, -2)
    auto new_state = mx::multiply(mx::expand_dims(mx::multiply(delta, x), -1),
                                   mx::expand_dims(B_part, -2));

    // dtA = exp(expand_dims(delta, -1) * A)
    auto dtA = mx::exp(mx::multiply(mx::expand_dims(delta, -1), A));

    // Recurrence over time steps
    std::optional<mx::array> current_state = state;
    for (int t = 0; t < T; ++t) {
        // new_state[:, t] slice
        auto ns_t = mx::slice(new_state, {0, t, 0, 0},
                              {new_state.shape(0), t + 1, new_state.shape(2), new_state.shape(3)});
        auto dta_t = mx::slice(dtA, {0, t, 0, 0},
                               {dtA.shape(0), t + 1, dtA.shape(2), dtA.shape(3)});

        if (current_state.has_value()) {
            auto updated = mx::add(mx::multiply(current_state.value(), dta_t), ns_t);
            // Write back into new_state at position t
            new_state = mx::scatter(new_state, mx::array({t}, {1}, mx::int32), updated, 1);
            current_state = updated;
        } else {
            current_state = ns_t;
        }
    }

    // y = (new_state @ expand_dims(C, -1)).squeeze(-1)
    auto y = mx::squeeze(mx::matmul(new_state, mx::expand_dims(C_part, -1)), -1);

    // y + D * x
    y = mx::add(y, mx::multiply(D_, x));

    // Return last state: new_state[:, -1]
    auto last_state = mx::slice(new_state, {0, T - 1, 0, 0},
                                {new_state.shape(0), T, new_state.shape(2), new_state.shape(3)});
    return {y, last_state};
}

mx::array JambaMambaMixer::operator()(const mx::array& x, KVCache* cache) {
    // in_proj → split into x_part and z
    auto xz = linear_fwd(x, in_proj_weight_, in_proj_bias_.has_value() ? &in_proj_bias_.value() : nullptr);
    auto x_part = mx::slice(xz, {0, 0, 0}, {xz.shape(0), xz.shape(1), intermediate_size_});
    auto z = mx::slice(xz, {0, 0, intermediate_size_}, {xz.shape(0), xz.shape(1), intermediate_size_ * 2});

    // Conv1d with state
    int K = conv_kernel_size_;

    std::optional<mx::array> conv_state;
    if (cache) {
        auto* mc = cache->as_mamba();
        if (mc && (*mc)[0].has_value()) {
            conv_state = (*mc)[0].value();
        }
    }

    mx::array x_full = conv_state.has_value()
        ? mx::concatenate({conv_state.value(), x_part}, 1)
        : mx::pad(x_part, {{0, 0}, {K - 1, 0}, {0, 0}});

    // Depthwise conv1d
    auto conv_out = mx::conv1d(x_full, conv1d_weight_, /*stride=*/1, /*padding=*/0, /*dilation=*/1, intermediate_size_);
    if (conv1d_bias_.has_value()) {
        conv_out = mx::add(conv_out, conv1d_bias_.value());
    }

    // Save new conv state: last (K-1) timesteps
    mx::array new_conv_state = mx::slice(x_full, {0, x_full.shape(1) - (K - 1), 0},
                                         {x_full.shape(0), x_full.shape(1), x_full.shape(2)});

    // silu(conv_out)
    auto x_conv = silu(conv_out);

    // SSM
    auto A = mx::negative(mx::exp(A_log_));

    std::optional<mx::array> ssm_state;
    if (cache) {
        auto* mc = cache->as_mamba();
        if (mc && (*mc)[1].has_value()) {
            ssm_state = (*mc)[1].value();
        }
    }

    auto [y, new_ssm_state] = ssm_step(x_conv, A, ssm_state);

    // output = out_proj(silu(z) * y)
    auto output = linear_fwd(swiglu(z, y), out_proj_weight_,
                              out_proj_bias_.has_value() ? &out_proj_bias_.value() : nullptr);

    // Update cache
    if (cache) {
        auto* mc = cache->as_mamba();
        if (mc) {
            (*mc)[0] = new_conv_state;
            (*mc)[1] = new_ssm_state;
        }
    }

    return output;
}

std::unordered_map<std::string, mx::array*> JambaMambaMixer::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"in_proj.weight", &in_proj_weight_},
        {"conv1d.weight", &conv1d_weight_},
        {"x_proj.weight", &x_proj_weight_},
        {"dt_proj.weight", &dt_proj_weight_},
        {"dt_proj.bias", &dt_proj_bias_},
        {"A_log", &A_log_},
        {"D", &D_},
        {"out_proj.weight", &out_proj_weight_},
        {"dt_layernorm.weight", &dt_layernorm_weight_},
        {"b_layernorm.weight", &b_layernorm_weight_},
        {"c_layernorm.weight", &c_layernorm_weight_},
    };
    if (in_proj_bias_.has_value()) map["in_proj.bias"] = &in_proj_bias_.value();
    if (conv1d_bias_.has_value()) map["conv1d.bias"] = &conv1d_bias_.value();
    if (out_proj_bias_.has_value()) map["out_proj.bias"] = &out_proj_bias_.value();
    return map;
}

// --- JambaSparseMoeBlock ---

JambaSparseMoeBlock::JambaSparseMoeBlock(const JambaConfiguration& config)
    : num_experts_per_tok_(config.num_experts_per_tok),
      router_weight_(mx::zeros({config.num_experts, config.hidden_size})),
      switch_mlp_(config.hidden_size, config.intermediate_size, config.num_experts)
{}

mx::array JambaSparseMoeBlock::operator()(const mx::array& x) {
    auto gates = linear_fwd(x, router_weight_);
    int k = num_experts_per_tok_;

    auto inds = mx::stop_gradient(mx::argpartition(mx::negative(gates), k - 1, -1));
    inds = mx::slice(inds, {0, 0, 0}, {inds.shape(0), inds.shape(1), k});
    auto scores = mx::take_along_axis(gates, inds, -1);
    scores = mx::softmax(scores, -1);

    auto y = switch_mlp_(x, inds);
    auto weighted = mx::multiply(y, mx::expand_dims(scores, -1));
    return mx::sum(weighted, -2);
}

std::unordered_map<std::string, mx::array*> JambaSparseMoeBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["router.weight"] = &router_weight_;
    for (auto& [k, v] : switch_mlp_.weight_map()) map["switch_mlp." + k] = v;
    return map;
}

// --- JambaDecoderLayer ---

JambaDecoderLayer::JambaDecoderLayer(const JambaConfiguration& config, const std::string& layer_type)
    : is_attn_(layer_type == "attention"),
      is_sparse_moe_(config.num_experts > 1),
      input_layernorm_weight_(mx::ones({config.hidden_size})),
      pre_ff_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps)
{
    if (is_attn_) {
        self_attn_.emplace(config);
    } else {
        mamba_.emplace(config);
    }

    if (is_sparse_moe_) {
        moe_.emplace(config);
    } else {
        mlp_.emplace(config);
    }
}

mx::array JambaDecoderLayer::operator()(const mx::array& x,
                                         const AttentionMask& mask,
                                         KVCache* cache) {
    auto normed = mx::fast::rms_norm(x, input_layernorm_weight_, norm_eps_);

    auto h = is_attn_
        ? (*self_attn_)(normed, mask, cache)
        : (*mamba_)(normed, cache);

    auto r = mx::add(x, h);
    auto ff_normed = mx::fast::rms_norm(r, pre_ff_layernorm_weight_, norm_eps_);

    return is_sparse_moe_
        ? mx::add(r, (*moe_)(ff_normed))
        : mx::add(r, (*mlp_)(ff_normed));
}

std::unordered_map<std::string, mx::array*> JambaDecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    if (is_attn_) {
        for (auto& [k, v] : self_attn_->weight_map()) map["self_attn." + k] = v;
    } else {
        for (auto& [k, v] : mamba_->weight_map()) map["mamba." + k] = v;
    }

    if (is_sparse_moe_) {
        for (auto& [k, v] : moe_->weight_map()) map["feed_forward." + k] = v;
    } else {
        for (auto& [k, v] : mlp_->weight_map()) map["feed_forward." + k] = v;
    }

    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["pre_ff_layernorm.weight"] = &pre_ff_layernorm_weight_;
    return map;
}

// --- JambaModelInner ---

JambaModelInner::JambaModelInner(const JambaConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      final_layernorm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.rms_norm_eps),
      attn_idx_(-1)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, config.layers_block_type[i]);
    }

    // Find first attention layer index (for mask creation)
    for (int i = 0; i < static_cast<int>(config.layers_block_type.size()); ++i) {
        if (config.layers_block_type[i] == "attention") {
            attn_idx_ = i;
            break;
        }
    }
}

mx::array JambaModelInner::operator()(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    // Create attention mask only if there are attention layers
    AttentionMask attn_mask;
    if (attn_idx_ >= 0) {
        KVCache* attn_cache = (cache && attn_idx_ < static_cast<int>(cache->size()))
            ? &(*cache)[attn_idx_] : nullptr;
        attn_mask = create_attention_mask(h, attn_cache);
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        if (layers_[i].is_attn()) {
            h = layers_[i](h, attn_mask, lc);
        } else {
            h = layers_[i](h, AttentionMask{}, lc);
        }
    }

    return mx::fast::rms_norm(h, final_layernorm_weight_, norm_eps_);
}

mx::array JambaModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> JambaModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["final_layernorm.weight"] = &final_layernorm_weight_;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// --- JambaModel ---

JambaModel::JambaModel(const JambaConfiguration& config)
    : config_(config), model_(config_)
{
    if (!config.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({config.vocab_size, config.hidden_size});
    }
}

PrepareResult JambaModel::prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int ws) {
    return llm_default_prepare(*this, input, cache, ws);
}

LMOutput JambaModel::call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*) {
    return LMOutput(forward_impl(input.tokens, cache));
}

mx::array JambaModel::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    auto out = model_(inputs, cache);

    if (config_.tie_word_embeddings) {
        return model_.embed_as_linear(out);
    } else {
        return linear_fwd(out, lm_head_weight_.value());
    }
}

std::vector<KVCache> JambaModel::new_cache_impl(const GenerateParameters& params) {
    std::vector<KVCache> caches;
    caches.reserve(config_.num_hidden_layers);

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        if (config_.layers_block_type[i] == "attention") {
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
JambaModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Handle conv1d weight reshaping
    std::unordered_map<std::string, mx::array> sanitized;
    for (auto& [k, v] : weights) {
        auto tensor = std::move(v);
        if (k.find("conv1d.weight") != std::string::npos && tensor.shape(-1) != 1) {
            // Move axis: [out, in, kernel] → [out, kernel, 1] for depthwise
            tensor = mx::moveaxis(tensor, 2, 1);
        }
        sanitized.insert_or_assign(k, std::move(tensor));
    }

    // Remove lm_head.weight if tied
    if (config_.tie_word_embeddings) {
        sanitized.erase("lm_head.weight");
    }

    // Stack per-expert weights into SwitchGLU format
    std::string check_key = "model.layers.0.feed_forward.experts.0.gate_proj.weight";
    if (sanitized.find(check_key) == sanitized.end()) {
        return sanitized;
    }

    for (int l = 0; l < config_.num_hidden_layers; ++l) {
        std::string prefix = "model.layers." + std::to_string(l) + ".feed_forward.";
        std::string expert_prefix = prefix + "experts.";

        for (const char* swift_name : {"gate_proj", "down_proj", "up_proj"}) {
            for (const char* suffix : {"weight", "scales", "biases"}) {
                std::string check = expert_prefix + "0." + swift_name + "." + suffix;
                if (sanitized.find(check) == sanitized.end()) continue;

                std::vector<mx::array> to_join;
                to_join.reserve(config_.num_experts);
                for (int e = 0; e < config_.num_experts; ++e) {
                    std::string ek = expert_prefix + std::to_string(e) + "." + swift_name + "." + suffix;
                    auto it = sanitized.find(ek);
                    to_join.push_back(std::move(it->second));
                    sanitized.erase(it);
                }
                std::string new_key = prefix + "switch_mlp." + swift_name + "." + suffix;
                sanitized.insert_or_assign(new_key, mx::stack(to_join));
            }
        }
    }

    return sanitized;
}

void JambaModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> JambaModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    if (lm_head_weight_.has_value()) map["lm_head.weight"] = &lm_head_weight_.value();
    return map;
}

} // namespace mlx_lm
