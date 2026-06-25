// BitNet 1.58-bit model implementation — Llama variant with ternary weights.
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/bitnet.py

#include <mlx-lm/llm/models/bitnet.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/bitnet_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Linear helper ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight)
{
    return linear_forward(x, weight, nullptr);
}

// --- BitNet Attention ---

BitNetAttention::BitNetAttention(const BitNetConfiguration& args)
    : args_(args),
      use_relu2_(args.hidden_act == "relu2"),
      has_sub_norm_(args.hidden_act == "relu2"),
      scale_(std::pow(static_cast<float>(args.resolved_head_dim()), -0.5f)),
      wq_weight_(mx::zeros({args.num_attention_heads * args.resolved_head_dim(), args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.resolved_head_dim(), args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.resolved_head_dim()})),
      attn_sub_norm_weight_(mx::ones({args.hidden_size})),
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
{}

mx::array BitNetAttention::linear(const mx::array& x, const mx::array& weight) const {
    return linear_fwd(x, weight);
}

mx::array BitNetAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);
    int head_dim = args_.resolved_head_dim();

    auto queries = linear(x, wq_weight_);
    auto keys = linear(x, wk_weight_);
    auto values = linear(x, wv_weight_);

    queries = mx::transpose(mx::reshape(queries, {B, L, args_.num_attention_heads, head_dim}), {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {B, L, args_.num_key_value_heads, head_dim}), {0, 2, 1, 3});

    int offset = cache ? cache->offset() : 0;
    queries = rope_(queries, offset);
    keys = rope_(keys, offset);

    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);

    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});

    // BitNet: sub-layer norm before output projection (only for true BitNet models)
    if (has_sub_norm_) {
        output = mx::fast::rms_norm(output, attn_sub_norm_weight_, args_.rms_norm_eps);
    }

    return linear(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> BitNetAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    if (has_sub_norm_) {
        map["attn_sub_norm.weight"] = &attn_sub_norm_weight_;
    }
    return map;
}

// --- BitNet MLP (relu² + sub-layer norm) ---

BitNetMLP::BitNetMLP(const BitNetConfiguration& args)
    : use_relu2_(args.hidden_act == "relu2"),
      has_sub_norm_(args.hidden_act == "relu2"),
      gate_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      down_weight_(mx::zeros({args.hidden_size, args.intermediate_size})),
      up_weight_(mx::zeros({args.intermediate_size, args.hidden_size})),
      ffn_sub_norm_weight_(mx::ones({args.intermediate_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array BitNetMLP::linear(const mx::array& x, const mx::array& weight) const {
    return linear_fwd(x, weight);
}

mx::array BitNetMLP::rms_norm(const mx::array& x, const mx::array& weight) const {
    return mx::fast::rms_norm(x, weight, rms_norm_eps_);
}

mx::array BitNetMLP::operator()(const mx::array& x) {
    auto gate_out = linear(x, gate_weight_);
    auto gate = use_relu2_ ? relu_squared(gate_out) : silu(gate_out);
    auto up = linear(x, up_weight_);
    auto hidden = mx::multiply(gate, up);

    if (has_sub_norm_) {
        hidden = rms_norm(hidden, ffn_sub_norm_weight_);
    }

    return linear(hidden, down_weight_);
}

std::unordered_map<std::string, mx::array*> BitNetMLP::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
    if (has_sub_norm_) {
        map["ffn_sub_norm.weight"] = &ffn_sub_norm_weight_;
    }
    return map;
}

// --- BitNet Transformer Block ---

BitNetTransformerBlock::BitNetTransformerBlock(const BitNetConfiguration& args)
    : attention_(args),
      mlp_(args),
      input_layernorm_weight_(mx::ones({args.hidden_size})),
      post_attention_layernorm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{}

mx::array BitNetTransformerBlock::rms_norm(const mx::array& x, const mx::array& weight) const {
    return mx::fast::rms_norm(x, weight, rms_norm_eps_);
}

mx::array BitNetTransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    auto r = attention_(rms_norm(x, input_layernorm_weight_), mask, cache);
    auto h = mx::add(x, r);
    r = mlp_(rms_norm(h, post_attention_layernorm_weight_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> BitNetTransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;

    for (auto& [k, v] : attention_.weight_map()) {
        map["self_attn." + k] = v;
    }
    for (auto& [k, v] : mlp_.weight_map()) {
        map["mlp." + k] = v;
    }
    map["input_layernorm.weight"] = &input_layernorm_weight_;
    map["post_attention_layernorm.weight"] = &post_attention_layernorm_weight_;

    return map;
}

// --- BitNet Model Inner ---

BitNetModelInner::BitNetModelInner(const BitNetConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_weight_(mx::ones({args.hidden_size})),
      rms_norm_eps_(args.rms_norm_eps)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i) {
        layers_.emplace_back(args);
    }
}

mx::array BitNetModelInner::rms_norm(const mx::array& x, const mx::array& weight) const {
    return mx::fast::rms_norm(x, weight, rms_norm_eps_);
}

mx::array BitNetModelInner::operator()(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto h = mx::take(embed_tokens_weight_, inputs, 0);

    auto mask = create_attention_mask(h, cache && !cache->empty() ? &(*cache)[0] : nullptr);

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* layer_cache = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, layer_cache);
    }

    return rms_norm(h, norm_weight_);
}

mx::array BitNetModelInner::embed_as_linear(const mx::array& x) const {
    return linear_forward(x, embed_tokens_weight_);
}

std::unordered_map<std::string, mx::array*> BitNetModelInner::weight_map() {
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

// --- BitNet Model (top-level) ---

BitNetModel::BitNetModel(const BitNetConfiguration& args)
    : config_(args), model_(config_)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);

    if (!args.tie_word_embeddings) {
        lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
    }
}

PrepareResult BitNetModel::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int window_size)
{
    return llm_default_prepare(*this, input, cache, window_size);
}

LMOutput BitNetModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    auto logits = forward_impl(input.tokens, cache);
    return LMOutput(logits);
}

mx::array BitNetModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    auto out = model_(inputs, cache);
    if (lm_head_weight_.has_value()) {
        return linear_forward(out, lm_head_weight_.value());
    } else {
        return model_.embed_as_linear(out);
    }
}

std::unordered_map<std::string, mx::array>
BitNetModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights)
{
    // Dequantize uint8 packed ternary weights at load time.
    // Each *.weight (uint8, shape [out/4, in]) is paired with a *.weight_scale (bf16, shape [1]).
    // After dequantization, the weight becomes float16 [out, in] and the scale is removed.
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
    for (const auto& k : to_remove) {
        weights.erase(k);
    }

    // Remove unused precomputed rotary frequencies
    std::vector<std::string> rotary_remove;
    for (auto& [k, v] : weights) {
        if (k.find("self_attn.rotary_emb.inv_freq") != std::string::npos) {
            rotary_remove.push_back(k);
        }
    }
    for (const auto& k : rotary_remove) {
        weights.erase(k);
    }

    return weights;
}

void BitNetModel::load_weights(
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

std::unordered_map<std::string, mx::array*> BitNetModel::weight_map() {
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
