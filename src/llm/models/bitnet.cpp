// BitNet 1.58-bit model implementation — Llama variant with ternary weights.
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/bitnet.py

#include <mlx-lm/llm/models/bitnet.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/bitnet_utils.h>
#include <mlx-lm/common/quantized_linear.h>
#include <mlx/mlx.h>
#include <cmath>
#include <cstring>

namespace mx = mlx::core;

namespace mlx_lm {

// --- Linear helper ---

static mx::array linear_fwd(
    const mx::array& x,
    const mx::array& weight)
{
    return linear_forward(x, weight, nullptr);
}

// Repack BitNet uint8 packed ternary weights into standard MLX uint32 2-bit
// quantized format. Returns {wq_uint32, scales_fp16, biases_fp16}.
//
// BitNet packs 4 ternary codes {0→-1, 1→0, 2→+1} per byte across output lanes:
//   uint8[row, c] = lane0[1:0] | lane1[3:2] | lane2[5:4] | lane3[7:6]
//   where row = oc/4, lane = oc%4.
//
// MLX 2-bit format: uint32[out, ceil(in/16)], each uint32 = 16 codes at 2 bits
// each, least-significant code first, padding with 0.
//
// The quantized matmul kernel requires group_size ∈ {32, 64, 128}.
static constexpr int kBitnetGroupSize = 128;

static std::tuple<mx::array, mx::array, mx::array>
bitnet_repack_weights(
    const mx::array& packed_weight,  // uint8 [out/4, in]
    const mx::array& weight_scale)   // scalar (bf16 or fp16)
{
    auto shape = packed_weight.shape();
    int packed_rows = shape[0];
    int in_features = shape[1];
    int out_features = packed_rows * 4;

    if (in_features % kBitnetGroupSize != 0) {
        throw std::runtime_error(
            "BitNet: in_features " + std::to_string(in_features) +
            " must be divisible by group_size " +
            std::to_string(kBitnetGroupSize));
    }
    int num_groups = in_features / kBitnetGroupSize;

    int in_rounded = ((in_features + 15) / 16) * 16;
    int cols_uint32 = in_rounded / 16;

    // Convert scale to fp16 and materialize
    mx::array ws_fp16 = mx::astype(weight_scale, mx::float16);
    mx::eval(ws_fp16);
    auto ws = static_cast<float>(ws_fp16.data<mx::float16_t>()[0]);

    // Materialize packed weight and read uint8 data
    mx::eval(packed_weight);
    auto w_data = packed_weight.data<uint8_t>();

    // Allocate outputs: scales[out, num_groups], biases[out, num_groups]
    std::vector<uint32_t> wq(out_features * cols_uint32, 0);
    std::vector<mx::float16_t> scales(out_features * num_groups);
    std::vector<mx::float16_t> biases(out_features * num_groups);

    auto ws_h = static_cast<mx::float16_t>(ws);
    auto neg_ws_h = static_cast<mx::float16_t>(-ws);

    for (int oc = 0; oc < out_features; ++oc) {
        int row = oc / 4;
        int lane = oc % 4;
        int bit_shift = lane * 2;

        // Replicate the single BitNet scale across all groups
        for (int g = 0; g < num_groups; ++g) {
            scales[oc * num_groups + g] = ws_h;
            biases[oc * num_groups + g] = neg_ws_h;
        }

        // Pack 16 input values per uint32
        for (int g = 0; g < cols_uint32; ++g) {
            uint32_t packed = 0;
            for (int i = 0; i < 16; ++i) {
                int c = g * 16 + i;
                uint32_t val = 0;
                if (c < in_features) {
                    val = (w_data[row * in_features + c] >> bit_shift) & 0x03;
                }
                packed |= (val << (i * 2));
            }
            wq[oc * cols_uint32 + g] = packed;
        }
    }

    auto wq_arr = mx::array(wq.data(), {out_features, cols_uint32}, mx::uint32);
    auto scales_arr = mx::array(scales.data(), {out_features, num_groups}, mx::float16);
    auto biases_arr = mx::array(biases.data(), {out_features, num_groups}, mx::float16);

    return {std::move(wq_arr), std::move(scales_arr), std::move(biases_arr)};
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
    // Repack uint8 packed ternary weights into standard MLX uint32 2-bit
    // quantized format and register directly in QuantizedWeightRegistry.
    //
    // Each *.weight (uint8, shape [out/4, in]) is paired with a
    // *.weight_scale (bf16, shape [1]). After repacking:
    //   *.weight  → uint32 [out, ceil(in/16)] (standard MLX 2-bit format)
    //   *.scales  → fp16   [out, 1] (replicated weight_scale per output)
    //   *.biases  → fp16   [out, 1] (= -scales, so dequant gives {-ws,0,+ws})
    // The BitNet weight_scale entry is removed.
    //
    // group_size = 128 (kernel-compatible, scale replicated across groups).
    std::vector<std::string> to_remove;
    std::vector<std::pair<std::string, mx::array>> to_add;

    // Get weight_map() for member array pointers — these addresses are valid
    // even before load_weights() fills the data.
    auto wmap = weight_map();
    auto& reg = QuantizedWeightRegistry::instance();

    const std::string scale_suffix = ".weight_scale";

    for (auto& [key, val] : weights) {
        if (key.size() > scale_suffix.size() &&
            key.compare(key.size() - scale_suffix.size(), scale_suffix.size(), scale_suffix) == 0) {

            auto prefix = key.substr(0, key.size() - scale_suffix.size());
            auto weight_key = prefix + ".weight";

            auto w_it = weights.find(weight_key);
            if (w_it != weights.end() && w_it->second.dtype() == mx::uint8) {
                int in_features = w_it->second.shape(1);
                auto [wq, scales, biases] = bitnet_repack_weights(w_it->second, val);

                // Replace uint8 weight with packed uint32 weight
                to_add.emplace_back(weight_key, std::move(wq));
                to_remove.push_back(key); // remove the .weight_scale entry

                // Register in QuantizedWeightRegistry if the member array exists
                auto wm_it = wmap.find(weight_key);
                if (wm_it != wmap.end()) {
                    reg.register_weight(
                        wm_it->second,  // member array pointer (address stable)
                        scales,
                        biases,
                        /*group_size=*/kBitnetGroupSize,
                        /*bits=*/2,
                        "affine");
                }
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
