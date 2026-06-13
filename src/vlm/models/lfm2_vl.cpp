// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of LFM2VL.swift — LFM2 VL VLM (SigLip vision + LFM2 hybrid attention/conv language)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <mlx-lm/vlm/models/lfm2_vl.h>
#include <mlx-lm/common/activations.h>
#include <mlx-lm/common/attention_utils.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON Deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, LFM2VLTextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("lfm2"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_key_value_heads = j.at("num_key_value_heads").get<int>();
    c.vocab_size = j.at("vocab_size").get<int>();
    c.norm_eps = j.value("norm_eps", 1e-5f);
    c.conv_bias = j.value("conv_bias", false);
    c.conv_l_cache = j.value("conv_L_cache", 3);
    c.block_dim = j.value("block_dim", -1);
    c.block_ff_dim = j.value("block_ff_dim", -1);
    c.block_multiple_of = j.value("block_multiple_of", 256);
    c.block_ffn_dim_multiplier = j.value("block_ffn_dim_multiplier", 1.0f);
    c.block_auto_adjust_ff_dim = j.value("block_auto_adjust_ff_dim", true);
    c.rope_theta = j.value("rope_theta", 1000000.0f);

    // Derive full_attn_idxs from either full_attn_idxs or layer_types
    if (j.contains("full_attn_idxs")) {
        c.full_attn_idxs = j["full_attn_idxs"].get<std::vector<int>>();
    } else if (j.contains("layer_types")) {
        auto layer_types = j["layer_types"].get<std::vector<std::string>>();
        c.full_attn_idxs.clear();
        for (size_t i = 0; i < layer_types.size(); ++i) {
            if (layer_types[i] == "full_attention") {
                c.full_attn_idxs.push_back(static_cast<int>(i));
            }
        }
    } else {
        // Default: all layers are attention layers
        c.full_attn_idxs.resize(c.num_hidden_layers);
        for (int i = 0; i < c.num_hidden_layers; ++i) {
            c.full_attn_idxs[i] = i;
        }
    }
}

void from_json(const nlohmann::json& j, LFM2VLVisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("siglip_vision_model"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.num_channels = j.value("num_channels", 3);
    c.image_size = j.value("image_size", 224);
    c.patch_size = j.value("patch_size", 16);
    c.num_patches = j.value("num_patches", 256);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
}

void from_json(const nlohmann::json& j, LFM2VLConfiguration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<LFM2VLTextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<LFM2VLVisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("lfm2_vl"));
    c.downsample_factor = j.value("downsample_factor", 2);
    c.image_token_index = j.value("image_token_id", 396);
    c.projector_bias = j.value("projector_bias", true);
    c.projector_hidden_size = j.value("projector_hidden_size", 2560);
    c.projector_use_layernorm = j.value("projector_use_layernorm", true);
    c.vision_feature_layer = j.value("vision_feature_layer", -1);
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

// GELU approximate (matching Swift's geluApproximate)
static mx::array gelu_approx(const mx::array& x) {
    // tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    auto sqrt_2_pi = mx::array(std::sqrt(2.0f / M_PI));
    auto coeff = mx::array(0.044715f);
    auto half = mx::array(0.5f);
    auto one = mx::array(1.0f);
    auto inner = mx::multiply(sqrt_2_pi,
        mx::add(x, mx::multiply(coeff, mx::power(x, mx::array(3.0f)))));
    return mx::multiply(mx::multiply(half, x), mx::add(one, mx::tanh(inner)));
}

// Standard GELU (exact)
static mx::array gelu_exact(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// Depthwise Conv1d forward pass
// input: [B, L, C], weight: [C, 1, K] (groups=C), optional bias: [C]
// Output: [B, L_out, C] where L_out = L - K + 1
static mx::array conv1d_depthwise(const mx::array& input, const mx::array& weight,
                                    const mx::array* bias = nullptr) {
    int B = input.shape(0);
    int L = input.shape(1);
    int C = input.shape(2);
    int K = weight.shape(2);
    int L_out = L - K + 1;

    if (L_out <= 0) {
        throw std::runtime_error("Conv1d: input length too short for kernel size");
    }

    // Build output by sliding window matmul per position
    // For depthwise conv: for each output position p, output[p] = sum_k input[p+k] * weight[k]
    // Reshape weight from [C, 1, K] to [C, K] then transpose to [K, C]
    auto w_flat = mx::reshape(weight, {C, K}); // [C, K]

    // Unfold input: create [B, L_out, K, C] from input
    // We can do this by gathering slices
    std::vector<mx::array> slices;
    slices.reserve(K);
    for (int k = 0; k < K; ++k) {
        // input[:, k:k+L_out, :] -> [B, L_out, C]
        auto sl = mx::slice(input, {0, k, 0}, {B, k + L_out, C});
        slices.push_back(mx::expand_dims(sl, 2)); // [B, L_out, 1, C]
    }
    auto unfolded = mx::concatenate(slices, 2); // [B, L_out, K, C]

    // For depthwise: element-wise multiply and sum over K
    // w_flat: [C, K] -> broadcast to [1, 1, K, C]
    auto w_broadcast = mx::reshape(mx::transpose(w_flat), {1, 1, K, C}); // [1, 1, K, C]
    auto result = mx::sum(mx::multiply(unfolded, w_broadcast), 2); // [B, L_out, C]

    if (bias) {
        result = mx::add(result, *bias);
    }
    return result;
}

// Bicubic interpolation for resizing positional embeddings
// input: [1, C, H, W], output: [1, C, target_h, target_w]
static mx::array bicubic_interpolate(const mx::array& input, int target_h, int target_w) {
    // input shape: [1, C, src_H, src_W]
    int C = input.shape(1);
    int src_h = input.shape(2);
    int src_w = input.shape(3);

    if (src_h == target_h && src_w == target_w) {
        return input;
    }

    // Use bilinear interpolation as a practical approximation
    // Compute sampling grid
    auto h_coords = mx::divide(
        mx::multiply(mx::arange(0, target_h, mx::float32), mx::array(static_cast<float>(src_h - 1))),
        mx::array(static_cast<float>(std::max(target_h - 1, 1))));
    auto w_coords = mx::divide(
        mx::multiply(mx::arange(0, target_w, mx::float32), mx::array(static_cast<float>(src_w - 1))),
        mx::array(static_cast<float>(std::max(target_w - 1, 1))));

    // Floor and ceil indices
    auto h_floor = mx::astype(mx::floor(h_coords), mx::int32);
    auto w_floor = mx::astype(mx::floor(w_coords), mx::int32);
    auto h_ceil = mx::minimum(mx::add(h_floor, mx::array(1, mx::int32)), mx::array(src_h - 1, mx::int32));
    auto w_ceil = mx::minimum(mx::add(w_floor, mx::array(1, mx::int32)), mx::array(src_w - 1, mx::int32));

    // Fractional parts
    auto h_frac = mx::subtract(h_coords, mx::astype(h_floor, mx::float32)); // [target_h]
    auto w_frac = mx::subtract(w_coords, mx::astype(w_floor, mx::float32)); // [target_w]

    // input is [1, C, src_H, src_W]
    // Reshape for gather: [C, src_H, src_W]
    auto img = mx::reshape(input, {C, src_h, src_w});

    // For bilinear interpolation:
    // out[h, w] = (1-hf)(1-wf)*img[hf_i, wf_i] + (1-hf)*wf*img[hf_i, wc_i]
    //           + hf*(1-wf)*img[hc_i, wf_i] + hf*wf*img[hc_i, wc_i]

    // Gather corner values for all target positions
    // Create index arrays: [target_h, target_w]
    auto hf_2d = mx::expand_dims(h_floor, 1); // [target_h, 1]
    auto hc_2d = mx::expand_dims(h_ceil, 1);
    auto wf_2d = mx::expand_dims(w_floor, 0); // [1, target_w]
    auto wc_2d = mx::expand_dims(w_ceil, 0);

    hf_2d = mx::broadcast_to(hf_2d, {target_h, target_w});
    hc_2d = mx::broadcast_to(hc_2d, {target_h, target_w});
    wf_2d = mx::broadcast_to(wf_2d, {target_h, target_w});
    wc_2d = mx::broadcast_to(wc_2d, {target_h, target_w});

    // Compute linear indices: h * src_w + w
    auto idx_ff = mx::add(mx::multiply(hf_2d, mx::array(src_w, mx::int32)), wf_2d); // [target_h, target_w]
    auto idx_fc = mx::add(mx::multiply(hf_2d, mx::array(src_w, mx::int32)), wc_2d);
    auto idx_cf = mx::add(mx::multiply(hc_2d, mx::array(src_w, mx::int32)), wf_2d);
    auto idx_cc = mx::add(mx::multiply(hc_2d, mx::array(src_w, mx::int32)), wc_2d);

    // Flatten spatial dims of img: [C, src_H*src_W]
    auto img_flat = mx::reshape(img, {C, src_h * src_w});

    // Flatten index arrays: [target_h * target_w]
    auto flat_ff = mx::reshape(idx_ff, {target_h * target_w});
    auto flat_fc = mx::reshape(idx_fc, {target_h * target_w});
    auto flat_cf = mx::reshape(idx_cf, {target_h * target_w});
    auto flat_cc = mx::reshape(idx_cc, {target_h * target_w});

    // Gather: [C, target_h * target_w]
    auto v_ff = mx::take(img_flat, flat_ff, 1);
    auto v_fc = mx::take(img_flat, flat_fc, 1);
    auto v_cf = mx::take(img_flat, flat_cf, 1);
    auto v_cc = mx::take(img_flat, flat_cc, 1);

    // Reshape to [C, target_h, target_w]
    v_ff = mx::reshape(v_ff, {C, target_h, target_w});
    v_fc = mx::reshape(v_fc, {C, target_h, target_w});
    v_cf = mx::reshape(v_cf, {C, target_h, target_w});
    v_cc = mx::reshape(v_cc, {C, target_h, target_w});

    // Weights: [target_h, 1] and [1, target_w]
    auto hf_w = mx::expand_dims(h_frac, 1); // [target_h, 1]
    auto wf_w = mx::expand_dims(w_frac, 0); // [1, target_w]

    auto one_f = mx::array(1.0f);
    auto one_minus_hf = mx::subtract(one_f, hf_w);
    auto one_minus_wf = mx::subtract(one_f, wf_w);

    // Broadcast weights to [1, target_h, target_w] for [C, target_h, target_w]
    auto w_ff = mx::expand_dims(mx::multiply(one_minus_hf, one_minus_wf), 0); // [1, th, tw]
    auto w_fc = mx::expand_dims(mx::multiply(one_minus_hf, wf_w), 0);
    auto w_cf = mx::expand_dims(mx::multiply(hf_w, one_minus_wf), 0);
    auto w_cc = mx::expand_dims(mx::multiply(hf_w, wf_w), 0);

    auto result = mx::add(mx::add(mx::multiply(v_ff, w_ff), mx::multiply(v_fc, w_fc)),
                          mx::add(mx::multiply(v_cf, w_cf), mx::multiply(v_cc, w_cc)));

    // result: [C, target_h, target_w], reshape to [1, C, target_h, target_w]
    return mx::reshape(result, {1, C, target_h, target_w});
}

// ── Vision Components ──────────────────────────────────────────────────

// -- Vision Embeddings --

LFM2VLVisionEmbeddings::LFM2VLVisionEmbeddings(const LFM2VLVisionConfiguration& config)
    : patch_embedding_weight_(mx::zeros({config.hidden_size,
                                          config.num_channels * config.patch_size * config.patch_size})),
      position_embedding_weight_(mx::zeros({config.num_patches, config.hidden_size})),
      embed_dim_(config.hidden_size),
      patch_size_(config.patch_size),
      num_patches_(config.num_patches),
      position_embedding_size_(config.position_embedding_size()),
      num_channels_(config.num_channels)
{}

mx::array LFM2VLVisionEmbeddings::operator()(
    const mx::array& pixel_values,
    const mx::array& spatial_shapes)
{
    auto target_dtype = patch_embedding_weight_.dtype();
    auto patch_embeds = linear_fwd(mx::astype(pixel_values, target_dtype), patch_embedding_weight_);

    // Reshape position embeddings to [pos_size, pos_size, embed_dim]
    auto pos_embeds = mx::reshape(position_embedding_weight_,
                                   {position_embedding_size_, position_embedding_size_, -1});

    // Resize positional embeddings for each batch element using bicubic interpolation
    int batch_size = spatial_shapes.shape(0);
    int max_length = pixel_values.shape(1);
    auto source_dtype = position_embedding_weight_.dtype();

    // Reshape pos_embeds from [H, W, D] to [1, D, H, W] for interpolation
    auto reshaped_embeds = mx::transpose(pos_embeds, {2, 0, 1}); // [D, H, W]
    reshaped_embeds = mx::reshape(reshaped_embeds, {1, embed_dim_, position_embedding_size_, position_embedding_size_});

    // For single batch (common case), do a single interpolation
    if (batch_size == 1) {
        // Extract target shape: spatial_shapes[0] = [h, w]
        auto h_arr = mx::slice(mx::reshape(spatial_shapes, {batch_size, 2}), {0, 0}, {1, 1});
        auto w_arr = mx::slice(mx::reshape(spatial_shapes, {batch_size, 2}), {0, 1}, {1, 2});

        // Evaluate to get integer values
        mx::eval(h_arr);
        mx::eval(w_arr);
        int target_h = h_arr.item<int>();
        int target_w = w_arr.item<int>();

        auto interpolated = bicubic_interpolate(reshaped_embeds, target_h, target_w);
        // [1, D, th, tw] -> [D, th*tw] -> [th*tw, D]
        auto resized = mx::transpose(mx::reshape(interpolated, {embed_dim_, target_h * target_w}));
        // [th*tw, D] -> [1, th*tw, D]
        resized = mx::expand_dims(resized, 0);

        // Pad to max_length if needed
        int num_positions = target_h * target_w;
        if (num_positions < max_length) {
            int pad_len = max_length - num_positions;
            // Pad with zeros (or first embedding repeated) -- use zeros for simplicity
            auto padding = mx::zeros({1, pad_len, embed_dim_}, source_dtype);
            resized = mx::concatenate({resized, padding}, 1);
        }

        return mx::add(patch_embeds, resized);
    }

    // Multi-batch case: build result with per-image interpolation
    auto result = mx::zeros({batch_size, max_length, embed_dim_}, source_dtype);

    for (int i = 0; i < batch_size; ++i) {
        auto shape_i = mx::slice(mx::reshape(spatial_shapes, {batch_size, 2}), {i, 0}, {i + 1, 2});
        mx::eval(shape_i);
        auto h_val = mx::slice(shape_i, {0, 0}, {1, 1});
        auto w_val = mx::slice(shape_i, {0, 1}, {1, 2});
        mx::eval(h_val);
        mx::eval(w_val);
        int target_h = h_val.item<int>();
        int target_w = w_val.item<int>();

        auto interpolated = bicubic_interpolate(reshaped_embeds, target_h, target_w);
        auto resized = mx::transpose(mx::reshape(interpolated, {embed_dim_, target_h * target_w}));

        int num_positions = target_h * target_w;

        // Assign into result using scatter-like operations
        // For simplicity, we build the full-length embedding per batch and stack
        if (num_positions < max_length) {
            auto padding = mx::zeros({max_length - num_positions, embed_dim_}, source_dtype);
            resized = mx::concatenate({resized, padding}, 0);
        }
        // result[i] = resized (handled later via stacking)
        if (i == 0) {
            result = mx::expand_dims(resized, 0);
        } else {
            result = mx::concatenate({result, mx::expand_dims(resized, 0)}, 0);
        }
    }

    return mx::add(patch_embeds, result);
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionEmbeddings::weight_map() {
    return {
        {"patch_embedding.weight", &patch_embedding_weight_},
        {"position_embedding.weight", &position_embedding_weight_},
    };
}

// -- Vision Attention --

LFM2VLVisionAttention::LFM2VLVisionAttention(int dims, int num_heads)
    : num_heads_(num_heads),
      head_dim_(dims / num_heads),
      scale_(std::pow(static_cast<float>(dims / num_heads), -0.5f)),
      wq_weight_(mx::zeros({dims, dims})),
      wq_bias_(mx::zeros({dims})),
      wk_weight_(mx::zeros({dims, dims})),
      wk_bias_(mx::zeros({dims})),
      wv_weight_(mx::zeros({dims, dims})),
      wv_bias_(mx::zeros({dims})),
      wo_weight_(mx::zeros({dims, dims})),
      wo_bias_(mx::zeros({dims}))
{}

mx::array LFM2VLVisionAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys    = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values  = linear_fwd(x, wv_weight_, &wv_bias_);

    int S = keys.shape(1);

    // Reshape to [B, num_heads, L/S, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, S, num_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, S, num_heads_, head_dim_}), {0, 2, 1, 3});

    auto output = sdpa(queries, keys, values, scale_, mask);

    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"out_proj.weight", &wo_weight_}, {"out_proj.bias", &wo_bias_},
    };
}

// -- Vision MLP --

LFM2VLVisionMLP::LFM2VLVisionMLP(const LFM2VLVisionConfiguration& config)
    : fc1_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      fc1_bias_(mx::zeros({config.intermediate_size})),
      fc2_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      fc2_bias_(mx::zeros({config.hidden_size}))
{}

mx::array LFM2VLVisionMLP::operator()(const mx::array& x) {
    return linear_fwd(gelu_approx(linear_fwd(x, fc1_weight_, &fc1_bias_)), fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// -- Vision Encoder Layer --

LFM2VLVisionEncoderLayer::LFM2VLVisionEncoderLayer(const LFM2VLVisionConfiguration& config)
    : attention_(config.hidden_size, config.num_attention_heads),
      mlp_(config),
      layer_norm1_weight_(mx::ones({config.hidden_size})),
      layer_norm1_bias_(mx::zeros({config.hidden_size})),
      layer_norm2_weight_(mx::ones({config.hidden_size})),
      layer_norm2_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array LFM2VLVisionEncoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask)
{
    auto r = attention_(mx::fast::layer_norm(x, layer_norm1_weight_, layer_norm1_bias_, eps_), mask);
    auto h = mx::add(x, r);
    r = mlp_(mx::fast::layer_norm(h, layer_norm2_weight_, layer_norm2_bias_, eps_));
    return mx::add(h, r);
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionEncoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["layer_norm1.weight"] = &layer_norm1_weight_;
    map["layer_norm1.bias"] = &layer_norm1_bias_;
    map["layer_norm2.weight"] = &layer_norm2_weight_;
    map["layer_norm2.bias"] = &layer_norm2_bias_;
    return map;
}

// -- Vision Encoder --

LFM2VLVisionEncoder::LFM2VLVisionEncoder(
    const LFM2VLVisionConfiguration& config,
    int vision_feature_layer)
{
    int num_layers;
    if (vision_feature_layer == -1) {
        num_layers = config.num_hidden_layers;
    } else {
        int actual_layer = vision_feature_layer < 0
            ? config.num_hidden_layers + vision_feature_layer
            : vision_feature_layer;
        if (actual_layer >= 0 && actual_layer < config.num_hidden_layers) {
            num_layers = actual_layer + 1;
        } else {
            num_layers = config.num_hidden_layers;
        }
    }

    layers_.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        layers_.emplace_back(config);
    }
}

mx::array LFM2VLVisionEncoder::operator()(
    const mx::array& x,
    bool /*output_hidden_states*/,
    const AttentionMask& mask)
{
    auto h = x;
    for (auto& layer : layers_) {
        h = layer(h, mask);
    }
    return h;
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Vision Model --

LFM2VLVisionModel::LFM2VLVisionModel(
    const LFM2VLVisionConfiguration& config,
    int vision_feature_layer)
    : embeddings_(config),
      encoder_(config, vision_feature_layer),
      post_layernorm_weight_(mx::ones({config.hidden_size})),
      post_layernorm_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array LFM2VLVisionModel::operator()(
    const mx::array& x,
    bool output_hidden_states,
    const mx::array& spatial_shapes)
{
    auto embeds = embeddings_(x, spatial_shapes);
    embeds = mx::astype(embeds, embeddings_.patch_embedding_weight().dtype());

    auto encoder_out = encoder_(embeds, output_hidden_states, AttentionMask{});
    return mx::fast::layer_norm(encoder_out, post_layernorm_weight_, post_layernorm_bias_, eps_);
}

std::unordered_map<std::string, mx::array*> LFM2VLVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embeddings_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    map["post_layernorm.weight"] = &post_layernorm_weight_;
    map["post_layernorm.bias"] = &post_layernorm_bias_;
    return map;
}

// ── Language Components (LFM2) ─────────────────────────────────────────

// -- LFM2 Attention --

LFM2Attention::LFM2Attention(const LFM2VLTextConfiguration& config)
    : heads_(config.num_attention_heads),
      kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim()),
      scale_(std::pow(static_cast<float>(config.head_dim()), -0.5f)),
      rope_theta_(config.rope_theta),
      norm_eps_(config.norm_eps),
      wq_weight_(mx::zeros({config.num_attention_heads * config.head_dim(), config.hidden_size})),
      wk_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wv_weight_(mx::zeros({config.num_key_value_heads * config.head_dim(), config.hidden_size})),
      wo_weight_(mx::zeros({config.hidden_size, config.num_attention_heads * config.head_dim()})),
      q_layernorm_weight_(mx::ones({config.head_dim()})),
      k_layernorm_weight_(mx::ones({config.head_dim()}))
{}

mx::array LFM2Attention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    // Reshape for per-head layernorm: [B, L, heads, head_dim]
    queries = mx::reshape(queries, {B, L, heads_, head_dim_});
    keys    = mx::reshape(keys,    {B, L, kv_heads_, head_dim_});

    // Apply q/k layernorm (RMSNorm on last dim = head_dim)
    queries = mx::fast::rms_norm(queries, q_layernorm_weight_, norm_eps_);
    keys    = mx::fast::rms_norm(keys, k_layernorm_weight_, norm_eps_);

    // Transpose to [B, heads, L, head_dim]
    queries = mx::transpose(queries, {0, 2, 1, 3});
    keys    = mx::transpose(keys,    {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values, {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply RoPE
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, false, rope_theta_, 1.0f, offset);
    keys    = mx::fast::rope(keys,    head_dim_, false, rope_theta_, 1.0f, offset);

    // Update KV cache
    if (cache) {
        auto [k, v] = cache->update(keys, values);
        keys = k;
        values = v;
    }

    auto output = sdpa(queries, keys, values, scale_, mask);
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2Attention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"out_proj.weight", &wo_weight_},
        {"q_layernorm.weight", &q_layernorm_weight_},
        {"k_layernorm.weight", &k_layernorm_weight_},
    };
}

// -- LFM2 Short Conv --

LFM2ShortConv::LFM2ShortConv(const LFM2VLTextConfiguration& config, int /*layer_idx*/)
    : l_cache_(config.conv_l_cache),
      hidden_size_(config.hidden_size),
      bias_(config.conv_bias),
      conv_weight_(mx::zeros({config.hidden_size, 1, config.conv_l_cache})),
      in_proj_weight_(mx::zeros({3 * config.hidden_size, config.hidden_size})),
      out_proj_weight_(mx::zeros({config.hidden_size, config.hidden_size}))
{
    if (bias_) {
        conv_bias_ = mx::zeros({config.hidden_size});
        in_proj_bias_ = mx::zeros({3 * config.hidden_size});
        out_proj_bias_ = mx::zeros({config.hidden_size});
    }
}

mx::array LFM2ShortConv::operator()(const mx::array& x, ConvState* conv_state) {
    // in_proj: [B, L, hidden] -> [B, L, 3*hidden]
    mx::array bcx = in_proj_bias_.has_value()
        ? linear_fwd(x, in_proj_weight_, &in_proj_bias_.value())
        : linear_fwd(x, in_proj_weight_);

    // Split into B, C, x parts
    int split_size = hidden_size_;
    auto b_part = mx::slice(bcx, {0, 0, 0}, {bcx.shape(0), bcx.shape(1), split_size});
    auto c_part = mx::slice(bcx, {0, 0, split_size}, {bcx.shape(0), bcx.shape(1), 2 * split_size});
    auto x_part = mx::slice(bcx, {0, 0, 2 * split_size}, {bcx.shape(0), bcx.shape(1), 3 * split_size});

    // Bx = B * x
    auto bx = mx::multiply(b_part, x_part);

    // Get or initialize conv state
    mx::array state = (conv_state && conv_state->state.has_value())
        ? conv_state->state.value()
        : mx::zeros({bx.shape(0), l_cache_ - 1, hidden_size_}, bx.dtype());

    // Concatenate state with current input
    bx = mx::concatenate({state, bx}, 1);

    // Update conv state: store last (l_cache_ - 1) elements
    if (conv_state) {
        int bx_len = bx.shape(1);
        int start = bx_len - (l_cache_ - 1);
        conv_state->state = mx::slice(bx, {0, start, 0},
                                       {bx.shape(0), bx_len, bx.shape(2)});
    }

    // Apply depthwise conv1d
    const mx::array* bias_ptr = conv_bias_.has_value() ? &conv_bias_.value() : nullptr;
    auto conv_out = conv1d_depthwise(bx, conv_weight_, bias_ptr);

    // y = C * conv_out
    auto y = mx::multiply(c_part, conv_out);

    // out_proj
    if (out_proj_bias_.has_value()) {
        return linear_fwd(y, out_proj_weight_, &out_proj_bias_.value());
    }
    return linear_fwd(y, out_proj_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2ShortConv::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"conv.weight", &conv_weight_},
        {"in_proj.weight", &in_proj_weight_},
        {"out_proj.weight", &out_proj_weight_},
    };
    if (conv_bias_.has_value()) {
        map["conv.bias"] = &conv_bias_.value();
    }
    if (in_proj_bias_.has_value()) {
        map["in_proj.bias"] = &in_proj_bias_.value();
    }
    if (out_proj_bias_.has_value()) {
        map["out_proj.bias"] = &out_proj_bias_.value();
    }
    return map;
}

// -- LFM2 MLP --

LFM2MLP::LFM2MLP(const LFM2VLTextConfiguration& config)
    : w1_weight_(mx::zeros({config.adjusted_ff_dim(), config.effective_block_dim()})),
      w2_weight_(mx::zeros({config.effective_block_dim(), config.adjusted_ff_dim()})),
      w3_weight_(mx::zeros({config.adjusted_ff_dim(), config.effective_block_dim()}))
{}

mx::array LFM2MLP::operator()(const mx::array& x) {
    // w2(swiglu(w1(x), w3(x)))
    return linear_fwd(swiglu(linear_fwd(x, w1_weight_),
                             linear_fwd(x, w3_weight_)),
                      w2_weight_);
}

std::unordered_map<std::string, mx::array*> LFM2MLP::weight_map() {
    return {
        {"w1.weight", &w1_weight_},
        {"w2.weight", &w2_weight_},
        {"w3.weight", &w3_weight_},
    };
}

// -- LFM2 Decoder Layer --

LFM2DecoderLayer::LFM2DecoderLayer(const LFM2VLTextConfiguration& config, int layer_idx)
    : is_attention_layer_(config.is_attention_layer(layer_idx)),
      feed_forward_(config),
      operator_norm_weight_(mx::ones({config.hidden_size})),
      ffn_norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.norm_eps)
{
    if (is_attention_layer_) {
        attention_.emplace(config);
    } else {
        conv_.emplace(config, layer_idx);
    }
}

mx::array LFM2DecoderLayer::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* kv_cache,
    ConvState* conv_state)
{
    auto normed = mx::fast::rms_norm(x, operator_norm_weight_, norm_eps_);

    mx::array r = is_attention_layer_
        ? attention_.value()(normed, mask, kv_cache)
        : conv_.value()(normed, conv_state);

    auto h = mx::add(x, r);
    auto out = mx::add(h, feed_forward_(mx::fast::rms_norm(h, ffn_norm_weight_, norm_eps_)));
    return out;
}

std::unordered_map<std::string, mx::array*> LFM2DecoderLayer::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (is_attention_layer_ && attention_.has_value()) {
        for (auto& [k, v] : attention_.value().weight_map()) map["self_attn." + k] = v;
    }
    if (!is_attention_layer_ && conv_.has_value()) {
        for (auto& [k, v] : conv_.value().weight_map()) map["conv." + k] = v;
    }
    for (auto& [k, v] : feed_forward_.weight_map()) map["feed_forward." + k] = v;
    map["operator_norm.weight"] = &operator_norm_weight_;
    map["ffn_norm.weight"] = &ffn_norm_weight_;
    return map;
}

// -- LFM2 Model Inner --

LFM2ModelInner::LFM2ModelInner(const LFM2VLTextConfiguration& config)
    : embed_tokens_weight_(mx::zeros({config.vocab_size, config.hidden_size})),
      embedding_norm_weight_(mx::ones({config.hidden_size})),
      norm_eps_(config.norm_eps),
      full_attn_idxs_(config.full_attn_idxs)
{
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(config, i);
    }

    // Initialize conv states (one per layer)
    conv_states_.resize(config.num_hidden_layers);
}

mx::array LFM2ModelInner::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    mx::array h = [&]() -> mx::array {
        if (input_embedding.has_value()) {
            return input_embedding.value();
        } else if (inputs.has_value()) {
            return mx::take(embed_tokens_weight_, inputs.value(), 0);
        } else {
            throw std::runtime_error("Either inputs or input_embedding must be provided");
        }
    }();

    // Create attention mask using the first attention layer's cache
    auto mask = create_attention_mask(h, nullptr);
    if (cache && !cache->empty()) {
        // Find the first attention layer to get the offset for mask creation
        for (size_t i = 0; i < layers_.size(); ++i) {
            if (layers_[i].is_attention_layer() && i < cache->size()) {
                mask = create_attention_mask(h, &(*cache)[i]);
                break;
            }
        }
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        KVCache* kv = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        ConvState* cs = (i < conv_states_.size()) ? &conv_states_[i] : nullptr;
        h = layers_[i](h, mask, kv, cs);
    }

    return mx::fast::rms_norm(h, embedding_norm_weight_, norm_eps_);
}

mx::array LFM2ModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

mx::array LFM2ModelInner::embed_as_linear(const mx::array& x) const {
    return mx::matmul(x, mx::transpose(embed_tokens_weight_));
}

void LFM2ModelInner::reset_conv_states() {
    for (auto& cs : conv_states_) {
        cs.state = std::nullopt;
    }
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

// -- LFM2 Language Model --

LFM2LanguageModel::LFM2LanguageModel(const LFM2VLTextConfiguration& config)
    : model_(config)
{
    // Per-layer kv_heads: attention layers get config.num_key_value_heads,
    // conv layers get 0 (they use ConvState instead of KVCache)
    kv_heads_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        kv_heads_[i] = config.is_attention_layer(i) ? config.num_key_value_heads : 0;
    }
}

LMOutput LFM2LanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);
    // Tied embeddings: use embed_tokens weight as the output projection
    out = model_.embed_as_linear(out);
    return LMOutput(out);
}

std::unordered_map<std::string, mx::array*> LFM2LanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    return map;
}

// ── Multi-Modal Components ─────────────────────────────────────────────

// -- Pixel Unshuffle Block --

PixelUnshuffleBlock::PixelUnshuffleBlock(int factor)
    : factor_(factor)
{}

mx::array PixelUnshuffleBlock::operator()(const mx::array& input) {
    auto x = input;
    int n = x.shape(0);
    int w = x.shape(1);
    int h = x.shape(2);
    int c = x.shape(3);

    // Pad width if necessary
    if (w % factor_ != 0) {
        int pad_w = factor_ - (w % factor_);
        auto padding = mx::zeros({n, pad_w, h, c}, x.dtype());
        x = mx::concatenate({x, padding}, 1);
        w = x.shape(1);
    }

    // Pad height if necessary
    if (h % factor_ != 0) {
        int pad_h = factor_ - (h % factor_);
        auto padding = mx::zeros({n, w, pad_h, c}, x.dtype());
        x = mx::concatenate({x, padding}, 2);
        h = x.shape(2);
    }

    // Pixel unshuffle: rearrange spatial dimensions into channel dimension
    x = mx::reshape(x, {n, w, h / factor_, c * factor_});
    x = mx::transpose(x, {0, 2, 1, 3});
    x = mx::reshape(x, {n, h / factor_, w / factor_, c * factor_ * factor_});
    x = mx::transpose(x, {0, 2, 1, 3});

    return x;
}

// -- Multi-Modal Projector --

static int lfm2_projector_in_channels(const LFM2VLConfiguration& config) {
    return config.vision_config.hidden_size
           * (config.downsample_factor * config.downsample_factor);
}

LFM2VLMultiModalProjector::LFM2VLMultiModalProjector(const LFM2VLConfiguration& config)
    : use_layernorm_(config.projector_use_layernorm),
      layer_norm_weight_(config.projector_use_layernorm
          ? mx::ones({lfm2_projector_in_channels(config)})
          : mx::array(0.0f)),
      layer_norm_bias_(config.projector_use_layernorm
          ? mx::zeros({lfm2_projector_in_channels(config)})
          : mx::array(0.0f)),
      linear1_weight_(mx::zeros({config.projector_hidden_size, lfm2_projector_in_channels(config)})),
      linear2_weight_(mx::zeros({config.text_config.hidden_size, config.projector_hidden_size}))
{
    if (config.projector_bias) {
        linear1_bias_ = mx::zeros({config.projector_hidden_size});
        linear2_bias_ = mx::zeros({config.text_config.hidden_size});
    }
}

mx::array LFM2VLMultiModalProjector::operator()(const mx::array& input) {
    auto x = input;

    if (use_layernorm_) {
        x = mx::fast::layer_norm(x, layer_norm_weight_, layer_norm_bias_, layer_norm_eps_);
    }

    if (linear1_bias_.has_value()) {
        x = linear_fwd(x, linear1_weight_, &linear1_bias_.value());
    } else {
        x = linear_fwd(x, linear1_weight_);
    }

    x = gelu_exact(x);

    if (linear2_bias_.has_value()) {
        x = linear_fwd(x, linear2_weight_, &linear2_bias_.value());
    } else {
        x = linear_fwd(x, linear2_weight_);
    }

    return x;
}

std::unordered_map<std::string, mx::array*> LFM2VLMultiModalProjector::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    if (use_layernorm_) {
        map["layer_norm.weight"] = &layer_norm_weight_;
        map["layer_norm.bias"] = &layer_norm_bias_;
    }
    map["linear_1.weight"] = &linear1_weight_;
    map["linear_2.weight"] = &linear2_weight_;
    if (linear1_bias_.has_value()) {
        map["linear_1.bias"] = &linear1_bias_.value();
    }
    if (linear2_bias_.has_value()) {
        map["linear_2.bias"] = &linear2_bias_.value();
    }
    return map;
}

// ── Top-Level LFM2VL Model ────────────────────────────────────────────

LFM2VLModel::LFM2VLModel(const LFM2VLConfiguration& config)
    : config_(config),
      vision_tower_(config.vision_config, config.vision_feature_layer),
      language_model_(config.text_config),
      multi_modal_projector_(config)
{
    if (config.downsample_factor > 1) {
        pixel_unshuffle_.emplace(config.downsample_factor);
    }
    kv_heads_cache_ = language_model_.kv_heads();
}

mx::array LFM2VLModel::get_input_embeddings(
    const mx::array& input_ids,
    const mx::array* pixel_values,
    const mx::array* spatial_shapes,
    const mx::array* pixel_attention_mask)
{
    // Ensure batch dimension
    auto batched_ids = input_ids;
    if (input_ids.ndim() == 1) {
        batched_ids = mx::expand_dims(input_ids, 0);
    }

    auto inputs_embeds = language_model_.inner().embed_tokens(batched_ids);
    if (inputs_embeds.ndim() == 2) {
        inputs_embeds = mx::expand_dims(inputs_embeds, 0);
    }

    if (!pixel_values || !spatial_shapes || !pixel_attention_mask) {
        return inputs_embeds;
    }

    // Run vision model
    auto hidden_states = vision_tower_(*pixel_values, true, *spatial_shapes);

    // Get feature lengths from attention mask
    auto img_feature_lengths = mx::sum(*pixel_attention_mask, 1); // [num_images]
    mx::eval(img_feature_lengths);

    int num_images = hidden_states.shape(0);

    // Process each image: slice valid features, reshape spatially, pixel unshuffle, project
    std::vector<mx::array> image_features;
    image_features.reserve(num_images);

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        // Get this image's features: [seq_len, hidden]
        auto feature = mx::slice(hidden_states, {img_idx, 0, 0},
                                  {img_idx + 1, hidden_states.shape(1), hidden_states.shape(2)});
        feature = mx::reshape(feature, {hidden_states.shape(1), hidden_states.shape(2)});

        // Get valid feature length
        auto feat_len_arr = mx::slice(img_feature_lengths, {img_idx}, {img_idx + 1});
        mx::eval(feat_len_arr);
        int feat_len = feat_len_arr.item<int>();

        // Slice to valid features
        feature = mx::slice(feature, {0, 0}, {feat_len, feature.shape(1)});
        feature = mx::expand_dims(feature, 0); // [1, feat_len, hidden]

        // Get spatial dimensions for this image
        auto shapes_i = mx::slice(mx::reshape(*spatial_shapes, {-1, 2}),
                                   {img_idx, 0}, {img_idx + 1, 2});
        mx::eval(shapes_i);
        int org_h = mx::slice(shapes_i, {0, 0}, {1, 1}).item<int>();
        int org_w = mx::slice(shapes_i, {0, 1}, {1, 2}).item<int>();

        // Reshape to spatial: [1, H, W, hidden]
        feature = mx::reshape(feature, {1, org_h, org_w, -1});

        // Apply pixel unshuffle if configured
        if (pixel_unshuffle_.has_value()) {
            feature = pixel_unshuffle_.value()(feature);
        }

        // Project to language model dimension
        auto img_embedding = multi_modal_projector_(feature);

        // Flatten back: [num_tokens, hidden]
        img_embedding = mx::reshape(img_embedding, {-1, img_embedding.shape(-1)});
        image_features.push_back(img_embedding);
    }

    // Concatenate all image features
    auto concatenated_features = image_features[0];
    for (size_t i = 1; i < image_features.size(); ++i) {
        concatenated_features = mx::concatenate({concatenated_features, image_features[i]}, 0);
    }

    // Merge image features with text embeddings
    return merge_input_ids_with_image_features(
        concatenated_features, inputs_embeds, input_ids, config_.image_token_index);
}

mx::array LFM2VLModel::merge_input_ids_with_image_features(
    const mx::array& image_features,
    const mx::array& inputs_embeds,
    const mx::array& input_ids,
    int image_token_index)
{
    // Find image token positions in flattened input_ids
    auto flat_ids = mx::reshape(input_ids, {-1});
    auto image_mask = mx::equal(flat_ids, mx::array(image_token_index, flat_ids.dtype()));

    int num_image_features = image_features.shape(0);

    // Ensure result has batch dimension
    auto result = inputs_embeds;
    if (result.ndim() == 2) {
        result = mx::expand_dims(result, 0);
    }

    int B = result.shape(0);
    int L = result.shape(1);
    int D = result.shape(2);

    // Use cumulative sum to create sequential indices for image features
    auto cum_mask = mx::cumsum(mx::astype(image_mask, mx::int32), 0);
    auto gather_indices = mx::subtract(cum_mask, mx::array(1, mx::int32));
    gather_indices = mx::clip(gather_indices, mx::array(0, mx::int32),
                               mx::array(std::max(num_image_features - 1, 0), mx::int32));

    // Expand image features to match all positions, then use where to select
    // image_features: [N, D]
    // gather_indices: [L] (flattened)
    auto gathered = mx::take(image_features, gather_indices, 0); // [L, D]
    gathered = mx::expand_dims(gathered, 0); // [1, L, D]
    gathered = mx::broadcast_to(gathered, {B, L, D});

    auto mask_expanded = mx::expand_dims(image_mask, -1); // [L, 1]
    mask_expanded = mx::expand_dims(mask_expanded, 0);     // [1, L, 1]
    mask_expanded = mx::broadcast_to(mask_expanded, {B, L, 1});

    result = mx::where(mask_expanded, gathered, result);
    return result;
}

PrepareResult LFM2VLModel::prepare_impl(
    const LMInput& input,
    std::vector<KVCache>& cache,
    int /*window_size*/)
{
    auto dtype = vision_tower_.embeddings().patch_embedding_weight().dtype();

    // Get image data if available
    const mx::array* pixel_values_ptr = nullptr;
    mx::array pixel_values_storage = mx::array(0.0f);

    mx::array spatial_shapes_storage = mx::array(0.0f);
    const mx::array* spatial_shapes_ptr = nullptr;

    mx::array pixel_attention_mask_storage = mx::array(0.0f);
    const mx::array* pixel_attention_mask_ptr = nullptr;

    if (input.image.has_value()) {
        pixel_values_storage = mx::astype(input.image->pixels, dtype);
        pixel_values_ptr = &pixel_values_storage;

        if (input.image->frames.has_value() && !input.image->frames->empty()) {
            const auto& frames = input.image->frames.value();

            // Convert frames to spatial shapes array [num_images, 2]
            std::vector<mx::array> shape_arrays;
            shape_arrays.reserve(frames.size());
            for (const auto& f : frames) {
                shape_arrays.push_back(mx::array({f.h, f.w}, {2}, mx::int32));
            }
            spatial_shapes_storage = mx::stack(shape_arrays, 0);
            spatial_shapes_ptr = &spatial_shapes_storage;

            // Create attention mask
            std::vector<mx::array> mask_arrays;
            mask_arrays.reserve(frames.size());
            for (const auto& f : frames) {
                int num_patches = f.h * f.w;
                mask_arrays.push_back(mx::ones({num_patches}, mx::int32));
            }

            if (mask_arrays.size() == 1) {
                pixel_attention_mask_storage = mx::expand_dims(mask_arrays[0], 0);
            } else {
                // Pad masks to max length
                int max_len = 0;
                for (const auto& m : mask_arrays) {
                    max_len = std::max(max_len, m.shape(0));
                }
                std::vector<mx::array> padded;
                padded.reserve(mask_arrays.size());
                for (auto& m : mask_arrays) {
                    if (m.shape(0) < max_len) {
                        auto padding = mx::zeros({max_len - m.shape(0)}, mx::int32);
                        padded.push_back(mx::concatenate({m, padding}, 0));
                    } else {
                        padded.push_back(m);
                    }
                }
                pixel_attention_mask_storage = mx::stack(padded, 0);
            }
            pixel_attention_mask_ptr = &pixel_attention_mask_storage;
        } else if (input.image.has_value()) {
            // Fallback: infer spatial shapes from pixel dimensions (assume square)
            int num_patches = input.image->pixels.shape(1);
            int side = static_cast<int>(std::sqrt(static_cast<double>(num_patches)));
            spatial_shapes_storage = mx::reshape(mx::array({side, side}, {2}, mx::int32), {1, 2});
            spatial_shapes_ptr = &spatial_shapes_storage;
            pixel_attention_mask_storage = mx::ones({1, num_patches}, mx::int32);
            pixel_attention_mask_ptr = &pixel_attention_mask_storage;
        }
    }

    auto input_embeddings = get_input_embeddings(
        input.text.tokens,
        pixel_values_ptr,
        spatial_shapes_ptr,
        pixel_attention_mask_ptr);

    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, input_embeddings);

    return PrepareResult::logits(std::move(result));
}

LMOutput LFM2VLModel::call_impl(
    const LMInput::Text& input,
    std::vector<KVCache>* cache,
    const LMOutput::State* /*state*/)
{
    return language_model_(input.tokens, cache);
}

mx::array LFM2VLModel::forward_impl(
    const mx::array& inputs,
    std::vector<KVCache>* cache)
{
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
LFM2VLModel::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        // Skip position_ids
        if (k.find("position_ids") != std::string::npos) {
            continue;
        }

        // Transform key names
        std::string new_key = k;

        // Vision tower key transformations
        if (new_key.find("vision_tower") != std::string::npos) {
            // Remove "model." prefix
            {
                auto pos = new_key.find("model.");
                while (pos != std::string::npos) {
                    new_key.erase(pos, 6);
                    pos = new_key.find("model.");
                }
            }
            // vision_encoder -> encoder
            {
                auto pos = new_key.find("vision_encoder");
                if (pos != std::string::npos) {
                    new_key.replace(pos, 14, "encoder");
                }
            }
            // vision_embeddings -> embeddings
            {
                auto pos = new_key.find("vision_embeddings");
                if (pos != std::string::npos) {
                    new_key.replace(pos, 17, "embeddings");
                }
            }
            // vision_post_layernorm -> post_layernorm
            {
                auto pos = new_key.find("vision_post_layernorm");
                if (pos != std::string::npos) {
                    new_key.replace(pos, 21, "post_layernorm");
                }
            }
        }

        // Language model key transformation
        if (new_key.find("language_model") != std::string::npos) {
            auto pos = new_key.find("model.language_model");
            if (pos != std::string::npos) {
                new_key.replace(pos, 20, "language_model.model");
            }
        }

        // Multi-modal projector key transformation
        if (new_key.find("multi_modal_projector") != std::string::npos) {
            auto pos = new_key.find("model.multi_modal_projector");
            if (pos != std::string::npos) {
                new_key.replace(pos, 27, "multi_modal_projector");
            }
        }

        // Handle conv weight transposition
        auto value = v;
        if (new_key.find("conv.weight") != std::string::npos) {
            if (v.ndim() == 3 && v.shape(v.ndim() - 1) > v.shape(1)) {
                value = mx::transpose(v, {0, 2, 1});
            }
        }

        sanitized.insert_or_assign(new_key, value);
    }

    return sanitized;
}

void LFM2VLModel::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) {
            *target = it->second;
        }
    }
}

std::unordered_map<std::string, mx::array*> LFM2VLModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    // Vision tower: prefix "vision_tower."
    for (auto& [k, v] : vision_tower_.weight_map())
        map["vision_tower." + k] = v;
    // Language model: prefix "language_model."
    for (auto& [k, v] : language_model_.weight_map())
        map["language_model." + k] = v;
    // Multi-modal projector: prefix "multi_modal_projector."
    for (auto& [k, v] : multi_modal_projector_.weight_map())
        map["multi_modal_projector." + k] = v;
    return map;
}

std::vector<KVCache> LFM2VLModel::new_cache_impl(const GenerateParameters& params) const {
    const auto& text_config = config_.text_config;
    int num_layers = text_config.num_hidden_layers;

    std::vector<KVCache> caches;
    caches.reserve(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        if (text_config.is_attention_layer(i)) {
            // Attention layers get real KV caches
            if (params.max_kv_size.has_value()) {
                caches.emplace_back(RotatingKVCache(params.max_kv_size.value(), 4));
            } else {
                caches.emplace_back(KVCacheSimple{});
            }
        } else {
            // Conv layers get dummy KVCacheSimple entries
            // (actual conv state is stored in LFM2ModelInner::conv_states_)
            caches.emplace_back(KVCacheSimple{});
        }
    }

    // Reset the conv states in the model inner
    // Note: const_cast is needed because new_cache_impl is const (required by CRTP interface)
    // but we need to reset the mutable conv state
    const_cast<LFM2VLModel*>(this)->language_model_.inner().reset_conv_states();

    return caches;
}

} // namespace mlx_lm
