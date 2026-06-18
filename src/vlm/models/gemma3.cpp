// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Port of Gemma3.swift — Gemma3 VLM (SigLip vision + Gemma3 language)

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <mlx-lm/vlm/models/gemma3.h>
#include <mlx-lm/common/attention_utils.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mx = mlx::core;

namespace mlx_lm {

// ── JSON deserialization ───────────────────────────────────────────────

void from_json(const nlohmann::json& j, Gemma3TextConfiguration& c) {
    c.model_type = j.value("model_type", std::string("gemma3"));
    c.hidden_size = j.at("hidden_size").get<int>();
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.sliding_window = j.at("sliding_window").get<int>();

    if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
        std::unordered_map<std::string, StringOrNumber> scaling;
        for (auto& [key, val] : j["rope_scaling"].items()) {
            StringOrNumber sn;
            from_json(val, sn);
            scaling[key] = sn;
        }
        c.rope_scaling = scaling;
    }

    if (j.contains("final_logit_softcapping") && !j["final_logit_softcapping"].is_null()) {
        c.final_logit_softcapping = j["final_logit_softcapping"].get<float>();
    }

    c.vocab_size = j.value("vocab_size", 262208);
    c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    c.num_attention_heads = j.value("num_attention_heads", 8);
    c.num_key_value_heads = j.value("num_key_value_heads", 4);
    c.head_dim = j.value("head_dim", 256);
    c.query_pre_attn_scalar = j.value("query_pre_attn_scalar", 256.0f);
    c.rope_theta = j.value("rope_theta", 1000000.0f);
    c.rope_local_base_freq = j.value("rope_local_base_freq", 10000.0f);
    c.rope_traditional = j.value("rope_traditional", false);
    c.mm_tokens_per_image = j.value("mm_tokens_per_image", 256);
    c.sliding_window_pattern = j.value("sliding_window_pattern", 6);
    c.max_position_embeddings = j.value("max_position_embeddings", 4096);
}

void from_json(const nlohmann::json& j, Gemma3VisionConfiguration& c) {
    c.model_type = j.value("model_type", std::string("siglip_vision_model"));
    c.num_hidden_layers = j.at("num_hidden_layers").get<int>();
    c.hidden_size = j.at("hidden_size").get<int>();
    c.intermediate_size = j.at("intermediate_size").get<int>();
    c.num_attention_heads = j.at("num_attention_heads").get<int>();
    c.patch_size = j.at("patch_size").get<int>();
    c.image_size = j.at("image_size").get<int>();
    c.num_channels = j.value("num_channels", 3);
    c.layer_norm_eps = j.value("layer_norm_eps", 1e-6f);
}

void from_json(const nlohmann::json& j, Gemma3Configuration& c) {
    if (j.contains("text_config")) {
        c.text_config = j["text_config"].get<Gemma3TextConfiguration>();
    }
    if (j.contains("vision_config")) {
        c.vision_config = j["vision_config"].get<Gemma3VisionConfiguration>();
    }
    c.model_type = j.value("model_type", std::string("gemma3"));
    c.mm_tokens_per_image = j.at("mm_tokens_per_image").get<int>();
    c.vocab_size = j.value("vocab_size", -1);
    c.pad_token_id = j.value("pad_token_id", 0);
}

// ── Helpers ────────────────────────────────────────────────────────────

static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const mx::array* bias = nullptr) {
    auto out = mx::matmul(x, mx::transpose(w));
    if (bias) out = mx::add(out, *bias);
    return out;
}

// GELU precise: x * 0.5 * (1 + erf(x / sqrt(2)))
static mx::array gelu_precise(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto inv_sqrt2 = mx::array(1.0f / std::sqrt(2.0f));
    return mx::multiply(mx::multiply(x, half),
                        mx::add(mx::array(1.0f), mx::erf(mx::multiply(x, inv_sqrt2))));
}

// GELU approximate (fast): used by Gemma3 language MLP
// x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static mx::array gelu_approx(const mx::array& x) {
    auto half = mx::array(0.5f);
    auto coeff = mx::array(0.044715f);
    auto sqrt_2_over_pi = mx::array(std::sqrt(2.0f / M_PI));
    auto x3 = mx::multiply(mx::multiply(x, x), x);
    auto inner = mx::multiply(sqrt_2_over_pi, mx::add(x, mx::multiply(coeff, x3)));
    return mx::multiply(mx::multiply(x, half), mx::add(mx::array(1.0f), mx::tanh(inner)));
}

// ── Vision Components (SigLip) ─────────────────────────────────────────

// -- Vision Embeddings --

Gemma3VisionEmbeddings::Gemma3VisionEmbeddings(const Gemma3VisionConfiguration& config)
    : patch_embedding_weight_(mx::zeros({config.hidden_size, config.patch_size,
                                          config.patch_size, config.num_channels})),
      position_embedding_weight_(mx::zeros({config.num_positions(), config.hidden_size})),
      patch_size_(config.patch_size),
      hidden_size_(config.hidden_size),
      num_positions_(config.num_positions())
{}

mx::array Gemma3VisionEmbeddings::operator()(const mx::array& x) {
    // x: [B, H, W, C] (already in MLX channel-last format)
    int B = x.shape(0);
    int H = x.shape(1);
    int W = x.shape(2);

    // Apply Conv2d via reshaping into patches and matmul
    int nH = H / patch_size_;
    int nW = W / patch_size_;
    int num_patches = nH * nW;

    // Reshape x into patches: [B, nH, patch_size, nW, patch_size, C]
    auto patches = mx::reshape(x, {B, nH, patch_size_, nW, patch_size_, -1});
    // Transpose to [B, nH, nW, patch_size, patch_size, C]
    patches = mx::transpose(patches, {0, 1, 3, 2, 4, 5});
    // Flatten patches: [B, num_patches, patch_size * patch_size * C]
    int kernel_elements = patch_size_ * patch_size_ * x.shape(3);
    patches = mx::reshape(patches, {B, num_patches, kernel_elements});

    // Flatten conv kernel: [out, kH*kW*in]
    auto flat_kernel = mx::reshape(patch_embedding_weight_, {hidden_size_, kernel_elements});
    // Matmul: [B, num_patches, kernel_elements] x [kernel_elements, out] -> [B, num_patches, out]
    auto patch_embeds = mx::matmul(patches, mx::transpose(flat_kernel));

    // Add positional embeddings
    return mx::add(patch_embeds, position_embedding_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3VisionEmbeddings::weight_map() {
    return {
        {"patch_embedding.weight", &patch_embedding_weight_},
        {"position_embedding.weight", &position_embedding_weight_},
    };
}

// -- Vision Attention --

Gemma3VisionAttention::Gemma3VisionAttention(int dims, int num_heads)
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

mx::array Gemma3VisionAttention::operator()(const mx::array& x) {
    int B = x.shape(0);
    int L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_, &wq_bias_);
    auto keys    = linear_fwd(x, wk_weight_, &wk_bias_);
    auto values  = linear_fwd(x, wv_weight_, &wv_bias_);

    // Reshape to [B, num_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, num_heads_, head_dim_}), {0, 2, 1, 3});

    // No mask for vision self-attention (attend to all positions)
    auto output = mx::fast::scaled_dot_product_attention(queries, keys, values, scale_);

    // Reshape back to [B, L, dims]
    output = mx::reshape(mx::transpose(output, {0, 2, 1, 3}), {B, L, -1});
    return linear_fwd(output, wo_weight_, &wo_bias_);
}

std::unordered_map<std::string, mx::array*> Gemma3VisionAttention::weight_map() {
    return {
        {"q_proj.weight", &wq_weight_}, {"q_proj.bias", &wq_bias_},
        {"k_proj.weight", &wk_weight_}, {"k_proj.bias", &wk_bias_},
        {"v_proj.weight", &wv_weight_}, {"v_proj.bias", &wv_bias_},
        {"out_proj.weight", &wo_weight_}, {"out_proj.bias", &wo_bias_},
    };
}

// -- Vision MLP --

Gemma3VisionMLP::Gemma3VisionMLP(const Gemma3VisionConfiguration& config)
    : fc1_weight_(mx::zeros({config.intermediate_size, config.hidden_size})),
      fc1_bias_(mx::zeros({config.intermediate_size})),
      fc2_weight_(mx::zeros({config.hidden_size, config.intermediate_size})),
      fc2_bias_(mx::zeros({config.hidden_size}))
{}

mx::array Gemma3VisionMLP::operator()(const mx::array& x) {
    // fc1 -> GELU precise -> fc2
    return linear_fwd(gelu_precise(linear_fwd(x, fc1_weight_, &fc1_bias_)),
                      fc2_weight_, &fc2_bias_);
}

std::unordered_map<std::string, mx::array*> Gemma3VisionMLP::weight_map() {
    return {
        {"fc1.weight", &fc1_weight_}, {"fc1.bias", &fc1_bias_},
        {"fc2.weight", &fc2_weight_}, {"fc2.bias", &fc2_bias_},
    };
}

// -- Vision Encoder Layer --

Gemma3VisionEncoderLayer::Gemma3VisionEncoderLayer(const Gemma3VisionConfiguration& config)
    : attention_(config.hidden_size, config.num_attention_heads),
      mlp_(config),
      layer_norm1_weight_(mx::ones({config.hidden_size})),
      layer_norm1_bias_(mx::zeros({config.hidden_size})),
      layer_norm2_weight_(mx::ones({config.hidden_size})),
      layer_norm2_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array Gemma3VisionEncoderLayer::operator()(const mx::array& x) {
    // Pre-norm: h = x + attn(layernorm1(x))
    auto h = mx::add(x,
        attention_(mx::fast::layer_norm(x, layer_norm1_weight_, layer_norm1_bias_, eps_)));
    // Pre-norm: h = h + mlp(layernorm2(h))
    h = mx::add(h,
        mlp_(mx::fast::layer_norm(h, layer_norm2_weight_, layer_norm2_bias_, eps_)));
    return h;
}

std::unordered_map<std::string, mx::array*> Gemma3VisionEncoderLayer::weight_map() {
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

Gemma3VisionEncoder::Gemma3VisionEncoder(const Gemma3VisionConfiguration& config) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i)
        layers_.emplace_back(config);
}

mx::array Gemma3VisionEncoder::operator()(const mx::array& x) {
    auto h = x;
    for (auto& layer : layers_) {
        h = layer(h);
    }
    return h;
}

std::unordered_map<std::string, mx::array*> Gemma3VisionEncoder::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- SigLip Vision Model --

Gemma3SigLipVisionModel::Gemma3SigLipVisionModel(const Gemma3VisionConfiguration& config)
    : embeddings_(config),
      encoder_(config),
      post_layernorm_weight_(mx::ones({config.hidden_size})),
      post_layernorm_bias_(mx::zeros({config.hidden_size})),
      eps_(config.layer_norm_eps)
{}

mx::array Gemma3SigLipVisionModel::operator()(const mx::array& x) {
    auto h = embeddings_(x);
    h = encoder_(h);
    return mx::fast::layer_norm(h, post_layernorm_weight_, post_layernorm_bias_, eps_);
}

std::unordered_map<std::string, mx::array*> Gemma3SigLipVisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : embeddings_.weight_map()) map["embeddings." + k] = v;
    for (auto& [k, v] : encoder_.weight_map()) map["encoder." + k] = v;
    map["post_layernorm.weight"] = &post_layernorm_weight_;
    map["post_layernorm.bias"] = &post_layernorm_bias_;
    return map;
}

// -- Vision Model Wrapper --

Gemma3VisionModel::Gemma3VisionModel(const Gemma3VisionConfiguration& config)
    : vision_model_(config),
      num_channels_(config.num_channels)
{}

mx::array Gemma3VisionModel::operator()(const mx::array& x) {
    // Input x: [B, C, H, W] from processor
    // Transpose to [B, H, W, C] for MLX conv/patch operations
    auto input = mx::transpose(x, {0, 2, 3, 1});
    return vision_model_(input);
}

std::unordered_map<std::string, mx::array> Gemma3VisionModel::sanitize(
    std::unordered_map<std::string, mx::array> weights)
{
    std::unordered_map<std::string, mx::array> sanitized;

    for (auto& [k, v] : weights) {
        if (k.find("patch_embedding.weight") != std::string::npos) {
            // PyTorch conv2d weight: [out, in, kH, kW]
            // MLX format: [out, kH, kW, in]
            if (v.ndim() == 4 && v.shape(1) == num_channels_) {
                sanitized.insert_or_assign(k, mx::transpose(v, {0, 2, 3, 1}));
            } else {
                sanitized.insert_or_assign(k, v);
            }
        } else {
            sanitized.insert_or_assign(k, v);
        }
    }

    return sanitized;
}

std::unordered_map<std::string, mx::array*> Gemma3VisionModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : vision_model_.weight_map()) map["vision_model." + k] = v;
    return map;
}

// ── Language Components (Gemma3-style) ─────────────────────────────────

// -- Language Attention --

Gemma3LanguageAttention::Gemma3LanguageAttention(const Gemma3TextConfiguration& args, int layer_idx)
    : heads_(args.num_attention_heads),
      kv_heads_(args.num_key_value_heads),
      head_dim_(args.head_dim),
      layer_idx_(layer_idx),
      // Gemma3 uses query_pre_attn_scalar for attention scale, NOT head_dim
      scale_(std::pow(args.query_pre_attn_scalar, -0.5f)),
      // Layer is sliding if (layer_idx + 1) % sliding_window_pattern != 0
      is_sliding_((layer_idx + 1) % args.sliding_window_pattern != 0),
      // Sliding layers use rope_local_base_freq, global layers use rope_theta
      rope_theta_(is_sliding_ ? args.rope_local_base_freq : args.rope_theta),
      rope_traditional_(args.rope_traditional),
      wq_weight_(mx::zeros({args.num_attention_heads * args.head_dim, args.hidden_size})),
      wk_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wv_weight_(mx::zeros({args.num_key_value_heads * args.head_dim, args.hidden_size})),
      wo_weight_(mx::zeros({args.hidden_size, args.num_attention_heads * args.head_dim})),
      q_norm_(args.head_dim, args.rms_norm_eps),
      k_norm_(args.head_dim, args.rms_norm_eps)
{}

mx::array Gemma3LanguageAttention::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    int B = x.shape(0), L = x.shape(1);

    auto queries = linear_fwd(x, wq_weight_);
    auto keys    = linear_fwd(x, wk_weight_);
    auto values  = linear_fwd(x, wv_weight_);

    // Reshape to [B, num_heads, L, head_dim]
    queries = mx::transpose(mx::reshape(queries, {B, L, heads_, head_dim_}), {0, 2, 1, 3});
    keys    = mx::transpose(mx::reshape(keys,    {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});
    values  = mx::transpose(mx::reshape(values,  {B, L, kv_heads_, head_dim_}), {0, 2, 1, 3});

    // Apply Q/K normalization (Gemma RMSNorm with 1+weight trick)
    queries = q_norm_(queries);
    keys = k_norm_(keys);

    // Apply RoPE with per-layer theta
    int offset = cache ? cache->offset() : 0;
    queries = mx::fast::rope(queries, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);
    keys    = mx::fast::rope(keys, head_dim_, rope_traditional_, rope_theta_, 1.0f, offset);

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

std::unordered_map<std::string, mx::array*> Gemma3LanguageAttention::weight_map() {
    std::unordered_map<std::string, mx::array*> map = {
        {"q_proj.weight", &wq_weight_},
        {"k_proj.weight", &wk_weight_},
        {"v_proj.weight", &wv_weight_},
        {"o_proj.weight", &wo_weight_},
    };
    // Q/K norms
    map["q_norm.weight"] = q_norm_.weight_ptr();
    map["k_norm.weight"] = k_norm_.weight_ptr();
    return map;
}

// -- Language MLP --

Gemma3LanguageMLP::Gemma3LanguageMLP(int dimensions, int hidden_dimensions)
    : gate_weight_(mx::zeros({hidden_dimensions, dimensions})),
      down_weight_(mx::zeros({dimensions, hidden_dimensions})),
      up_weight_(mx::zeros({hidden_dimensions, dimensions}))
{}

mx::array Gemma3LanguageMLP::operator()(const mx::array& x) {
    // down(gelu_approx(gate(x)) * up(x))
    return linear_fwd(mx::multiply(gelu_approx(linear_fwd(x, gate_weight_)),
                                    linear_fwd(x, up_weight_)),
                      down_weight_);
}

std::unordered_map<std::string, mx::array*> Gemma3LanguageMLP::weight_map() {
    return {
        {"gate_proj.weight", &gate_weight_},
        {"down_proj.weight", &down_weight_},
        {"up_proj.weight", &up_weight_},
    };
}

// -- Transformer Block --

Gemma3TransformerBlock::Gemma3TransformerBlock(const Gemma3TextConfiguration& args, int layer_idx)
    : attention_(args, layer_idx),
      mlp_(args.hidden_size, args.intermediate_size),
      input_layernorm_(args.hidden_size, args.rms_norm_eps),
      post_attention_layernorm_(args.hidden_size, args.rms_norm_eps),
      pre_feedforward_layernorm_(args.hidden_size, args.rms_norm_eps),
      post_feedforward_layernorm_(args.hidden_size, args.rms_norm_eps)
{}

mx::array Gemma3TransformerBlock::operator()(
    const mx::array& x,
    const AttentionMask& mask,
    KVCache* cache)
{
    // Gemma3 has 4 layer norms and uses clipResidual pattern:
    //   r = attn(inputLayerNorm(x))
    //   h = x + postAttentionLayerNorm(r)
    //   r2 = mlp(preFeedforwardLayerNorm(h))
    //   out = h + postFeedforwardLayerNorm(r2)
    auto r = attention_(input_layernorm_(x), mask, cache);
    auto h = mx::add(x, post_attention_layernorm_(r));
    auto r2 = mlp_(pre_feedforward_layernorm_(h));
    return mx::add(h, post_feedforward_layernorm_(r2));
}

std::unordered_map<std::string, mx::array*> Gemma3TransformerBlock::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : attention_.weight_map()) map["self_attn." + k] = v;
    for (auto& [k, v] : mlp_.weight_map()) map["mlp." + k] = v;
    map["input_layernorm.weight"] = input_layernorm_.weight_ptr();
    map["post_attention_layernorm.weight"] = post_attention_layernorm_.weight_ptr();
    map["pre_feedforward_layernorm.weight"] = pre_feedforward_layernorm_.weight_ptr();
    map["post_feedforward_layernorm.weight"] = post_feedforward_layernorm_.weight_ptr();
    return map;
}

// -- Language Model Inner --

Gemma3LanguageModelInner::Gemma3LanguageModelInner(const Gemma3TextConfiguration& args)
    : embed_tokens_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      norm_(args.hidden_size, args.rms_norm_eps),
      hidden_scale_(std::sqrt(static_cast<float>(args.hidden_size))),
      sliding_window_pattern_(args.sliding_window_pattern),
      sliding_window_(args.sliding_window)
{
    layers_.reserve(args.num_hidden_layers);
    for (int i = 0; i < args.num_hidden_layers; ++i)
        layers_.emplace_back(args, i);
}

mx::array Gemma3LanguageModelInner::operator()(
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

    // Gemma3 scales embeddings by sqrt(hidden_size)
    h = mx::multiply(h, mx::array(hidden_scale_));

    // Gemma3 uses different masks for global vs sliding window layers.
    // Global layers: full causal mask (use cache from a global layer for offset).
    // Sliding layers: windowed causal mask with sliding_window size.

    // Find a global layer to get the cache offset for global mask
    int global_layer_idx = sliding_window_pattern_ - 1; // first global layer
    KVCache* global_cache = (cache && global_layer_idx < static_cast<int>(cache->size()))
        ? &(*cache)[global_layer_idx] : nullptr;
    auto global_mask = create_attention_mask(h, global_cache);

    // Sliding window mask (only needed if sliding_window_pattern > 1)
    AttentionMask sliding_mask;
    if (sliding_window_pattern_ > 1) {
        KVCache* sliding_cache = (cache && !cache->empty()) ? &(*cache)[0] : nullptr;
        sliding_mask = create_attention_mask(h, sliding_cache,
                                             std::optional<int>(sliding_window_));
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
        bool is_global = (static_cast<int>(i) % sliding_window_pattern_ ==
                          sliding_window_pattern_ - 1);
        const auto& mask = is_global ? global_mask : sliding_mask;
        KVCache* lc = (cache && i < cache->size()) ? &(*cache)[i] : nullptr;
        h = layers_[i](h, mask, lc);
    }

    return norm_(h);
}

mx::array Gemma3LanguageModelInner::embed_tokens(const mx::array& ids) const {
    return mx::take(embed_tokens_weight_, ids, 0);
}

std::unordered_map<std::string, mx::array*> Gemma3LanguageModelInner::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    map["embed_tokens.weight"] = &embed_tokens_weight_;
    map["norm.weight"] = norm_.weight_ptr();
    for (size_t i = 0; i < layers_.size(); ++i) {
        auto prefix = "layers." + std::to_string(i) + ".";
        for (auto& [k, v] : layers_[i].weight_map()) map[prefix + k] = v;
    }
    return map;
}

// -- Language Model --

Gemma3LanguageModel::Gemma3LanguageModel(const Gemma3TextConfiguration& args)
    : model_(args),
      lm_head_weight_(mx::zeros({args.vocab_size, args.hidden_size})),
      final_logit_softcapping_(args.final_logit_softcapping),
      sliding_window_(args.sliding_window),
      sliding_window_pattern_(args.sliding_window_pattern),
      num_hidden_layers_(args.num_hidden_layers)
{
    kv_heads_.resize(args.num_hidden_layers, args.num_key_value_heads);
}

LMOutput Gemma3LanguageModel::operator()(
    const std::optional<mx::array>& inputs,
    std::vector<KVCache>* cache,
    const std::optional<mx::array>& input_embedding)
{
    auto out = model_(inputs, cache, input_embedding);

    // Gemma3 always uses a separate lm_head (not tied)
    out = linear_fwd(out, lm_head_weight_);

    // Apply final logit softcapping if configured
    if (final_logit_softcapping_.has_value()) {
        float softcap = final_logit_softcapping_.value();
        if (softcap > 0.0f) {
            auto scale = mx::array(softcap);
            out = mx::multiply(mx::tanh(mx::divide(out, scale)), scale);
        }
    }

    return LMOutput(out);
}

std::vector<KVCache> Gemma3LanguageModel::new_cache() const {
    std::vector<KVCache> caches;
    caches.reserve(num_hidden_layers_);
    int effective_sliding_window = (sliding_window_ > 0) ? sliding_window_ : 4096;

    for (int i = 0; i < num_hidden_layers_; ++i) {
        bool is_global = (i % sliding_window_pattern_ == sliding_window_pattern_ - 1);
        if (is_global) {
            caches.emplace_back(KVCacheSimple{});
        } else {
            caches.emplace_back(RotatingKVCache(effective_sliding_window, 0));
        }
    }
    return caches;
}

std::unordered_map<std::string, mx::array*> Gemma3LanguageModel::weight_map() {
    std::unordered_map<std::string, mx::array*> map;
    for (auto& [k, v] : model_.weight_map()) map["model." + k] = v;
    map["lm_head.weight"] = &lm_head_weight_;
    return map;
}

// ── Multimodal Projector ───────────────────────────────────────────────

Gemma3MultiModalProjector::Gemma3MultiModalProjector(const Gemma3Configuration& config)
    : mm_input_projection_weight_(mx::ones({config.vision_config.hidden_size,
                                             config.text_config.hidden_size})),
      mm_soft_emb_norm_(config.vision_config.hidden_size, config.vision_config.layer_norm_eps),
      patches_per_image_(config.vision_config.image_size / config.vision_config.patch_size),
      tokens_per_side_(static_cast<int>(std::sqrt(static_cast<double>(config.mm_tokens_per_image)))),
      kernel_size_(patches_per_image_ / tokens_per_side_)
{}

mx::array Gemma3MultiModalProjector::operator()(const mx::array& x) {
    // x: [B, num_patches, vision_hidden_size]
    int b = x.shape(0);
    int l = x.shape(2); // vision_hidden_size (channels)

    // Reshape for spatial avg pool:
    //   transpose: [B, vision_hidden_size, num_patches]
    //   reshape:   [B, vision_hidden_size, patches_per_image, patches_per_image]
    auto reshaped = mx::transpose(x, {0, 2, 1});
    reshaped = mx::reshape(reshaped, {b, l, patches_per_image_, patches_per_image_});

    // Transpose to [B, patches_per_image, patches_per_image, vision_hidden_size]
    reshaped = mx::transpose(reshaped, {0, 2, 3, 1});

    // AvgPool2d via reshape + mean over kernel dimensions:
    //   [B, H/k, k, W/k, k, C] -> mean over axes 2,4 -> [B, H/k, W/k, C]
    int out_h = patches_per_image_ / kernel_size_;
    int out_w = patches_per_image_ / kernel_size_;
    auto pooled = mx::reshape(reshaped, {b, out_h, kernel_size_, out_w, kernel_size_, l});
    pooled = mx::mean(pooled, {2, 4});

    // Transpose back: [B, C, out_h, out_w] -> flatten spatial -> [B, C, out_h*out_w] -> [B, out_h*out_w, C]
    pooled = mx::transpose(pooled, {0, 3, 1, 2});
    pooled = mx::reshape(pooled, {b, l, -1});
    pooled = mx::transpose(pooled, {0, 2, 1});

    // Apply Gemma RMSNorm (with 1+weight trick)
    auto normed = mm_soft_emb_norm_(pooled);

    // einsum "btm,md->btd" is matmul(normed, weight)
    // mm_input_projection_weight_ has shape [m, d] = [vision_hidden, text_hidden]
    auto projected = mx::matmul(normed, mm_input_projection_weight_);

    return mx::astype(projected, x.dtype());
}

std::unordered_map<std::string, mx::array*> Gemma3MultiModalProjector::weight_map() {
    return {
        {"mm_input_projection_weight", &mm_input_projection_weight_},
        {"mm_soft_emb_norm.weight", mm_soft_emb_norm_.weight_ptr()},
    };
}

// ── Top-Level Gemma3 Model ─────────────────────────────────────────────

Gemma3Model::Gemma3Model(const Gemma3Configuration& config)
    : config_(config),
      vision_tower_(config.vision_config),
      language_model_(config.text_config),
      multi_modal_projector_(config)
{
    kv_heads_cache_ = language_model_.kv_heads();
}

std::vector<KVCache> Gemma3Model::new_cache_impl(const GenerateParameters& /*params*/) const {
    // Gemma3 requires per-layer cache types (simple for global, rotating for sliding)
    return language_model_.new_cache();
}

PrepareResult Gemma3Model::prepare_impl(
    const LMInput& input, std::vector<KVCache>& cache, int /*window_size*/)
{
    auto input_ids = input.text.tokens;
    if (input_ids.ndim() == 1) {
        input_ids = mx::expand_dims(input_ids, 0);
    }

    if (!input.image.has_value()) {
        // Text-only: run through language model directly
        std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
        auto result = language_model_(input_ids, cache_ptr);
        return PrepareResult::logits(std::move(result));
    }

    // Get image pixels and run through vision tower
    auto pixel_values = input.image->pixels;
    auto vision_outputs = vision_tower_(pixel_values);

    // Project vision features to language model dimension
    auto image_features = multi_modal_projector_(vision_outputs);

    // Scale projected features by 1/sqrt(text_hidden_size)
    float inv_scale = 1.0f / std::sqrt(static_cast<float>(config_.text_config.hidden_size));
    image_features = mx::multiply(image_features, mx::array(inv_scale));

    // Get text embeddings (embed_tokens only, scaling happens inside the model)
    auto input_embeds = language_model_.inner().embed_tokens(input_ids);

    // Gemma3 scales embeddings by sqrt(hidden_size) — done inside the model forward,
    // but for the merge we need unscaled embeddings, then let the model scale them.
    // Actually, looking at the Swift code, embeddings are fetched unscaled and the
    // model forward scales them. But here we merge before the model's forward call,
    // so we should NOT scale the embeddings here — the model inner will scale h.
    // However the image features are already projected and scaled by 1/sqrt(hidden_size),
    // so they also need to be scaled by sqrt(hidden_size) when the model does its scaling.
    // The Swift code handles this differently: it passes inputEmbedding which bypasses
    // embed_tokens but still gets the hidden_scale multiplication.
    // So we should just merge and let the model handle scaling uniformly.

    // Ensure image_features has batch dimension
    if (image_features.ndim() == 2) {
        image_features = mx::expand_dims(image_features, 0);
    }

    // Merge image tokens into text embeddings using masking
    // Image token ID used after token expansion (262144)
    int image_token_id = 262144;
    int pad_token_id = config_.pad_token_id;
    int B = input_ids.shape(0);
    int L = input_ids.shape(1);
    int D = input_embeds.shape(-1);
    int num_image_tokens = image_features.shape(1);

    // Create masks
    auto image_mask = mx::equal(input_ids, mx::array(image_token_id));
    auto pad_mask = mx::equal(input_ids, mx::array(pad_token_id));

    // Zero out pad token positions
    auto pad_mask_expanded = mx::expand_dims(pad_mask, -1);
    input_embeds = mx::where(pad_mask_expanded,
                              mx::zeros_like(input_embeds),
                              input_embeds);

    // Build image-aligned embedding tensor for scatter
    // Use cumulative sum to map image token positions to sequential image feature indices
    auto mask_expanded = mx::expand_dims(image_mask, -1);
    auto cum_mask = mx::cumsum(mx::astype(image_mask, mx::int32), 1);
    auto gather_indices = mx::subtract(cum_mask, mx::array(1, mx::int32));
    gather_indices = mx::clip(gather_indices, mx::array(0, mx::int32),
                               mx::array(num_image_tokens - 1, mx::int32));

    auto idx_expanded = mx::expand_dims(gather_indices, -1);
    idx_expanded = mx::broadcast_to(idx_expanded, {B, L, D});

    // Gather image features at the right positions
    auto image_expanded = mx::take_along_axis(image_features, idx_expanded, 1);

    // Use where to merge: at image_token positions use image features, otherwise text
    auto final_embeds = mx::where(mask_expanded, image_expanded, input_embeds);

    // Run through language model with pre-computed embeddings
    std::vector<KVCache>* cache_ptr = cache.empty() ? nullptr : &cache;
    auto result = language_model_(std::nullopt, cache_ptr, final_embeds);

    return PrepareResult::logits(std::move(result));
}

LMOutput Gemma3Model::call_impl(
    const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State*)
{
    return language_model_(input.tokens, cache);
}

mx::array Gemma3Model::forward_impl(const mx::array& inputs, std::vector<KVCache>* cache) {
    return language_model_(inputs, cache).logits;
}

std::unordered_map<std::string, mx::array>
Gemma3Model::sanitize_impl(std::unordered_map<std::string, mx::array> weights) {
    // Remove rotary embedding inverse frequency keys
    std::unordered_map<std::string, mx::array> filtered;
    for (auto& [k, v] : weights) {
        if (k.find("rotary_emb.inv_freq") != std::string::npos) {
            continue;
        }
        filtered.insert_or_assign(k, v);
    }

    // If lm_head weight is missing, copy from embed_tokens
    if (filtered.find("language_model.lm_head.weight") == filtered.end()) {
        auto it = filtered.find("language_model.model.embed_tokens.weight");
        if (it != filtered.end()) {
            filtered.insert_or_assign(std::string("language_model.lm_head.weight"), it->second);
        }
    }

    // Sanitize vision conv weights (PyTorch [out,in,kH,kW] -> MLX [out,kH,kW,in])
    return vision_tower_.sanitize(std::move(filtered));
}

void Gemma3Model::load_weights(const std::unordered_map<std::string, mx::array>& weights) {
    auto wmap = weight_map();
    for (auto& [name, target] : wmap) {
        auto it = weights.find(name);
        if (it != weights.end()) *target = it->second;
    }
}

std::unordered_map<std::string, mx::array*> Gemma3Model::weight_map() {
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

} // namespace mlx_lm
