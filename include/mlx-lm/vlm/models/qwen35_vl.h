// Copyright © 2024-2025 Apple Inc. — Ported to C++
// Qwen3.5 Vision-Language Model
#pragma once

#include <mlx-lm/common/attention_utils.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/string_utils.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/vlm/vlm_model.h>
#include <mlx-lm/vlm/qwen_vl_utils.h>
#include <mlx-lm/llm/models/qwen35_moe.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// ── Vision Components ──────────────────────────────────────────────────

// Vision PatchEmbed: Conv3d implemented as reshape + matmul
class Qwen35VLPatchEmbed {
    mlx::core::array proj_weight_; // Conv3d kernel [out, T, H, W, C]
    mlx::core::array proj_bias_;
    int patch_size_, temporal_patch_size_, in_channels_, hidden_size_;

public:
    Qwen35VLPatchEmbed(int patch_size, int temporal_patch_size, int in_channels, int hidden_size);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision PatchMerger: LayerNorm + Linear + gelu_tanh + Linear
// Uses config.merger_intermediate_size (0 = auto: hidden * spatial_merge^2)
class Qwen35VLPatchMerger {
    int hidden_size_;
    int merger_intermediate_size_;
    mlx::core::array norm_weight_, norm_bias_; // LayerNorm
    mlx::core::array linear_fc1_weight_, linear_fc1_bias_;
    mlx::core::array linear_fc2_weight_, linear_fc2_bias_;
    float eps_;

public:
    Qwen35VLPatchMerger(const Qwen35VLVisionConfiguration& config);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Attention: combined QKV with bias, proj without bias
class Qwen35VLVisionAttention {
    int num_heads_;
    int head_dim_;
    float scale_;
    mlx::core::array qkv_weight_, qkv_bias_;
    mlx::core::array proj_weight_;

public:
    Qwen35VLVisionAttention(int dims, int num_heads);
    mlx::core::array operator()(
        const mlx::core::array& x,
        const std::vector<int>& seqlens,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision MLP: Linear + gelu_tanh + Linear
class Qwen35VLVisionMLP {
    mlx::core::array linear_fc1_weight_, linear_fc1_bias_;
    mlx::core::array linear_fc2_weight_, linear_fc2_bias_;

public:
    Qwen35VLVisionMLP(int dimensions, int hidden_dimensions);
    mlx::core::array operator()(const mlx::core::array& x);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Transformer Block
class Qwen35VLVisionBlock {
    Qwen35VLVisionAttention attention_;
    Qwen35VLVisionMLP mlp_;
    mlx::core::array norm1_weight_, norm1_bias_;
    mlx::core::array norm2_weight_, norm2_bias_;
    float eps_;

public:
    explicit Qwen35VLVisionBlock(const Qwen35VLVisionConfiguration& config);
    mlx::core::array operator()(
        const mlx::core::array& hidden_states,
        const std::vector<int>& seqlens,
        const mlx::core::array& rotary_pos_emb);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// Vision Model: PatchEmbed + PosEmbed + VisionBlocks + PatchMerger
class Qwen35VLVisionModel {
    Qwen35VLPatchEmbed patch_embed_;
    qwen_vl::VisionRotaryEmbedding rotary_pos_emb_;
    mlx::core::array pos_embed_weight_; // Learned positional embeddings [N, D]
    std::vector<Qwen35VLVisionBlock> blocks_;
    Qwen35VLPatchMerger merger_;
    std::vector<Qwen35VLPatchMerger> deepstack_mergers_;
    std::vector<int> deepstack_visual_indexes_;
    int spatial_merge_size_;
    int num_grid_per_side_;
    int hidden_size_;
    int in_channels_;
    int rotary_half_dim_;

    // Compute rotary position embeddings for vision tokens
    mlx::core::array compute_rotary_pos_emb(const std::vector<THW>& grids);
    // Compute learned positional embeddings with bilinear interpolation
    mlx::core::array compute_positional_embeddings(const std::vector<THW>& grids);
    // Compute cumulative sequence lengths for block-diagonal attention (host-side)
    std::vector<int> compute_cu_seqlens_host(const std::vector<THW>& grids);

public:
    explicit Qwen35VLVisionModel(const Qwen35VLVisionConfiguration& config);

    // Returns (merged_features, deepstack_outputs)
    std::pair<mlx::core::array, std::vector<mlx::core::array>>
    operator()(const mlx::core::array& pixel_values, const std::vector<THW>& grid_thw);

    std::unordered_map<std::string, mlx::core::array> sanitize(
        std::unordered_map<std::string, mlx::core::array> weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// ── Language Components ────────────────────────────────────────────────

// Language Model wrapping Qwen35MoEModelInner with multimodal support
class Qwen35VLLanguageModel {
    Qwen35MoEModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;
    Qwen35VLConfiguration config_;

    // Persistent rope deltas between prefill and generation
    std::optional<mlx::core::array> rope_deltas_;

public:
    explicit Qwen35VLLanguageModel(const Qwen35VLConfiguration& config);

    LMOutput operator()(
        const std::optional<mlx::core::array>& inputs,
        std::vector<KVCache>* cache = nullptr,
        const std::optional<mlx::core::array>& input_embedding = std::nullopt,
        const AttentionMask& mask = AttentionMask{},
        const std::optional<mlx::core::array>& position_ids = std::nullopt,
        const std::optional<mlx::core::array>& visual_mask = std::nullopt,
        const std::vector<mlx::core::array>* deepstack_embeds = nullptr,
        const mlx::core::array* pixel_values = nullptr,
        const std::vector<THW>* image_grid_thw = nullptr,
        const std::vector<THW>* video_grid_thw = nullptr);

    const std::vector<int>& kv_heads() const { return kv_heads_; }
    Qwen35MoEModelInner& inner() { return model_; }
    std::unordered_map<std::string, mlx::core::array*> weight_map();

    // Compute MRoPE position IDs for multimodal input
    static std::pair<mlx::core::array, mlx::core::array> get_rope_index(
        const mlx::core::array& input_ids,
        const std::vector<THW>* image_grid_thw,
        const std::vector<THW>* video_grid_thw,
        int spatial_merge_size,
        int image_token_id,
        int video_token_id,
        int vision_start_token_id,
        const mlx::core::array* attention_mask = nullptr);
};

// ── Top-Level Model ────────────────────────────────────────────────────

class Qwen35VLModel
    : public VLMModel<Qwen35VLModel>,
      public KVCacheDimensionProvider<Qwen35VLModel> {

    friend class LanguageModel<Qwen35VLModel>;
    friend class KVCacheDimensionProvider<Qwen35VLModel>;

    Qwen35VLConfiguration config_;
    Qwen35VLVisionModel vision_tower_;
    Qwen35VLLanguageModel language_model_;
    std::vector<int> kv_heads_cache_;

    // Merge vision features into text embeddings, returning (embeddings, visual_mask)
    std::pair<mlx::core::array, mlx::core::array> merge_input_ids_with_image_features(
        const mlx::core::array& image_features,
        const mlx::core::array& input_embeds,
        const mlx::core::array& input_ids,
        int image_token_index,
        int video_token_index);

    // CRTP implementations
    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache, const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array> sanitize_impl(
        std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit Qwen35VLModel(const Qwen35VLConfiguration& config);
    const std::vector<int>& kv_heads() const { return kv_heads_cache_; }
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
