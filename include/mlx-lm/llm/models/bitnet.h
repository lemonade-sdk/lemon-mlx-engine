// BitNet 1.58-bit model — Llama variant with ternary weights and relu² activation.
// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/bitnet.py
//
// Architecture: Llama with three differences:
//   1. relu_squared activation instead of silu
//   2. Sub-layer norms (attn_sub_norm before o_proj, ffn_sub_norm before down_proj)
//   3. Ternary weights {-1, 0, +1} packed as uint8 (4 values per byte), dequantized at load time
//
// Config reuses LlamaConfiguration since all fields are identical.
#pragma once

#include <mlx-lm/llm/models/llama.h>
#include <mlx/mlx.h>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

// BitNet reuses Llama's configuration and JSON deserializer.
using BitNetConfiguration = LlamaConfiguration;

// Dequantize uint8 packed ternary weights to float16.
// Each byte packs 4 ternary values as 2-bit values: 0→-1, 1→0, 2→+1.
// Result is multiplied by weight_scale.
mlx::core::array dequantize_bitnet_weight(
    const mlx::core::array& packed_weight,
    const mlx::core::array& weight_scale,
    int out_features);

// --- BitNet Attention (relu² + sub-layer norm) ---

class BitNetAttention {
    const BitNetConfiguration& args_;
    float scale_;
    LlamaDynamicNTKScalingRoPE rope_;

    mlx::core::array wq_weight_;
    mlx::core::array wk_weight_;
    mlx::core::array wv_weight_;
    mlx::core::array wo_weight_;
    mlx::core::array attn_sub_norm_weight_;

    mlx::core::array linear(const mlx::core::array& x,
                            const mlx::core::array& weight) const;

public:
    explicit BitNetAttention(const BitNetConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- BitNet MLP (relu² activation + sub-layer norm) ---

class BitNetMLP {
    mlx::core::array gate_weight_;
    mlx::core::array down_weight_;
    mlx::core::array up_weight_;
    mlx::core::array ffn_sub_norm_weight_;
    float rms_norm_eps_;

    mlx::core::array linear(const mlx::core::array& x,
                            const mlx::core::array& weight) const;
    mlx::core::array rms_norm(const mlx::core::array& x,
                               const mlx::core::array& weight) const;

public:
    explicit BitNetMLP(const BitNetConfiguration& args);

    mlx::core::array operator()(const mlx::core::array& x);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- BitNet Transformer Block ---

class BitNetTransformerBlock {
    BitNetAttention attention_;
    BitNetMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

    mlx::core::array rms_norm(const mlx::core::array& x,
                               const mlx::core::array& weight) const;

public:
    explicit BitNetTransformerBlock(const BitNetConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& x,
        const AttentionMask& mask,
        KVCache* cache);

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- BitNet Model Inner ---

class BitNetModelInner {
    mlx::core::array embed_tokens_weight_;
    std::vector<BitNetTransformerBlock> layers_;
    mlx::core::array norm_weight_;
    float rms_norm_eps_;

    mlx::core::array rms_norm(const mlx::core::array& x,
                               const mlx::core::array& weight) const;

public:
    explicit BitNetModelInner(const BitNetConfiguration& args);

    mlx::core::array operator()(
        const mlx::core::array& inputs,
        std::vector<KVCache>* cache = nullptr);

    mlx::core::array embed_as_linear(const mlx::core::array& x) const;

    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

// --- BitNet Model (top-level, CRTP) ---

class BitNetModel
    : public LanguageModel<BitNetModel>,
      public KVCacheDimensionProvider<BitNetModel> {

    friend class LanguageModel<BitNetModel>;
    friend class KVCacheDimensionProvider<BitNetModel>;

    BitNetConfiguration config_;
    BitNetModelInner model_;
    std::optional<mlx::core::array> lm_head_weight_;
    std::vector<int> kv_heads_;

    PrepareResult prepare_impl(const LMInput& input, std::vector<KVCache>& cache, int window_size);
    LMOutput call_impl(const LMInput::Text& input, std::vector<KVCache>* cache,
                       const LMOutput::State* state);
    mlx::core::array forward_impl(const mlx::core::array& inputs, std::vector<KVCache>* cache);
    std::unordered_map<std::string, mlx::core::array>
    sanitize_impl(std::unordered_map<std::string, mlx::core::array> weights);

public:
    explicit BitNetModel(const BitNetConfiguration& args);

    const std::vector<int>& kv_heads() const { return kv_heads_; }
    int vocab_size() const { return config_.vocab_size; }

    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights);
    std::unordered_map<std::string, mlx::core::array*> weight_map();
};

} // namespace mlx_lm
