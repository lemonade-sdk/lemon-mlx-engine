// BitNet 1.58-bit model for lemon-mlx-engine
// Architecture: Llama variant with relu2 activation and ternary weights
// "Little bones" — Gord Downie
#pragma once

#include <mlx-lm/llm/models/llama.h>

namespace mlx_lm {

// BitNet uses the same configuration as Llama
// The differences are handled in the model implementation:
// - relu_squared activation instead of SiLU
// - sub-layer norms (attn_sub_norm, ffn_sub_norm)
// - ternary weights (packed uint8 with global scale)
using BitNetConfiguration = LlamaConfiguration;

void from_json(const nlohmann::json& j, BitNetConfiguration& c);

// BitNet Attention — identical to Llama but with sub-layer norm
class BitNetAttention {
    const BitNetConfiguration& args_;
    float scale_;
    LlamaDynamicNTKScalingRoPE rope_;
    QuantizedLinear q_proj_, k_proj_, v_proj_, o_proj_;
    mlx::core::array attn_sub_norm_weight_;

public:
    BitNetAttention(const BitNetConfiguration& args);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x, const mlx::core::array& mask,
                                KVCache* cache);
};

// BitNet MLP — relu_squared instead of SiLU, with sub-layer norm
class BitNetMLP {
    QuantizedLinear gate_proj_, down_proj_, up_proj_;
    mlx::core::array ffn_sub_norm_weight_;

public:
    BitNetMLP(int hidden_size, int intermediate_size);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x);
};

// BitNet Decoder Layer
class BitNetDecoderLayer {
    BitNetAttention self_attn_;
    BitNetMLP mlp_;
    mlx::core::array input_layernorm_weight_;
    mlx::core::array post_attention_layernorm_weight_;
    float rms_norm_eps_;

public:
    BitNetDecoderLayer(const BitNetConfiguration& args);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights,
                      const std::string& prefix);
    mlx::core::array operator()(const mlx::core::array& x, const mlx::core::array& mask,
                                KVCache* cache);
};

// BitNet Model
class BitNetModel : public LanguageModel {
    std::vector<BitNetDecoderLayer> layers_;
    mlx::core::array embed_tokens_;
    mlx::core::array norm_weight_;
    std::optional<mlx::core::array> lm_head_weight_;
    BitNetConfiguration config_;

public:
    BitNetModel(const BitNetConfiguration& config);
    void load_weights(const std::unordered_map<std::string, mlx::core::array>& weights) override;
    mlx::core::array operator()(const mlx::core::array& inputs, KVCacheVector& cache) override;
    int vocab_size() const override { return config_.vocab_size; }
};

} // namespace mlx_lm
