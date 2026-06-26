// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/gguf_loader.h>
#include <fstream>
#include <regex>

namespace mlx_lm {

namespace {

// GGUF magic bytes: 'GGUF'
constexpr uint32_t GGUF_MAGIC = 0x46475547;

// Check magic bytes at the start of the file
bool check_gguf_magic(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return f.gcount() == sizeof(magic) && magic == GGUF_MAGIC;
}

// Extract scalar value from array in GGUFMetaData variant
template <typename T>
T get_scalar_from_array(const mlx::core::array& arr) {
    // Must evaluate first for GPU arrays
    mlx::core::eval(arr);
    const T* ptr = arr.data<T>();
    return ptr[0];
}

// Extract int32 from GGUFMetaData
std::optional<int32_t> get_meta_int32(
    const mlx::core::GGUFMetaData& meta,
    bool* is_present = nullptr) {
    if (auto pv = std::get_if<mlx::core::array>(&meta)) {
        if (pv->size() == 1 && pv->dtype() == mlx::core::int32) {
            if (is_present) *is_present = true;
            return get_scalar_from_array<int32_t>(*pv);
        }
    }
    if (is_present) *is_present = false;
    return {};
}

// Extract int64 from GGUFMetaData
std::optional<int64_t> get_meta_int64(
    const mlx::core::GGUFMetaData& meta,
    bool* is_present = nullptr) {
    if (auto pv = std::get_if<mlx::core::array>(&meta)) {
        if (pv->size() == 1) {
            int64_t val = 0;
            mlx::core::Dtype dtype = pv->dtype();
            if (dtype == mlx::core::int32) {
                val = get_scalar_from_array<int32_t>(*pv);
            } else if (dtype == mlx::core::int64) {
                val = get_scalar_from_array<int64_t>(*pv);
            } else if (dtype == mlx::core::float32) {
                val = static_cast<int64_t>(get_scalar_from_array<float>(*pv));
            }
            if (is_present) *is_present = true;
            return val;
        }
    }
    if (is_present) *is_present = false;
    return {};
}

// Extract float from GGUFMetaData
std::optional<float> get_meta_float(
    const mlx::core::GGUFMetaData& meta,
    bool* is_present = nullptr) {
    if (auto pv = std::get_if<mlx::core::array>(&meta)) {
        if (pv->size() == 1) {
            float val = 0.0f;
            mlx::core::Dtype dtype = pv->dtype();
            if (dtype == mlx::core::float32) {
                val = get_scalar_from_array<float>(*pv);
            } else if (dtype == mlx::core::float16) {
                val = static_cast<float>(get_scalar_from_array<mlx::core::float16_t>(*pv));
            }
            if (is_present) *is_present = true;
            return val;
        }
    }
    if (is_present) *is_present = false;
    return {};
}

// Extract string from GGUFMetaData
std::optional<std::string> get_meta_string(
    const mlx::core::GGUFMetaData& meta,
    bool* is_present = nullptr) {
    if (auto pv = std::get_if<std::string>(&meta)) {
        if (is_present) *is_present = true;
        return *pv;
    }
    if (is_present) *is_present = false;
    return {};
}

// Helper to set JSON field if value is present
template <typename T>
void set_if_present(
    nlohmann::json& config,
    const std::string& key,
    const std::optional<T>& value) {
    if (value.has_value()) {
        config[key] = value.value();
    }
}

// Get architecture prefix from architecture name
std::string get_arch_prefix(const std::string& arch) {
    if (arch == "llama") return "llama";
    if (arch == "qwen2") return "qwen2";
    if (arch == "mistral") return "mistral";
    if (arch == "mixtral") return "mixtral";
    if (arch == "gemma") return "gemma";
    if (arch == "phi") return "phi";
    if (arch == "qwen") return "qwen";
    if (arch == "stablelm") return "stablelm";
    if (arch == "starcoder") return "starcoder";
    if (arch == "mamba") return "mamba";
    // Default to llama-style keys for unknown architectures
    return "llama";
}

// Remap GGUF tensor names to HuggingFace names
std::string remap_tensor_name(const std::string& gguf_name) {
    static const std::vector<std::pair<std::regex, std::string>> patterns = {
        // Embedding and output layers
        {std::regex(R"(^(token_embd)\.(\w+)$)"), "model.embed_tokens.$2"},
        {std::regex(R"(^(output_norm)\.(\w+)$)"), "model.norm.$2"},
        {std::regex(R"(^(output)\.(\w+)$)"), "lm_head.$2"},
        
        // Attention projections
        {std::regex(R"(^blk\.(\d+)\.(attn_q)\.(\w+)$)"), "model.layers.$1.self_attn.q_proj.$3"},
        {std::regex(R"(^blk\.(\d+)\.(attn_k)\.(\w+)$)"), "model.layers.$1.self_attn.k_proj.$3"},
        {std::regex(R"(^blk\.(\d+)\.(attn_v)\.(\w+)$)"), "model.layers.$1.self_attn.v_proj.$3"},
        {std::regex(R"(^blk\.(\d+)\.(attn_output)\.(\w+)$)"), "model.layers.$1.self_attn.o_proj.$3"},
        
        // FFN layers
        {std::regex(R"(^blk\.(\d+)\.(ffn_gate)\.(\w+)$)"), "model.layers.$1.mlp.gate_proj.$3"},
        {std::regex(R"(^blk\.(\d+)\.(ffn_up)\.(\w+)$)"), "model.layers.$1.mlp.up_proj.$3"},
        {std::regex(R"(^blk\.(\d+)\.(ffn_down)\.(\w+)$)"), "model.layers.$1.mlp.down_proj.$3"},
        
        // Layer norms
        {std::regex(R"(^blk\.(\d+)\.(attn_norm)\.(\w+)$)"), "model.layers.$1.input_layernorm.$3"},
        {std::regex(R"(^blk\.(\d+)\.(ffn_norm)\.(\w+)$)"), "model.layers.$1.post_attention_layernorm.$3"},
    };

    for (const auto& [pattern, replacement] : patterns) {
        std::smatch match;
        if (std::regex_match(gguf_name, match, pattern)) {
            std::string result = replacement;
            // Replace $1, $2 etc with captured groups
            for (size_t i = 1; i < match.size(); ++i) {
                std::string placeholder = "$" + std::to_string(i);
                size_t pos;
                while ((pos = result.find(placeholder)) != std::string::npos) {
                    result.replace(pos, placeholder.length(), match[i].str());
                }
            }
            return result;
        }
    }
    
    // No match found, return original name
    return gguf_name;
}

} // anonymous namespace

bool is_gguf_file(const std::string& path) {
    // Check file extension first
    if (path.size() >= 5 && 
        (path.substr(path.size() - 5) == ".gguf" || 
         path.substr(path.size() - 5) == ".GGUF")) {
        return true;
    }
    // Fall back to magic bytes check
    return check_gguf_magic(path);
}

nlohmann::json gguf_config_from_metadata(
    const std::unordered_map<std::string, mlx::core::GGUFMetaData>& meta) {
    nlohmann::json config;

    // Get architecture to determine key prefixes
    std::string arch_prefix = "llama.";
    bool arch_found = false;
    if (auto it = meta.find("general.architecture"); it != meta.end()) {
        if (auto arch = get_meta_string(it->second)) {
            arch_prefix = get_arch_prefix(*arch) + ".";
            arch_found = true;
        }
    }
    
    if (arch_found) {
        config["model_type"] = arch_prefix.substr(0, arch_prefix.size() - 1);
    }

    // Model dimensions
    auto emb_it = meta.find(arch_prefix + "embedding_length");
    if (emb_it != meta.end()) {
        set_if_present<int64_t>(
            config, "hidden_size", 
            get_meta_int64(emb_it->second));
    }
    
    auto blk_it = meta.find(arch_prefix + "block_count");
    if (blk_it != meta.end()) {
        set_if_present<int64_t>(
            config, "num_hidden_layers",
            get_meta_int64(blk_it->second));
    }
    
    auto head_it = meta.find(arch_prefix + "attention.head_count");
    if (head_it != meta.end()) {
        set_if_present<int64_t>(
            config, "num_attention_heads",
            get_meta_int64(head_it->second));
    }
    
    auto kv_it = meta.find(arch_prefix + "attention.head_count_kv");
    if (kv_it != meta.end()) {
        set_if_present<int64_t>(
            config, "num_key_value_heads",
            get_meta_int64(kv_it->second));
    }
    
    auto ctx_it = meta.find(arch_prefix + "context_length");
    if (ctx_it != meta.end()) {
        set_if_present<int64_t>(
            config, "max_position_embeddings",
            get_meta_int64(ctx_it->second));
    }
    
    auto rope_it = meta.find(arch_prefix + "rope.dimension_count");
    if (rope_it != meta.end()) {
        set_if_present<int64_t>(
            config, "head_dim",
            get_meta_int64(rope_it->second));
    }
    
    auto norm_it = meta.find(arch_prefix + "attention.layer_norm_rms_epsilon");
    if (norm_it != meta.end()) {
        set_if_present<float>(
            config, "rms_norm_eps",
            get_meta_float(norm_it->second));
    }
    
    auto bos_it = meta.find("tokenizer.ggml.bos_token_id");
    if (bos_it != meta.end()) {
        set_if_present<int64_t>(
            config, "bos_token_id",
            get_meta_int64(bos_it->second));
    }
    
    auto eos_it = meta.find("tokenizer.ggml.eos_token_id");
    if (eos_it != meta.end()) {
        set_if_present<int64_t>(
            config, "eos_token_id",
            get_meta_int64(eos_it->second));
    }

    return config;
}

std::unordered_map<std::string, mlx::core::array>
load_gguf_weights(const std::string& path) {
    auto [weights, metadata] = mlx::core::load_gguf(path);
    
    std::unordered_map<std::string, mlx::core::array> remapped_weights;
    remapped_weights.reserve(weights.size());
    for (const auto& [name, arr] : weights) {
        std::string hf_name = remap_tensor_name(name);
        remapped_weights.insert({hf_name, arr});
    }
    
    return remapped_weights;
}

} // namespace mlx_lm
