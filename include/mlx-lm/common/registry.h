// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mlx_lm {

// ModelTypeRegistry maps model_type strings (e.g. "llama", "qwen2") to
// factory functions that create a model from JSON config data.
//
// The factory function takes raw JSON config bytes and returns a type-erased
// model context (via ModelContext binding).
class ModelTypeRegistry {
public:
    using CreatorFn = std::function<void*(const std::string& config_json)>;

    ModelTypeRegistry() = default;

    explicit ModelTypeRegistry(
        std::unordered_map<std::string, CreatorFn> creators)
        : creators_(std::move(creators)) {}

    void register_type(const std::string& model_type, CreatorFn creator) {
        creators_[model_type] = std::move(creator);
    }

    bool has_type(const std::string& model_type) const {
        return creators_.count(model_type) > 0;
    }

    // Create a model from config JSON. Returns an owning pointer.
    // Caller is responsible for casting to the correct type.
    void* create(const std::string& model_type, const std::string& config_json) const {
        auto it = creators_.find(model_type);
        if (it == creators_.end()) {
            throw std::runtime_error("Unknown model type: " + model_type);
        }
        return it->second(config_json);
    }

    const std::unordered_map<std::string, CreatorFn>& all() const { return creators_; }

private:
    std::unordered_map<std::string, CreatorFn> creators_;
};

// ModelConfiguration holds metadata about a specific model.
struct ModelConfiguration {
    std::string id;
    std::string default_prompt;
    std::optional<std::string> override_tokenizer;
    std::vector<std::string> extra_eos_tokens;
    std::optional<std::vector<int>> eos_token_ids;
    bool auto_quantize = false; // Auto-quantize unquantized bf16/fp16 weights to 4-bit at load time
};

// AbstractModelRegistry maps model IDs to ModelConfiguration.
class AbstractModelRegistry {
public:
    AbstractModelRegistry() = default;

    explicit AbstractModelRegistry(std::vector<ModelConfiguration> configs) {
        for (auto& c : configs) {
            configs_[c.id] = std::move(c);
        }
    }

    std::optional<ModelConfiguration> find(const std::string& id) const {
        auto it = configs_.find(id);
        if (it != configs_.end()) return it->second;
        return std::nullopt;
    }

    void register_model(ModelConfiguration config) {
        configs_[config.id] = std::move(config);
    }

private:
    std::unordered_map<std::string, ModelConfiguration> configs_;
};

// Architecture registration for custom/unknown model types.
// Users can register new architectures at runtime via JSON files
// without modifying C++ code.
struct ArchitectureRegistration {
    std::string model_type;           // e.g. "my_new_model"
    std::string base_model;           // e.g. "llama" (must match llm_loaders key)
    std::vector<std::pair<std::string, std::string>> key_remaps;  // old_prefix -> new_prefix
    std::unordered_map<std::string, std::string> config_defaults; // injected config values
    std::vector<std::string> skip_keys;  // weight keys to remove
    int activation_bits = 0;
    bool has_sub_norm = false;
};

// Architecture registry — maps model_type to runtime architecture registration.
// Populated by ArchitectureRegistrar or loaded from a JSON file.
// Consulted by llm_factory when a model_type is not in the hardcoded loaders.
class ArchitectureRegistry {
public:
    static ArchitectureRegistry& instance() {
        static ArchitectureRegistry reg;
        return reg;
    }

    void register_architecture(const ArchitectureRegistration& arch) {
        arches_[arch.model_type] = arch;
    }

    const ArchitectureRegistration* find(const std::string& model_type) const {
        auto it = arches_.find(model_type);
        return (it != arches_.end()) ? &it->second : nullptr;
    }

    // Load architectures from a JSON file.
    // Format:
    // [{"model_type": "foo", "base_model": "llama",
    //   "key_remaps": [["old", "new"], ...],
    //   "config_defaults": {"hidden_act": "gelu"},
    //   "skip_keys": ["rotary_emb.inv_freq"],
    //   "activation_bits": 8,
    //   "has_sub_norm": true}]
    void load_from_file(const std::string& path);

    // Get all registered architectures.
    const std::unordered_map<std::string, ArchitectureRegistration>& all() const {
        return arches_;
    }

private:
    ArchitectureRegistry() = default;
    std::unordered_map<std::string, ArchitectureRegistration> arches_;
};

} // namespace mlx_lm
