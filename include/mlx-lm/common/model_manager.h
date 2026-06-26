// Model manager for auto-discovering and loading MLX models.
// Compatible with HuggingFace Hub cache layout and Lemonade SDK conventions.
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/model_container.h>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

struct AvailableModel {
    std::string model_id;     // HF repo ID, e.g. "mlx-community/Qwen3-4B-4bit"
    std::string local_path;   // Absolute path to snapshot directory
    std::string model_type;   // From config.json, e.g. "qwen3"
    bool loaded = false;      // Currently in GPU memory?
};

class ModelManager {
public:
    ModelManager();

    // Add an already-loaded model (e.g. from startup preload).
    void add_loaded(const std::string& model_id,
                    std::shared_ptr<ModelContainer> container);

    // Get a loaded model by name. If not loaded, auto-loads it
    // (downloading from HF Hub if needed and downloads are enabled).
    // Evicts LRU model if at capacity.
    std::shared_ptr<ModelContainer> get_or_load(const std::string& model_id);

    // List all available MLX models (cached locally).
    // Includes loaded status for each.
    std::vector<AvailableModel> list_available() const;

    // List currently loaded model IDs.
    std::vector<std::string> list_loaded() const;

    // Get the default model ID (first loaded model, or empty).
    std::string default_model_id() const;

    // Unload a specific model.
    void unload(const std::string& model_id);

    // Unload all models.
    void unload_all();

    // Configuration.
    void set_max_loaded(int n) { max_loaded_ = n; }
    void set_default_params(const GenerateParameters& p) { default_params_ = p; }
    void set_no_download(bool v) { no_download_ = v; }
    void set_no_think(bool v) { no_think_ = v; }
    void set_auto_quantize(bool v) { auto_quantize_ = v; }

private:
    struct LoadedModel {
        std::shared_ptr<ModelContainer> container;
        int64_t last_access = 0; // unix timestamp
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, LoadedModel> loaded_;
    std::string first_loaded_; // tracks insertion order for default
    int max_loaded_ = 1;
    GenerateParameters default_params_;
    bool no_download_ = false;
    bool no_think_ = false;
    bool auto_quantize_ = false;

    void evict_lru_if_needed();
    static int64_t now_ts();
};

} // namespace mlx_lm
