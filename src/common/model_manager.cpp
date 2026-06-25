// Model manager: auto-discovery, loading, and LRU management of MLX models.

#include <mlx-lm/common/model_manager.h>
#include <mlx-lm/common/hub_api.h>
#include <mlx-lm/common/base_config.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/llm/llm_factory.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
namespace mx = mlx::core;

namespace mlx_lm {

ModelManager::ModelManager() = default;

int64_t ModelManager::now_ts() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void ModelManager::add_loaded(const std::string& model_id,
                               std::shared_ptr<ModelContainer> container) {
    std::lock_guard<std::mutex> lock(mutex_);
    loaded_[model_id] = {std::move(container), now_ts()};
    if (first_loaded_.empty()) first_loaded_ = model_id;
}

std::shared_ptr<ModelContainer> ModelManager::get_or_load(const std::string& model_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = loaded_.find(model_id);
        if (it != loaded_.end()) {
            it->second.last_access = now_ts();
            return it->second.container;
        }

        // Short-name alias: when a model was loaded from a local path
        // (e.g. /home/bcloud/models/llama-1b), requests with just the
        // basename ("llama-1b") should resolve to it.
        for (const auto& [loaded_id, lm] : loaded_) {
            fs::path loaded_path(loaded_id);
            if (loaded_path.is_absolute() && loaded_path.filename() == model_id) {
                std::cerr << "[ModelManager] Resolved short name \"" << model_id
                          << "\" -> \"" << loaded_id << "\"\n";
                // Return the container for the alias match.
                auto container = lm.container;
                // Update last_access on the canonical entry.
                loaded_[loaded_id].last_access = now_ts();
                return container;
            }
        }
    }

    // Not loaded — resolve and load outside the lock (loading is slow).
    std::cerr << "[ModelManager] Loading model: " << model_id << "\n";

    auto& hub = HubApi::shared();

    // Check if model is cached locally or can be downloaded.
    std::string model_dir;
    if (fs::exists(fs::path(model_id) / "config.json")) {
        // model_id is a local path
        model_dir = model_id;
    } else if (hub.is_cached(model_id)) {
        model_dir = hub.model_directory(model_id);
    } else if (!no_download_) {
        std::cerr << "[ModelManager] Model not cached, downloading: " << model_id << "\n";
        model_dir = hub.snapshot_download(model_id);
    } else {
        throw std::runtime_error("Model not found locally: " + model_id +
                                 " (auto-download disabled)");
    }

    // Check model_type to determine loading path.
    // MTP delta models (qwen3_5_mtp) contain only the MTP head weights and
    // must be merged with the base model's text backbone before loading.
    bool is_mtp_delta = false;
    {
        auto config_path = fs::path(model_dir) / "config.json";
        if (fs::exists(config_path)) {
            try {
                std::ifstream cf(config_path);
                nlohmann::json cj;
                cf >> cj;
                std::string mt = cj.value("model_type", "");
                is_mtp_delta = (mt == "qwen3_5_mtp");
            } catch (...) {}
        }
    }

    // Load the model.
    ModelContext ctx;
    if (is_mtp_delta) {
        std::cerr << "[ModelManager] MTP delta model detected, loading with base model merge\n";
        ctx = load_mtp_delta_model(model_id);
    } else {
        ctx = load_llm(model_id);
    }

    // Apply no-think if configured.
    if (no_think_ && ctx.template_extra_context) {
        (*ctx.template_extra_context)["enable_thinking"] = false;
    }

    // Warmup: prime GPU allocator cache.
    {
        GenerateParameters warmup_params;
        warmup_params.max_tokens = 1;
        warmup_params.temperature = 0.0f;
        auto warmup_cache = ctx.new_cache_fn(warmup_params);
        mx::array dummy_tokens = mx::reshape(mx::array({1}), {1, 1});
        LMInput::Text warmup_text(dummy_tokens);
        auto warmup_out = ctx.call_fn(warmup_text, &warmup_cache, nullptr);
        mx::eval(warmup_out.logits);
    }

    std::cerr << "[ModelManager] Model loaded. Memory: active="
              << mx::get_active_memory() / (1024 * 1024) << " MB, peak="
              << mx::get_peak_memory() / (1024 * 1024) << " MB\n";

    auto container = std::make_shared<ModelContainer>(std::move(ctx));

    // Now take the lock, evict if needed, and insert.
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check again in case another thread loaded it while we were working.
        auto it = loaded_.find(model_id);
        if (it != loaded_.end()) {
            it->second.last_access = now_ts();
            return it->second.container;
        }

        evict_lru_if_needed();
        loaded_[model_id] = {container, now_ts()};
        if (first_loaded_.empty()) first_loaded_ = model_id;
    }

    return container;
}

std::vector<AvailableModel> ModelManager::list_available() const {
    auto& hub = HubApi::shared();
    auto cached = hub.discover_cached_models();

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<AvailableModel> result;
    result.reserve(cached.size());

    for (const auto& cm : cached) {
        AvailableModel m;
        m.model_id = cm.model_id;
        m.local_path = cm.local_path;
        m.loaded = loaded_.count(cm.model_id) > 0;

        // Read model_type from config.json
        auto config_path = cm.local_path + "/config.json";
        if (fs::exists(config_path)) {
            try {
                std::ifstream f(config_path);
                nlohmann::json j;
                f >> j;
                m.model_type = j.value("model_type", "");
            } catch (...) {}
        }

        result.push_back(std::move(m));
    }

    // Also include loaded models that weren't in the cache scan
    // (e.g. loaded from a local directory path).
    for (const auto& [id, lm] : loaded_) {
        bool found = false;
        for (const auto& m : result) {
            if (m.model_id == id) { found = true; break; }
        }
        if (!found) {
            AvailableModel m;
            m.model_id = id;
            m.loaded = true;
            result.push_back(std::move(m));
        }
    }

    return result;
}

std::vector<std::string> ModelManager::list_loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(loaded_.size());
    for (const auto& [id, _] : loaded_) {
        ids.push_back(id);
    }
    return ids;
}

std::string ModelManager::default_model_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return first_loaded_;
}

void ModelManager::unload(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    loaded_.erase(model_id);
    if (first_loaded_ == model_id) {
        first_loaded_ = loaded_.empty() ? "" : loaded_.begin()->first;
    }
}

void ModelManager::unload_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    loaded_.clear();
    first_loaded_.clear();
}

void ModelManager::evict_lru_if_needed() {
    // Called with mutex_ held.
    while (static_cast<int>(loaded_.size()) >= max_loaded_) {
        // Find the least recently used model.
        std::string lru_id;
        int64_t lru_time = std::numeric_limits<int64_t>::max();
        for (const auto& [id, lm] : loaded_) {
            if (lm.last_access < lru_time) {
                lru_time = lm.last_access;
                lru_id = id;
            }
        }
        if (lru_id.empty()) break;

        std::cerr << "[ModelManager] Evicting LRU model: " << lru_id << "\n";
        loaded_.erase(lru_id);
        if (first_loaded_ == lru_id) {
            first_loaded_ = loaded_.empty() ? "" : loaded_.begin()->first;
        }
    }
}

} // namespace mlx_lm
