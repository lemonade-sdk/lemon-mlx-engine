// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/hub_api.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

// --- curl write callback ---
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* stream = static_cast<std::string*>(userp);
    stream->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

static size_t file_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

// --- progress callback ---
struct ProgressData {
    ProgressCallback callback;
};

static int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                              curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    auto* data = static_cast<ProgressData*>(clientp);
    if (data->callback && dltotal > 0) {
        data->callback(static_cast<size_t>(dlnow), static_cast<size_t>(dltotal));
    }
    return 0;
}

// --- HubApi ---

void HubApi::set_token(const std::string& token) {
    token_ = token;
}

std::string HubApi::cache_dir() const {
    if (!cache_dir_.empty()) return cache_dir_;

    // Priority: HF_HUB_CACHE > HF_HOME/hub > ~/.cache/huggingface/hub
    // Matches llama.cpp and huggingface-hub conventions.
    if (const char* hub_cache = std::getenv("HF_HUB_CACHE")) {
        return std::string(hub_cache);
    }
    if (const char* hf_home = std::getenv("HF_HOME")) {
        return std::string(hf_home) + "/hub";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::string(home) + "/.cache/huggingface/hub";
    }
#ifdef _WIN32
    if (const char* userprofile = std::getenv("USERPROFILE")) {
        return std::string(userprofile) + "/.cache/huggingface/hub";
    }
#endif
    return ".cache/huggingface/hub";
}

void HubApi::set_cache_dir(const std::string& dir) {
    cache_dir_ = dir;
}

// Convert repo_id ("org/repo") to HF cache dir name ("models--org--repo").
static std::string repo_id_to_cache_key(const std::string& repo_id) {
    std::string key = "models--";
    for (char c : repo_id) {
        if (c == '/') key += "--";
        else key += c;
    }
    return key;
}

// Given a models--X directory, find the best snapshot path.
// Standard HF: refs/main → commit hash → snapshots/{hash}/
// Our legacy:  snapshots/main/ directly
static std::string resolve_snapshot_dir(const std::string& model_dir,
                                         const std::string& revision) {
    // Try standard HF layout: read refs/{revision} to get commit hash
    auto refs_path = model_dir + "/refs/" + revision;
    if (fs::exists(refs_path) && fs::is_regular_file(refs_path)) {
        std::ifstream f(refs_path);
        std::string hash;
        std::getline(f, hash);
        // Trim whitespace
        while (!hash.empty() && (hash.back() == '\n' || hash.back() == '\r' || hash.back() == ' '))
            hash.pop_back();
        if (!hash.empty()) {
            auto snap = model_dir + "/snapshots/" + hash;
            if (fs::exists(snap)) return snap;
        }
    }

    // Fallback: snapshots/{revision}/ directly (our legacy format)
    auto direct = model_dir + "/snapshots/" + revision;
    if (fs::exists(direct)) return direct;

    // Last resort: find any snapshot directory
    auto snap_dir = model_dir + "/snapshots";
    if (fs::exists(snap_dir) && fs::is_directory(snap_dir)) {
        for (const auto& entry : fs::directory_iterator(snap_dir)) {
            if (entry.is_directory()) return entry.path().string();
        }
    }

    // Return the standard path even if it doesn't exist yet (for downloads)
    return direct;
}

std::string HubApi::resolve_cache_path(const std::string& repo_id,
                                        const std::string& revision) const {
    auto base = cache_dir();
    auto cache_key = repo_id_to_cache_key(repo_id);
    auto model_dir = base + "/" + cache_key;

    // If the standard-format directory exists, resolve its snapshot
    if (fs::exists(model_dir)) {
        return resolve_snapshot_dir(model_dir, revision);
    }

    // Check legacy format (models--org-repo with single dash)
    std::string legacy_key = "models--";
    for (char c : repo_id) {
        legacy_key += (c == '/') ? '-' : c;
    }
    auto legacy_dir = base + "/" + legacy_key;
    if (fs::exists(legacy_dir)) {
        return resolve_snapshot_dir(legacy_dir, revision);
    }

    // Neither exists — return standard format path for new downloads
    return model_dir + "/snapshots/" + revision;
}

bool HubApi::is_cached(const std::string& repo_id, const std::string& revision) const {
    auto path = resolve_cache_path(repo_id, revision);
    if (!fs::exists(path) || !fs::exists(path + "/config.json"))
        return false;

    // Must have safetensors files (MLX format only, not GGUF)
    for (const auto& entry : fs::directory_iterator(path)) {
        auto ext = entry.path().extension().string();
        if (ext == ".safetensors") return true;
    }
    // Also accept sharded format (model.safetensors.index.json)
    return fs::exists(path + "/model.safetensors.index.json");
}

std::string HubApi::model_directory(const std::string& repo_id,
                                     const std::string& revision) const {
    return resolve_cache_path(repo_id, revision);
}

std::string HubApi::http_get(const std::string& url,
                              const std::string& output_path,
                              ProgressCallback progress) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("Failed to initialize curl");

    std::string response_body;
    std::ofstream output_file;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "mlx-cpp-lm/0.1");

    if (!token_.empty()) {
        std::string auth = "Authorization: Bearer " + token_;
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    if (!output_path.empty()) {
        fs::create_directories(fs::path(output_path).parent_path());
        output_file.open(output_path, std::ios::binary);
        if (!output_file) {
            curl_easy_cleanup(curl);
            throw std::runtime_error("Failed to open output file: " + output_path);
        }
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, file_write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_file);
    } else {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    }

    ProgressData prog_data{progress};
    if (progress) {
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &prog_data);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (output_file.is_open()) output_file.close();

    if (res != CURLE_OK) {
        throw std::runtime_error("HTTP request failed: " + std::string(curl_easy_strerror(res)));
    }
    if (http_code >= 400) {
        throw std::runtime_error("HTTP error " + std::to_string(http_code) + " for " + url);
    }

    return output_path.empty() ? response_body : output_path;
}

std::string HubApi::download_file(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& revision,
    ProgressCallback progress)
{
    auto cache_path = resolve_cache_path(repo_id, revision);
    auto file_path = cache_path + "/" + filename;

    if (fs::exists(file_path)) return file_path;

    std::string url = "https://huggingface.co/" + repo_id + "/resolve/" + revision + "/" + filename;
    return http_get(url, file_path, progress);
}

std::string HubApi::snapshot_download(
    const std::string& repo_id,
    const std::string& revision,
    const std::vector<std::string>& allow_patterns,
    ProgressCallback progress)
{
    auto cache_path = resolve_cache_path(repo_id, revision);

    // Check if already cached
    if (fs::exists(cache_path + "/config.json")) {
        return cache_path;
    }

    // Fetch file list from the HF API
    std::string api_url = "https://huggingface.co/api/models/" + repo_id +
                          "/revision/" + revision;

    std::vector<std::string> files_to_download;
    bool api_ok = false;
    try {
        auto api_response = http_get(api_url);
        auto api_json = nlohmann::json::parse(api_response);
        if (api_json.contains("siblings") && api_json["siblings"].is_array()) {
            for (const auto& sib : api_json["siblings"]) {
                if (sib.contains("rfilename")) {
                    files_to_download.push_back(sib["rfilename"].get<std::string>());
                }
            }
            api_ok = !files_to_download.empty();
        }
    } catch (...) {
        // API call failed — fall back to hardcoded list below
    }

    // Extensions that are useful for MLX model loading.
    // SKIP large native formats we can't load without conversion.
    auto should_download = [](const std::string& fname) -> bool {
        auto ends_with = [](const std::string& s, const std::string& suf) {
            return s.size() >= suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf) == 0;
        };
        // Skip formats we cannot load directly
        for (const auto& skip : {".bin", ".pt", ".h5", ".msgpack", ".safetensors.index.json.bak"}) {
            if (ends_with(fname, skip)) return false;
        }
        // Download these useful formats
        for (const auto& good : {".json", ".safetensors", ".model", ".txt", ".jinja", ".token"}) {
            if (ends_with(fname, good)) return true;
        }
        return false;
    };

    // Filter by allow_patterns if provided
    auto matches_allow = [&](const std::string& fname) -> bool {
        if (allow_patterns.empty()) return true;
        for (const auto& pat : allow_patterns) {
            if (fname == pat) return true;
            // Simple glob: pat ends with '*' → prefix match
            if (!pat.empty() && pat.back() == '*' &&
                fname.size() >= pat.size()-1 &&
                fname.compare(0, pat.size()-1, pat, 0, pat.size()-1) == 0) {
                return true;
            }
        }
        return false;
    };

    if (api_ok) {
        // Universal: download every relevant file the repo actually has
        for (const auto& f : files_to_download) {
            if (!should_download(f) || !matches_allow(f)) continue;
            bool is_large = f.find(".safetensors") != std::string::npos;
            try {
                download_file(repo_id, f, revision, is_large ? progress : nullptr);
            } catch (...) {
                // Skip files that fail (optional or temporarily unavailable)
            }
        }
    } else {
        // Fallback: hardcoded list (preserves old behavior on API failure)
        std::vector<std::string> default_files = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        };
        for (const auto& f : default_files) {
            try { download_file(repo_id, f, revision, nullptr); } catch (...) {}
        }
        // Download safetensors (single or sharded)
        std::string last_error;
        try {
            download_file(repo_id, "model.safetensors", revision, progress);
        } catch (const std::exception& e) {
            last_error = e.what();
            try {
                download_file(repo_id, "model.safetensors.index.json", revision, nullptr);
                auto index_path = cache_path + "/model.safetensors.index.json";
                if (fs::exists(index_path)) {
                    std::ifstream index_file(index_path);
                    nlohmann::json index_json;
                    index_file >> index_json;
                    if (index_json.contains("weight_map")) {
                        std::set<std::string> shard_files;
                        for (auto& [key, val] : index_json["weight_map"].items()) {
                            shard_files.insert(val.get<std::string>());
                        }
                        for (const auto& shard : shard_files) {
                            download_file(repo_id, shard, revision, progress);
                        }
                    }
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Could not find model weights for " + repo_id +
                                         " (single-file error: " + last_error +
                                         ", sharded error: " + e.what() + ")");
            }
        }
    }

    return cache_path;
}

// Convert a cache directory name back to a HF repo_id.
// "models--org--repo-name" → "org/repo-name"
// "models--org-repo-name"  → ambiguous (legacy), use first dash as separator
static std::string cache_key_to_repo_id(const std::string& dir_name) {
    // Strip "models--" prefix
    if (dir_name.size() < 9 || dir_name.substr(0, 8) != "models--") return "";
    auto rest = dir_name.substr(8);

    // Standard format: org--repo (double dash)
    auto pos = rest.find("--");
    if (pos != std::string::npos) {
        return rest.substr(0, pos) + "/" + rest.substr(pos + 2);
    }

    // Legacy format: org-repo (single dash) — use first dash as separator
    pos = rest.find('-');
    if (pos != std::string::npos) {
        return rest.substr(0, pos) + "/" + rest.substr(pos + 1);
    }

    // No separator — bare model name
    return rest;
}

// Check if a snapshot directory contains MLX model files (safetensors).
static bool is_mlx_model_dir(const std::string& snap_path) {
    if (!fs::exists(snap_path + "/config.json")) return false;

    for (const auto& entry : fs::directory_iterator(snap_path)) {
        if (entry.path().extension() == ".safetensors") return true;
    }
    return fs::exists(snap_path + "/model.safetensors.index.json");
}

std::vector<HubApi::CachedModel> HubApi::discover_cached_models() const {
    std::vector<CachedModel> result;
    auto base = cache_dir();

    if (!fs::exists(base) || !fs::is_directory(base)) return result;

    for (const auto& entry : fs::directory_iterator(base)) {
        if (!entry.is_directory()) continue;

        auto dir_name = entry.path().filename().string();
        if (dir_name.substr(0, 8) != "models--") continue;

        auto repo_id = cache_key_to_repo_id(dir_name);
        if (repo_id.empty()) continue;

        // Resolve the best snapshot directory
        auto snap_path = resolve_snapshot_dir(entry.path().string(), "main");

        // Only include MLX models (safetensors + config.json, not GGUF)
        if (is_mlx_model_dir(snap_path)) {
            result.push_back({repo_id, snap_path});
        }
    }

    return result;
}

HubApi& HubApi::shared() {
    static HubApi instance;
    // Auto-detect token from environment or file
    if (instance.token_.empty()) {
        if (const char* token = std::getenv("HF_TOKEN")) {
            instance.token_ = token;
        } else {
            // Try reading from ~/.huggingface/token
            std::string token_path;
            if (const char* home = std::getenv("HOME")) {
                token_path = std::string(home) + "/.huggingface/token";
            }
#ifdef _WIN32
            if (token_path.empty()) {
                if (const char* userprofile = std::getenv("USERPROFILE")) {
                    token_path = std::string(userprofile) + "/.huggingface/token";
                }
            }
#endif
            if (!token_path.empty() && fs::exists(token_path)) {
                std::ifstream f(token_path);
                std::getline(f, instance.token_);
            }
        }
    }
    return instance;
}

} // namespace mlx_lm
