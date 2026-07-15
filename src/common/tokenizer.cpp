// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/tokenizer.h>
#include <fastokens_c.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>

namespace fs = std::filesystem;

namespace mlx_lm {

struct Tokenizer::Impl {
    FastokensHandle handle{nullptr};

    ~Impl() {
        if (handle) {
            fastokens_free(handle);
            handle = nullptr;
        }
    }
};

Tokenizer::~Tokenizer() = default;

std::shared_ptr<Tokenizer> Tokenizer::from_directory(const std::string& model_dir) {
    auto json_path = fs::path(model_dir) / "tokenizer.json";
    if (!fs::exists(json_path)) {
        throw std::runtime_error("tokenizer.json not found in " + model_dir);
    }

    std::ifstream f(json_path);
    if (!f) {
        throw std::runtime_error("Failed to open " + json_path.string());
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    return from_json_blob(ss.str());
}

std::shared_ptr<Tokenizer> Tokenizer::from_json_blob(const std::string& json_blob) {
    auto tokenizer = std::shared_ptr<Tokenizer>(new Tokenizer());
    tokenizer->impl_ = std::make_unique<Impl>();
    tokenizer->impl_->handle = fastokens_new_from_str(json_blob.data(), json_blob.size());
    if (!tokenizer->impl_->handle) {
        throw std::runtime_error("Failed to create fastokens tokenizer from JSON blob");
    }
    return tokenizer;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    FastokensEncodeResult result{};
    // Match previous tokenizers-cpp default: do not inject BOS/EOS via post-processor.
    fastokens_encode(impl_->handle, text.data(), text.size(), /*add_special_tokens=*/0, &result);
    std::vector<int> ids;
    if (result.token_ids && result.len > 0) {
        ids.assign(result.token_ids, result.token_ids + static_cast<std::ptrdiff_t>(result.len));
    }
    fastokens_free_encode_result(&result);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::vector<uint32_t> ids(token_ids.begin(), token_ids.end());
    fastokens_decode(impl_->handle, ids.data(), ids.size(), /*skip_special_tokens=*/0);
    const char* data = nullptr;
    size_t len = 0;
    fastokens_get_decode_str(impl_->handle, &data, &len);
    if (!data || len == 0) {
        return {};
    }
    return std::string(data, len);
}

size_t Tokenizer::vocab_size() const {
    size_t size = 0;
    fastokens_get_vocab_size(impl_->handle, &size);
    return size;
}

std::string Tokenizer::id_to_token(int token_id) const {
    const char* data = nullptr;
    size_t len = 0;
    fastokens_id_to_token(impl_->handle, static_cast<uint32_t>(token_id), &data, &len);
    if (!data || len == 0) {
        return {};
    }
    return std::string(data, len);
}

int Tokenizer::token_to_id(const std::string& token) const {
    int32_t id = -1;
    fastokens_token_to_id(impl_->handle, token.data(), token.size(), &id);
    return static_cast<int>(id);
}

} // namespace mlx_lm
