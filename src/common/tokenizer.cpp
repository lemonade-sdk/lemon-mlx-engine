// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/tokenizer.h>
#include <tokenizers_cpp.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace fs = std::filesystem;

namespace mlx_lm {

struct Tokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;
};

Tokenizer::~Tokenizer() = default;

std::shared_ptr<Tokenizer> Tokenizer::from_directory(const std::string& model_dir) {
    // 1. Try tokenizer.json (HuggingFace fast tokenizer — preferred)
    auto json_path = fs::path(model_dir) / "tokenizer.json";
    if (fs::exists(json_path)) {
        std::ifstream f(json_path);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            try {
                return from_json_blob(ss.str());
            } catch (const std::exception& e) {
                std::cerr << "[tokenizer] tokenizer.json failed: " << e.what()
                          << " — falling back" << std::endl;
            }
        }
    }

    // 2. Try tokenizer.model (SentencePiece — used by Llama, T5, many HF models)
    auto sp_path = fs::path(model_dir) / "tokenizer.model";
    if (fs::exists(sp_path)) {
        std::ifstream f(sp_path, std::ios::binary);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            auto blob = ss.str();
            try {
                auto tokenizer = std::shared_ptr<Tokenizer>(new Tokenizer());
                tokenizer->impl_ = std::make_unique<Impl>();
                tokenizer->impl_->tok = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
                if (tokenizer->impl_->tok) {
                    return tokenizer;
                }
            } catch (const std::exception& e) {
                std::cerr << "[tokenizer] tokenizer.model (SentencePiece) failed: "
                          << e.what() << std::endl;
            }
        }
    }

    // 3. Try vocab.json + merges.txt (GPT-style BPE)
    auto vocab_path = fs::path(model_dir) / "vocab.json";
    auto merges_path = fs::path(model_dir) / "merges.txt";
    if (fs::exists(vocab_path) && fs::exists(merges_path)) {
        try {
            auto tokenizer = std::shared_ptr<Tokenizer>(new Tokenizer());
            tokenizer->impl_ = std::make_unique<Impl>();
            std::ifstream vf(vocab_path), mf(merges_path);
            std::ostringstream vs, ms;
            vs << vf.rdbuf();
            ms << mf.rdbuf();
            tokenizer->impl_->tok = tokenizers::Tokenizer::FromBlobByteLevelBPE(
                vs.str(), ms.str());
            if (tokenizer->impl_->tok) {
                return tokenizer;
            }
        } catch (const std::exception& e) {
            std::cerr << "[tokenizer] vocab.json+merges.txt BPE failed: "
                      << e.what() << std::endl;
        }
    }

    throw std::runtime_error(
        "No usable tokenizer found in " + model_dir +
        " (tried tokenizer.json, tokenizer.model, vocab.json+merges.txt)");
}

std::shared_ptr<Tokenizer> Tokenizer::from_json_blob(const std::string& json_blob) {
    auto tokenizer = std::shared_ptr<Tokenizer>(new Tokenizer());
    tokenizer->impl_ = std::make_unique<Impl>();
    tokenizer->impl_->tok = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    if (!tokenizer->impl_->tok) {
        throw std::runtime_error("Failed to create tokenizer from JSON blob");
    }
    return tokenizer;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    auto ids = impl_->tok->Encode(text);
    return std::vector<int>(ids.begin(), ids.end());
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::vector<int32_t> ids(token_ids.begin(), token_ids.end());
    return impl_->tok->Decode(ids);
}

size_t Tokenizer::vocab_size() const {
    return impl_->tok->GetVocabSize();
}

std::string Tokenizer::id_to_token(int token_id) const {
    return impl_->tok->IdToToken(static_cast<int32_t>(token_id));
}

int Tokenizer::token_to_id(const std::string& token) const {
    return static_cast<int>(impl_->tok->TokenToId(token));
}

} // namespace mlx_lm
