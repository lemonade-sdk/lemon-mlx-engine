// Copyright © 2025 — Ported to C++
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlx_lm {

// Tokenizer wraps crusoecloud/fastokens (high-performance BPE).
// Loads from tokenizer.json in a model directory.
// https://github.com/crusoecloud/fastokens
class Tokenizer {
public:
    ~Tokenizer();

    // Load tokenizer from a model directory containing tokenizer.json.
    static std::shared_ptr<Tokenizer> from_directory(const std::string& model_dir);

    // Load tokenizer from a raw JSON blob string.
    static std::shared_ptr<Tokenizer> from_json_blob(const std::string& json_blob);

    // Encode text to token IDs.
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs to text.
    std::string decode(const std::vector<int>& token_ids) const;

    // Vocabulary size.
    size_t vocab_size() const;

    // Convert a single token ID to its string representation.
    std::string id_to_token(int token_id) const;

    // Convert a token string to its ID.
    int token_to_id(const std::string& token) const;

private:
    Tokenizer() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mlx_lm
