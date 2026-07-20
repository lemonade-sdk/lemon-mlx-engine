// Copyright © 2025 — Ported to C++
#pragma once

#include <nlohmann/json.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlx_lm {

using Message = std::unordered_map<std::string, std::string>;

/// Wraps a Jinja2 chat template from tokenizer_config.json.
/// Uses minja (Google's Jinja2 engine for HF chat templates) internally.
class ChatTemplate {
public:
    /// Construct from a Jinja2 template string and tokenizer config JSON.
    /// Extracts bos_token, eos_token, and other special tokens from config.
    ChatTemplate(const std::string& template_str,
                 const nlohmann::json& tokenizer_config);

    ~ChatTemplate();

    // Movable but not copyable (pImpl)
    ChatTemplate(ChatTemplate&&) noexcept;
    ChatTemplate& operator=(ChatTemplate&&) noexcept;
    ChatTemplate(const ChatTemplate&) = delete;
    ChatTemplate& operator=(const ChatTemplate&) = delete;

    /// Render messages to a formatted prompt string.
    /// add_generation_prompt=true appends the assistant turn opener.
    /// tools: optional OpenAI tools array (nullptr or null json = no tools).
    /// When non-null array, sets minja inputs.tools for native/polyfill tool schemas.
    std::string apply(
        const std::vector<Message>& messages,
        bool add_generation_prompt = true,
        const nlohmann::json& extra_context = {},
        const nlohmann::json* tools = nullptr) const;

    /// Get the raw template string.
    const std::string& template_string() const;

    /// Get the EOS token string (from tokenizer_config).
    const std::string& eos_token() const;

    /// Get the BOS token string (from tokenizer_config).
    const std::string& bos_token() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Load a ChatTemplate from a model directory (reads tokenizer_config.json).
/// Returns nullopt if no chat_template field is found.
std::optional<ChatTemplate> load_chat_template(const std::string& model_directory);

} // namespace mlx_lm
