// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++
#pragma once

#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/model_container.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlx_lm {

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// Example:
///
///     auto ctx = load_llm("mlx-community/Qwen3-4B-4bit");
///     auto container = std::make_shared<ModelContainer>(std::move(ctx));
///     ChatSession session(container);
///     std::cout << session.respond("What are two things to see in San Francisco?");
///     std::cout << session.respond("How about a great place to eat?");
///
/// ChatSession manages:
///   - Multi-turn message history
///   - Full re-prefill of templated history into a fresh KV each turn
///     (same strategy as HTTP chat: full messages + new cache per call)
///   - Streaming token-by-token output via callbacks
///
/// Thread safety: ChatSession itself is NOT thread-safe. Each session should be
/// used from a single thread at a time. The underlying ModelContainer handles
/// thread safety for model operations.
class ChatSession {
public:
    /// Callback for streaming generation. Called with each text chunk.
    /// Return true to continue generation, false to stop early.
    using StreamCallback = std::function<bool(const std::string& chunk)>;

    /// Callback for detailed generation events. Called with each GenerateChunk.
    /// Return true to continue generation, false to stop early.
    using DetailCallback = std::function<bool(const GenerateChunk& chunk)>;

    /// Callback invoked when generation completes, with summary info.
    using CompletionCallback = std::function<void(const GenerateInfo& info)>;

    // -- Constructors ---------------------------------------------------------

    /// Initialize a ChatSession with a ModelContainer.
    ///
    /// \param model            Shared pointer to the ModelContainer
    /// \param instructions     Optional system instructions for the session
    /// \param generate_params  Parameters controlling generation
    explicit ChatSession(
        std::shared_ptr<ModelContainer> model,
        std::optional<std::string> instructions = std::nullopt,
        GenerateParameters generate_params = GenerateParameters{});

    /// Re-hydrate from a message transcript (does not restore KV).
    ///
    /// \param model            Shared pointer to the ModelContainer
    /// \param history          Previous chat messages to restore
    /// \param instructions     Optional system instructions for the session
    /// \param generate_params  Parameters controlling generation
    ChatSession(
        std::shared_ptr<ModelContainer> model,
        std::vector<chat::ChatMessage> history,
        std::optional<std::string> instructions = std::nullopt,
        GenerateParameters generate_params = GenerateParameters{});

    // Non-copyable, movable
    ChatSession(const ChatSession&) = delete;
    ChatSession& operator=(const ChatSession&) = delete;
    ChatSession(ChatSession&&) = default;
    ChatSession& operator=(ChatSession&&) = default;

    // -- Respond (blocking, full response) ------------------------------------

    /// Produce a complete response to a user prompt.
    /// Blocks until generation is complete and returns the full response text.
    ///
    /// \param prompt   The user's message
    /// \return         The model's complete response
    std::string respond(const std::string& prompt);

    // -- Streaming responses --------------------------------------------------

    /// Produce a streaming response to a user prompt.
    /// The callback is invoked with each text chunk as it is generated.
    ///
    /// \param prompt       The user's message
    /// \param on_chunk     Called with each text chunk; return false to stop
    /// \param on_complete  Optional callback when generation finishes
    void stream_response(
        const std::string& prompt,
        StreamCallback on_chunk,
        CompletionCallback on_complete = nullptr);

    /// Produce a streaming response with detailed token information.
    /// The callback is invoked with each GenerateChunk (text + token_id).
    ///
    /// \param prompt       The user's message
    /// \param on_detail    Called with each GenerateChunk; return false to stop
    /// \param on_complete  Optional callback when generation finishes
    void stream_details(
        const std::string& prompt,
        DetailCallback on_detail,
        CompletionCallback on_complete = nullptr);

    // -- Session management ---------------------------------------------------

    /// Clear history and residual session state; keep system instructions.
    void clear();

    /// Current message history (pending re-hydrate until first generate).
    const std::vector<chat::ChatMessage>& message_history() const;

    /// Get/set system instructions.
    const std::optional<std::string>& instructions() const { return instructions_; }
    void set_instructions(const std::string& instructions) { instructions_ = instructions; }
    void clear_instructions() { instructions_ = std::nullopt; }

    /// Get/set generation parameters.
    const GenerateParameters& generate_parameters() const { return generate_params_; }
    void set_generate_parameters(const GenerateParameters& params) { generate_params_ = params; }

private:
    /// Conversation phase (KV is not retained across turns).
    enum class CacheState {
        Empty,     // No history
        KVCache,   // Active session; history in messages_ after successful turns
        History,   // pending_history_ not yet folded into messages_
    };

    /// Core generation implementation shared by all respond/stream methods.
    /// \param prompt       The user's message
    /// \param on_detail    Called with each GenerateChunk; return false to stop
    /// \param on_complete  Optional callback when generation finishes
    void generate_impl(
        const std::string& prompt,
        DetailCallback on_detail,
        CompletionCallback on_complete);

    /// Build the messages array for the current turn, including system prompt,
    /// any history messages, and the new user message.
    std::vector<chat::ChatMessage> build_messages(const std::string& user_prompt) const;

    /// Trim n positions from kv_cache_ (unused by default multi-turn path).
    void trim_cache(int n);

    // -- Members --------------------------------------------------------------

    std::shared_ptr<ModelContainer> model_;
    std::optional<std::string> instructions_;
    GenerateParameters generate_params_;

    // Conversation phase; kv_cache_ is ephemeral within a turn only
    CacheState cache_state_ = CacheState::Empty;
    std::vector<KVCache> kv_cache_;

    std::vector<chat::ChatMessage> messages_;
    // Re-hydrate buffer; folded into messages_ on first generate
    std::vector<chat::ChatMessage> pending_history_;
};

} // namespace mlx_lm
