// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++

#include <mlx-lm/common/chat_session.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace mlx_lm {

namespace mx = mlx::core;

// -- Constructors -------------------------------------------------------------

ChatSession::ChatSession(
    std::shared_ptr<ModelContainer> model,
    std::optional<std::string> instructions,
    GenerateParameters generate_params)
    : model_(std::move(model)),
      instructions_(std::move(instructions)),
      generate_params_(std::move(generate_params)),
      cache_state_(CacheState::Empty)
{}

ChatSession::ChatSession(
    std::shared_ptr<ModelContainer> model,
    std::vector<chat::ChatMessage> history,
    std::optional<std::string> instructions,
    GenerateParameters generate_params)
    : model_(std::move(model)),
      instructions_(std::move(instructions)),
      generate_params_(std::move(generate_params)),
      cache_state_(CacheState::History),
      pending_history_(std::move(history))
{}

// -- Respond (blocking) -------------------------------------------------------

std::string ChatSession::respond(const std::string& prompt) {
    std::string output;
    stream_response(
        prompt,
        [&output](const std::string& chunk) -> bool {
            output += chunk;
            return true;
        });
    return output;
}

// -- Streaming responses ------------------------------------------------------

void ChatSession::stream_response(
    const std::string& prompt,
    StreamCallback on_chunk,
    CompletionCallback on_complete)
{
    generate_impl(
        prompt,
        [&on_chunk](const GenerateChunk& chunk) -> bool {
            return on_chunk(chunk.text);
        },
        std::move(on_complete));
}

void ChatSession::stream_details(
    const std::string& prompt,
    DetailCallback on_detail,
    CompletionCallback on_complete)
{
    generate_impl(prompt, std::move(on_detail), std::move(on_complete));
}

// -- Session management -------------------------------------------------------

void ChatSession::clear() {
    cache_state_ = CacheState::Empty;
    kv_cache_.clear();
    messages_.clear();
    pending_history_.clear();
}

const std::vector<chat::ChatMessage>& ChatSession::message_history() const {
    // Prefer messages_ (post-fold). If still holding re-hydration pending only,
    // expose pending so callers see restored history before first generate.
    if (!messages_.empty() || pending_history_.empty()) {
        return messages_;
    }
    return pending_history_;
}

// -- Private: build messages --------------------------------------------------

std::vector<chat::ChatMessage> ChatSession::build_messages(
    const std::string& user_prompt) const
{
    std::vector<chat::ChatMessage> messages;

    // Add system instructions if present
    if (instructions_.has_value()) {
        messages.push_back(chat::ChatMessage::system(instructions_.value()));
    }

    // Full conversation so far (messages_ is source of truth after any
    // re-hydration has been folded in — see generate_impl).
    if (!messages_.empty()) {
        messages.insert(messages.end(), messages_.begin(), messages_.end());
    } else if (cache_state_ == CacheState::History && !pending_history_.empty()) {
        // First turn after history ctor, before pending is folded into messages_.
        messages.insert(messages.end(),
                        pending_history_.begin(),
                        pending_history_.end());
    }

    // Add the new user message
    messages.push_back(chat::ChatMessage::user(user_prompt));

    return messages;
}

// -- Private: trim cache ------------------------------------------------------

void ChatSession::trim_cache(int n) {
    if (kv_cache_.empty() || n <= 0) return;

    // Trim n positions from the end of each trimmable cache layer.
    // Available for advanced use cases; not used by the default multi-turn
    // path (which allocates a fresh cache each turn).
    for (auto& cache : kv_cache_) {
        if (cache.is_trimmable()) {
            cache.trim(n);
        }
    }
}

// -- Private: core generation -------------------------------------------------

void ChatSession::generate_impl(
    const std::string& prompt,
    DetailCallback on_detail,
    CompletionCallback on_complete)
{
    if (!model_) {
        throw std::runtime_error("ChatSession: model is null");
    }

    // Upgrade GPU wired memory for the duration of generation.
    WiredLimitGuard wired_guard;

    // Build messages for this turn, including system prompt + history + user msg
    auto turn_messages = build_messages(prompt);

    // Perform generation with model lock
    model_->perform([&](ModelContext& ctx) {
        // Correctness-first multi-turn: always prefill the full templated
        // history into a FRESH KV cache. Reusing a non-empty cache while
        // re-templating (or templating only the new user turn with BOS/framing)
        // double-prefills and corrupts decode on turn 2+.
        //
        // Trade-off: slower than true delta-prefill KV reuse, but matches the
        // HTTP path (fresh cache + full messages) and is safe in eager mode.
        // True delta-prefill can land later once prefix-token accounting is solid.
        //
        // Re-hydration: fold pending history into messages_ BEFORE generate so
        // messages_ remains the source of truth for later turns (and on throw).
        if (cache_state_ == CacheState::History && !pending_history_.empty()) {
            messages_.insert(messages_.end(),
                             pending_history_.begin(),
                             pending_history_.end());
            pending_history_.clear();
        }
        kv_cache_ = ctx.new_cache_fn(generate_params_);
        cache_state_ = CacheState::KVCache;

        // Convert ChatMessages to raw Messages for the chat template
        DefaultMessageGenerator msg_gen;
        auto raw_messages = msg_gen.generate(turn_messages);

        // Apply chat template to get token IDs
        if (!ctx.apply_chat_template_fn) {
            throw std::runtime_error(
                "ChatSession: apply_chat_template_fn is not set on ModelContext");
        }
        auto tokens = ctx.apply_chat_template_fn(raw_messages, /*tools=*/nullptr);

        if (tokens.empty()) {
            throw std::runtime_error("ChatSession: chat template produced no tokens");
        }


        // Create token array
        auto token_array = mx::array(
            tokens.data(),
            {static_cast<int>(tokens.size())},
            mx::int32);

        int prompt_token_count = static_cast<int>(tokens.size());

        // Build LMInput and use TokenIterator for generation.
        // This gives us async pipelining ("compute next, return current")
        // and all the optimizations from the shared generate path.
        LMInput lm_input(token_array);

        // External-cache + params constructor: pass the fresh cache we just
        // allocated (MTP may still be requested via generate_params_, but
        // multi-turn correctness no longer depends on residual KV reuse).
        TokenIterator iter(
            ctx, lm_input, std::move(kv_cache_), generate_params_);

        // Set up streaming detokenizer
        NaiveStreamingDetokenizer detokenizer;
        auto decode = [&ctx](const std::vector<int>& toks) -> std::string {
            return ctx.decode_fn(toks);
        };

        // Accumulate the full assistant response for message history
        std::string assistant_response;
        int generated_count = 0;

        while (auto maybe_token = iter.next()) {
            int token_id = *maybe_token;

            // Check for EOS
            if (ctx.eos_token_ids.has_value()) {
                auto& eos = ctx.eos_token_ids.value();
                if (std::find(eos.begin(), eos.end(), token_id) != eos.end()) {
                    break;
                }
            }

            generated_count++;

            // Periodically clear the memory cache.
            if (generated_count % 256 == 0) {
                mx::clear_cache();
            }

            // Decode token through streaming detokenizer
            detokenizer.append(token_id);
            auto text = detokenizer.next(decode);
            if (text.has_value() && !text->empty()) {
                assistant_response += text.value();

                GenerateChunk chunk{text.value(), token_id};
                if (on_detail && !on_detail(chunk)) {
                    break;
                }
            }
        }

        // Synchronize to ensure all pending GPU work completes.
        mx::synchronize();

        // Add user and assistant messages to history
        messages_.push_back(chat::ChatMessage::user(prompt));
        messages_.push_back(chat::ChatMessage::assistant(assistant_response));

        // Release iterator-owned cache and drop it. Next turn allocates a
        // fresh cache and re-prefills full history (messages_ is source of truth).
        kv_cache_ = iter.take_cache();
        kv_cache_.clear();

        // Invoke completion callback with stats
        if (on_complete) {
            auto info_full = iter.completion_info(prompt_token_count);
            GenerateInfo info;
            info.prompt_tokens = info_full.prompt_token_count;
            info.generated_tokens = info_full.generation_token_count;
            info.prompt_time_s = info_full.prompt_time;
            info.generation_time_s = info_full.generation_time;
            on_complete(info);
        }
    });
}

} // namespace mlx_lm
