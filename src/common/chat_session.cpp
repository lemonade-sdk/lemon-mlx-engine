// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++

#include <mlx-lm/common/chat_session.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    last_templated_tokens_.clear();
    messages_.clear();
    pending_history_.clear();
#if defined(MLX_BUILD_ROCM)
    mx::synchronize();
#endif
    mx::clear_cache();
}

const std::vector<chat::ChatMessage>& ChatSession::message_history() const {
    // Prefer folded messages_; else expose pending re-hydrate before first generate.
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

    // messages_ after fold; pending_history_ on first re-hydrate turn only
    if (!messages_.empty()) {
        messages.insert(messages.end(), messages_.begin(), messages_.end());
    } else if (cache_state_ == CacheState::History && !pending_history_.empty()) {
        messages.insert(messages.end(),
                        pending_history_.begin(),
                        pending_history_.end());
    }

    messages.push_back(chat::ChatMessage::user(user_prompt));

    return messages;
}

// -- Private: trim cache ------------------------------------------------------

void ChatSession::trim_cache(int n) {
    if (kv_cache_.empty() || n <= 0) return;

    for (auto& cache : kv_cache_) {
        if (cache.is_trimmable()) {
            cache.trim(n);
        }
    }
}

size_t ChatSession::token_lcp(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    for (; i < n && a[i] == b[i]; ++i) {
    }
    return i;
}

bool ChatSession::residual_kv_rollback_ok(const std::vector<KVCache>& caches) {
    if (caches.empty()) {
        return false;
    }
    for (const auto& c : caches) {
        // Mamba / non-trimmable layers cannot roll back via set_position.
        if (!c.is_trimmable()) {
            return false;
        }
    }
    return true;
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

    // Snapshot for this turn (includes pending re-hydrate if not yet folded).
    auto turn_messages = build_messages(prompt);

    model_->perform([&](ModelContext& ctx) {
        // Fold re-hydrate into messages_ for later turns (this turn already
        // templated from turn_messages). User/assistant append after success.
        if (cache_state_ == CacheState::History && !pending_history_.empty()) {
            messages_.insert(messages_.end(),
                             pending_history_.begin(),
                             pending_history_.end());
            pending_history_.clear();
        }
        // Default: fresh KV every turn — residual reuse + full re-template
        // double-prefills. Optional MLX_CHAT_RESIDUAL=1: keep last KV and
        // append-prefill only the token suffix after an exact LCP with
        // last_templated_tokens_ (never full-template onto residual without LCP).
        const bool residual_opt = [] {
            const char* e = std::getenv("MLX_CHAT_RESIDUAL");
            return e && e[0] == '1' && e[1] == '\0';
        }();
        const bool turn_log = residual_opt || [] {
            const char* e = std::getenv("MLX_CHAT_TURN_LOG");
            return e && e[0] == '1' && e[1] == '\0';
        }();

        cache_state_ = CacheState::KVCache;

        DefaultMessageGenerator msg_gen;
        auto raw_messages = msg_gen.generate(turn_messages);

        if (!ctx.apply_chat_template_fn) {
            throw std::runtime_error(
                "ChatSession: apply_chat_template_fn is not set on ModelContext");
        }
        auto tokens = ctx.apply_chat_template_fn(raw_messages, /*tools=*/nullptr);

        if (tokens.empty()) {
            throw std::runtime_error("ChatSession: chat template produced no tokens");
        }

        const int prompt_token_count = static_cast<int>(tokens.size());
        const char* residual_note = "full";
        std::vector<int> prefill_tokens = tokens;

        if (residual_opt && residual_kv_rollback_ok(kv_cache_) &&
            !last_templated_tokens_.empty()) {
            const size_t prefix = token_lcp(last_templated_tokens_, tokens);
            const bool have_suffix = prefix < tokens.size();
            const bool prefix_ok = prefix > 0 && have_suffix;
            if (prefix_ok) {
                for (auto& c : kv_cache_) {
                    c.set_position(prefix);
                }
                prefill_tokens.assign(tokens.begin() + static_cast<std::ptrdiff_t>(prefix),
                                      tokens.end());
                residual_note = "lcp-suffix";
            } else {
                kv_cache_.clear();
                last_templated_tokens_.clear();
                residual_note = "lcp-fallback-full";
            }
        }

        if (kv_cache_.empty()) {
            kv_cache_ = ctx.new_cache_fn(generate_params_);
            if (std::strcmp(residual_note, "lcp-suffix") != 0) {
                residual_note = residual_opt ? residual_note : "full";
            }
        }

        auto token_array = mx::array(
            prefill_tokens.data(),
            {static_cast<int>(prefill_tokens.size())},
            mx::int32);

        LMInput lm_input(token_array);

        // External-cache + params ctor (fresh or residual; MTP still via params).
        TokenIterator iter(
            ctx, lm_input, std::move(kv_cache_), generate_params_);

        NaiveStreamingDetokenizer detokenizer;
        auto decode = [&ctx](const std::vector<int>& toks) -> std::string {
            return ctx.decode_fn(toks);
        };

        std::string assistant_response;
        int generated_count = 0;

        while (auto maybe_token = iter.next()) {
            int token_id = *maybe_token;

            if (ctx.eos_token_ids.has_value()) {
                auto& eos = ctx.eos_token_ids.value();
                if (std::find(eos.begin(), eos.end(), token_id) != eos.end()) {
                    break;
                }
            }

            generated_count++;

            if (generated_count % 256 == 0) {
                mx::clear_cache();
            }

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

        mx::synchronize();

        messages_.push_back(chat::ChatMessage::user(prompt));
        messages_.push_back(chat::ChatMessage::assistant(assistant_response));

        kv_cache_ = iter.take_cache();
        if (residual_opt) {
            // Keep residual KV + exact last full-template tokens for next LCP.
            last_templated_tokens_ = tokens;
        } else {
            // I3-safe default: drop cache; next turn re-prefills from messages_.
            kv_cache_.clear();
            last_templated_tokens_.clear();
        }

        const auto info_full = iter.completion_info(
            static_cast<int>(prefill_tokens.size()));
        if (turn_log) {
            std::cerr << "[chat-session] turn hist_msgs=" << turn_messages.size()
                      << " template_tok=" << prompt_token_count
                      << " prefill_tok=" << prefill_tokens.size()
                      << " prefill_s=" << info_full.prompt_time
                      << " gen_tok=" << info_full.generation_token_count
                      << " residual=" << residual_note << "\n";
        }

        if (on_complete) {
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
