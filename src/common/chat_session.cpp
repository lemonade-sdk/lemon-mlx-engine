// Copyright (C) 2024-2025 Apple Inc. -- Ported to C++

#include <mlx-lm/common/chat_session.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
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
    last_sequence_tokens_.clear();
    last_body_tokens_.clear();
    last_mamba_snapshots_.clear();
    last_residual_instructions_.reset();
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
        if (c.is_compound()) {
            // Compound set_position is a no-op (including Rotating sub-cache).
            return false;
        }
        if (c.as_mamba() != nullptr) {
            // Pure Mamba/GDN: restore via snapshot.
            continue;
        }
        if (!c.is_trimmable()) {
            return false;
        }
    }
    return true;
}

bool ChatSession::restore_residual_to_template(
    std::vector<KVCache>& caches, size_t prefix)
{
    if (caches.size() != last_mamba_snapshots_.size()) {
        return false;
    }
    for (size_t i = 0; i < caches.size(); ++i) {
        if (caches[i].is_compound()) {
            return false;
        }
        if (auto* m = caches[i].as_mamba()) {
            if (!last_mamba_snapshots_[i].has_value()) {
                return false;
            }
            m->restore(last_mamba_snapshots_[i].value());
        } else {
            caches[i].set_position(prefix);
            const size_t pos = caches[i].get_position();
            if (pos != 0 && pos != prefix) {
                return false;
            }
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
        // Production: reuse residual KV. Never consume a non-prefix full
        // template on leftover cache (I3). MLX_CHAT_RESIDUAL=0 forces the
        // fail-closed full re-prefill path.
        const bool residual_opt = [](bool param) {
            const char* e = std::getenv("MLX_CHAT_RESIDUAL");
            if (!e || e[0] == '\0') {
                return param;
            }
            // Treat leading 0/1; ignore CR/whitespace so the kill switch
            // cannot silently stay ON.
            unsigned char c = static_cast<unsigned char>(e[0]);
            if (c == '0') {
                return false;
            }
            if (c == '1') {
                return true;
            }
            return param;
        }(generate_params_.chat_residual);
        const bool turn_log = [] {
            const char* e = std::getenv("MLX_CHAT_TURN_LOG");
            return e && e[0] == '1';
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

        std::vector<int> body;
        if (ctx.apply_chat_template_body_fn) {
            body = ctx.apply_chat_template_body_fn(raw_messages, /*tools=*/nullptr);
            if (!body.empty() &&
                (token_lcp(body, tokens) != body.size() || body.size() >= tokens.size())) {
                body.clear();
            }
        }

        const int prompt_token_count = static_cast<int>(tokens.size());
        const char* residual_note = "full";
        std::vector<int> prefill_tokens = tokens;
        size_t body_prefill_tok = 0;
        double body_prefill_s = 0.0;
        bool snap_at_body = false;
        std::vector<std::optional<MambaCache::Snapshot>> turn_mamba_snaps;

        // Fail closed: leftover residual KV must never see a full template.
        // seq-append / lcp-suffix / body-suffix (Qwen thinking: restore to
        // add_generation_prompt=false, then prefill rewritten assistant+user).
        auto clear_residual_state = [&]() {
            kv_cache_.clear();
            last_templated_tokens_.clear();
            last_sequence_tokens_.clear();
            last_body_tokens_.clear();
            last_mamba_snapshots_.clear();
            last_residual_instructions_.reset();
        };
        const bool rollback_ok = residual_kv_rollback_ok(kv_cache_);
        if (!residual_opt && !kv_cache_.empty()) {
            clear_residual_state();
            residual_note = "full";
        } else if (residual_opt && !last_templated_tokens_.empty() && !rollback_ok) {
            clear_residual_state();
            residual_note = "nontrim-fallback-full";
        } else if (residual_opt && rollback_ok && !last_templated_tokens_.empty()) {
            const bool system_same = (instructions_ == last_residual_instructions_);
            bool used = false;
            if (system_same && !last_sequence_tokens_.empty()) {
                const size_t seq_prefix = token_lcp(last_sequence_tokens_, tokens);
                if (seq_prefix == last_sequence_tokens_.size() &&
                    seq_prefix < tokens.size()) {
                    prefill_tokens.assign(
                        tokens.begin() + static_cast<std::ptrdiff_t>(seq_prefix),
                        tokens.end());
                    residual_note = "seq-append";
                    used = true;
                }
            }
            // lcp-suffix restores the last snapshot. That snapshot is at
            // last_body when body-suffix is in use — do not roll attention
            // to |T1| while Mamba is at |body|.
            if (!used && last_body_tokens_.empty()) {
                const size_t prefix = token_lcp(last_templated_tokens_, tokens);
                const bool prefix_ok = system_same &&
                    prefix == last_templated_tokens_.size() && prefix < tokens.size();
                if (prefix_ok && restore_residual_to_template(kv_cache_, prefix)) {
                    prefill_tokens.assign(
                        tokens.begin() + static_cast<std::ptrdiff_t>(prefix),
                        tokens.end());
                    residual_note = "lcp-suffix";
                    used = true;
                }
            }
            if (!used && system_same && !body.empty() && !last_body_tokens_.empty()) {
                const size_t bp = token_lcp(last_body_tokens_, body);
                if (bp == last_body_tokens_.size() &&
                    restore_residual_to_template(kv_cache_, bp)) {
                    if (bp < body.size()) {
                        std::vector<int> body_delta(
                            body.begin() + static_cast<std::ptrdiff_t>(bp),
                            body.end());
                        const auto t0 = std::chrono::steady_clock::now();
                        prefill_all_tokens(
                            ctx, body_delta, kv_cache_,
                            generate_params_.prefill_step_size);
                        body_prefill_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
                        body_prefill_tok = body_delta.size();
                    }
                    copy_mamba_snapshots_from(kv_cache_, turn_mamba_snaps);
                    prefill_tokens.assign(
                        tokens.begin() + static_cast<std::ptrdiff_t>(body.size()),
                        tokens.end());
                    residual_note = "body-suffix";
                    snap_at_body = true;
                    used = true;
                }
            }
            if (!used) {
                clear_residual_state();
                residual_note = "lcp-fallback-full";
            }
        }

        if (kv_cache_.empty()) {
            kv_cache_ = ctx.new_cache_fn(generate_params_);
            if (std::strcmp(residual_note, "lcp-suffix") != 0 &&
                std::strcmp(residual_note, "seq-append") != 0 &&
                std::strcmp(residual_note, "body-suffix") != 0 &&
                std::strcmp(residual_note, "nontrim-fallback-full") != 0 &&
                std::strcmp(residual_note, "offset-fallback-full") != 0 &&
                std::strcmp(residual_note, "lcp-fallback-full") != 0) {
                residual_note = residual_opt ? residual_note : "full";
            }
        }

        // First residual turn (or fallback): prefill stable body, snapshot,
        // then only the generation-prompt suffix goes through TokenIterator.
        if (residual_opt && !snap_at_body && !body.empty() &&
            residual_note != nullptr &&
            (std::strcmp(residual_note, "full") == 0 ||
             std::strcmp(residual_note, "lcp-fallback-full") == 0 ||
             std::strcmp(residual_note, "nontrim-fallback-full") == 0)) {
            const auto t0 = std::chrono::steady_clock::now();
            prefill_all_tokens(
                ctx, body, kv_cache_, generate_params_.prefill_step_size);
            body_prefill_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            body_prefill_tok = body.size();
            copy_mamba_snapshots_from(kv_cache_, turn_mamba_snaps);
            snap_at_body = true;
            prefill_tokens.assign(
                tokens.begin() + static_cast<std::ptrdiff_t>(body.size()),
                tokens.end());
        }

        if (prefill_tokens.empty()) {
            throw std::runtime_error("ChatSession: residual left an empty prefill");
        }

        auto token_array = mx::array(
            prefill_tokens.data(),
            {static_cast<int>(prefill_tokens.size())},
            mx::int32);

        LMInput lm_input(token_array);

        TokenIterator iter(
            ctx, lm_input, std::move(kv_cache_), generate_params_);

        if (residual_opt && !snap_at_body) {
            iter.copy_mamba_snapshots(turn_mamba_snaps);
        }

        NaiveStreamingDetokenizer detokenizer;
        auto decode = [&ctx](const std::vector<int>& toks) -> std::string {
            return ctx.decode_fn(toks);
        };

        std::string assistant_response;
        int generated_count = 0;
        std::vector<int> generated_ids;

        while (auto maybe_token = iter.next()) {
            int token_id = *maybe_token;
            generated_ids.push_back(token_id);

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
        bool residual_kept = false;
        if (residual_opt && residual_kv_rollback_ok(kv_cache_) &&
            (turn_mamba_snaps.empty() || turn_mamba_snaps.size() == kv_cache_.size())) {
            last_templated_tokens_ = tokens;
            last_sequence_tokens_ = tokens;
            last_sequence_tokens_.insert(
                last_sequence_tokens_.end(), generated_ids.begin(), generated_ids.end());
            last_body_tokens_ = body;
            last_mamba_snapshots_ = std::move(turn_mamba_snaps);
            last_residual_instructions_ = instructions_;
            residual_kept = true;
        } else {
            kv_cache_.clear();
            last_templated_tokens_.clear();
            last_sequence_tokens_.clear();
            last_body_tokens_.clear();
            last_mamba_snapshots_.clear();
            last_residual_instructions_.reset();
        }

        const auto info_full = iter.completion_info(
            static_cast<int>(prefill_tokens.size()));
        const size_t prefill_tok_logged = body_prefill_tok + prefill_tokens.size();
        const double prefill_s_logged = body_prefill_s + info_full.prompt_time;
        if (turn_log) {
            std::cerr << "[chat-session] turn hist_msgs=" << turn_messages.size()
                      << " template_tok=" << prompt_token_count
                      << " prefill_tok=" << prefill_tok_logged
                      << " prefill_s=" << prefill_s_logged
                      << " gen_tok=" << info_full.generation_token_count
                      << " residual=" << residual_note
                      << " keep=" << (residual_kept ? 1 : 0) << "\n";
        }

        if (on_complete) {
            GenerateInfo info;
            info.prompt_tokens = static_cast<int>(prefill_tok_logged);
            info.generated_tokens = info_full.generation_token_count;
            info.prompt_time_s = prefill_s_logged;
            info.generation_time_s = info_full.generation_time;
            on_complete(info);
        }
    });
}

} // namespace mlx_lm
