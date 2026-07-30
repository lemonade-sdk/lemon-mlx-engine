// Copyright (C) 2024-2025 — streaming generation loop brake (phrase / line)
#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mlx_lm {

/// Thresholds for detecting runaway decode loops (thinking / CoT thrash).
/// Defaults match residual multi-turn hard-loop criteria used in local ROCm
/// experiments: ~40–80 char (or 8–16 token) phrase repeated ≥4 consecutively,
/// or the same non-empty line repeated ≥6 times in a row.
struct LoopBrakeParams {
    // Task: ~40–80 chars; slightly widened so line-with-prefix units still match.
    int min_phrase_chars = 32;
    int max_phrase_chars = 120;
    // Consecutive suffix stack (tight). Frequency window uses ngram_freq_threshold.
    int phrase_repeat_threshold = 3;

    int min_phrase_tokens = 8;
    int max_phrase_tokens = 16;
    int token_phrase_repeat_threshold = 3;

    int same_line_threshold = 5;
    /// Ignore very short lines (list markers, blank-ish noise).
    int min_line_chars = 20;

    /// Non-adjacent thrash: same 8–12 word n-gram count in a trailing window.
    int ngram_min_words = 8;
    int ngram_max_words = 12;
    int ngram_freq_threshold = 5;
    /// Only scan the last N words for frequency (cost bound).
    int ngram_window_words = 400;
};

/// Why the brake fired (for tests / diagnostics).
enum class LoopBrakeReason {
    None,
    PhraseChars,
    PhraseTokens,
    SameLine,
    WordNgramFreq,
};

/// Split `text` into whitespace-separated words (simple; good enough for CoT).
inline std::vector<std::string_view> split_words(std::string_view text) {
    std::vector<std::string_view> words;
    std::size_t i = 0;
    const std::size_t n = text.size();
    while (i < n) {
        while (i < n && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t' ||
                         text[i] == '\r')) {
            ++i;
        }
        if (i >= n) {
            break;
        }
        std::size_t j = i;
        while (j < n && text[j] != ' ' && text[j] != '\n' && text[j] != '\t' &&
               text[j] != '\r') {
            ++j;
        }
        words.emplace_back(text.substr(i, j - i));
        i = j;
    }
    return words;
}

/// True if any 8–12 word n-gram appears ≥ threshold times in a trailing word window
/// (catches CoT thrash that is not a pure end-suffix stack).
inline bool has_frequent_word_ngram_loop(std::string_view text,
                                         int min_words,
                                         int max_words,
                                         int threshold,
                                         int window_words) {
    if (min_words < 1 || max_words < min_words || threshold < 2) {
        return false;
    }
    auto words = split_words(text);
    if (static_cast<int>(words.size()) < min_words * threshold) {
        return false;
    }
    const int n = static_cast<int>(words.size());
    const int start = std::max(0, n - window_words);
    // Map key = joined words with single spaces (bounded work: only max_words
    // lengths, and only ngrams that appear as candidates from end first).
    for (int L = max_words; L >= min_words; --L) {
        if (n - start < L * threshold) {
            continue;
        }
        // Count n-grams in [start, n).
        // Use a simple O(W*L) pass with a hash of string keys — W<=400.
        std::vector<std::string> keys;
        keys.reserve(static_cast<std::size_t>(n - start));
        for (int i = start; i + L <= n; ++i) {
            std::string key;
            key.reserve(static_cast<std::size_t>(L) * 8);
            for (int k = 0; k < L; ++k) {
                if (k) {
                    key.push_back(' ');
                }
                key.append(words[static_cast<std::size_t>(i + k)]);
            }
            keys.push_back(std::move(key));
        }
        std::sort(keys.begin(), keys.end());
        int run = 1;
        for (std::size_t i = 1; i < keys.size(); ++i) {
            if (keys[i] == keys[i - 1]) {
                ++run;
                if (run >= threshold) {
                    return true;
                }
            } else {
                run = 1;
            }
        }
    }
    return false;
}

/// Check a full string for a consecutive char-phrase loop (stateless helper).
inline bool has_consecutive_phrase_loop(std::string_view text,
                                        int min_chars,
                                        int max_chars,
                                        int threshold) {
    if (min_chars <= 0 || max_chars < min_chars || threshold < 2) {
        return false;
    }
    const std::size_t n = text.size();
    const std::size_t need_min =
        static_cast<std::size_t>(min_chars) * static_cast<std::size_t>(threshold);
    if (n < need_min) {
        return false;
    }

    // Bound work: only need the last max_chars * threshold characters.
    const std::size_t window =
        static_cast<std::size_t>(max_chars) * static_cast<std::size_t>(threshold);
    const std::size_t start = n > window ? n - window : 0;
    const std::string_view view = text.substr(start);
    const std::size_t m = view.size();

    for (int L = min_chars; L <= max_chars; ++L) {
        const std::size_t need =
            static_cast<std::size_t>(L) * static_cast<std::size_t>(threshold);
        if (m < need) {
            continue;
        }
        // Suffixed run: last need chars must be phrase L repeated threshold times.
        const std::size_t base = m - need;
        const std::string_view phrase = view.substr(base, static_cast<std::size_t>(L));
        // Skip pure punctuation / whitespace stacks (e.g. "----").
        bool has_alnum = false;
        for (char c : phrase) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9')) {
                has_alnum = true;
                break;
            }
        }
        if (!has_alnum) {
            continue;
        }
        bool match = true;
        for (int r = 1; r < threshold; ++r) {
            if (view.substr(base + static_cast<std::size_t>(r) * static_cast<std::size_t>(L),
                            static_cast<std::size_t>(L)) != phrase) {
                match = false;
                break;
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

/// Same as above for a token id sequence.
inline bool has_consecutive_token_phrase_loop(const std::vector<int>& tokens,
                                              int min_len,
                                              int max_len,
                                              int threshold) {
    if (min_len <= 0 || max_len < min_len || threshold < 2) {
        return false;
    }
    const std::size_t n = tokens.size();
    const std::size_t need_min =
        static_cast<std::size_t>(min_len) * static_cast<std::size_t>(threshold);
    if (n < need_min) {
        return false;
    }

    const std::size_t window =
        static_cast<std::size_t>(max_len) * static_cast<std::size_t>(threshold);
    const std::size_t start = n > window ? n - window : 0;

    for (int L = min_len; L <= max_len; ++L) {
        const std::size_t need =
            static_cast<std::size_t>(L) * static_cast<std::size_t>(threshold);
        if (n - start < need) {
            continue;
        }
        const std::size_t base = n - need;
        bool match = true;
        for (int r = 1; r < threshold && match; ++r) {
            for (int i = 0; i < L; ++i) {
                if (tokens[base + static_cast<std::size_t>(i)] !=
                    tokens[base + static_cast<std::size_t>(r) * static_cast<std::size_t>(L) +
                           static_cast<std::size_t>(i)]) {
                    match = false;
                    break;
                }
            }
        }
        if (match) {
            return true;
        }
    }
    return false;
}

/// Incremental detector for streaming generation.
/// Feed token ids and/or decoded text chunks; feed return true when generation
/// should stop (caller breaks the decode loop).
class LoopBrake {
public:
    explicit LoopBrake(LoopBrakeParams params = {}) : params_(params) {
        max_text_ =
            static_cast<std::size_t>(params_.max_phrase_chars) *
                static_cast<std::size_t>(params_.phrase_repeat_threshold) +
            512;
        max_tokens_ =
            static_cast<std::size_t>(params_.max_phrase_tokens) *
                static_cast<std::size_t>(params_.token_phrase_repeat_threshold) +
            32;
    }

    void reset() {
        text_.clear();
        tokens_.clear();
        line_buf_.clear();
        last_line_.clear();
        consecutive_same_line_ = 0;
        reason_ = LoopBrakeReason::None;
    }

    LoopBrakeReason reason() const { return reason_; }
    bool tripped() const { return reason_ != LoopBrakeReason::None; }

    /// Feed a decoded text chunk. Returns true if the brake trips.
    bool feed_text(std::string_view chunk) {
        if (tripped() || chunk.empty()) {
            return tripped();
        }
        text_.append(chunk.data(), chunk.size());
        trim_text_();
        update_lines_(chunk);  // may set SameLine

        if (!tripped() &&
            has_consecutive_phrase_loop(text_,
                                        params_.min_phrase_chars,
                                        params_.max_phrase_chars,
                                        params_.phrase_repeat_threshold)) {
            reason_ = LoopBrakeReason::PhraseChars;
        } else if (!tripped() &&
                   has_frequent_word_ngram_loop(text_,
                                               params_.ngram_min_words,
                                               params_.ngram_max_words,
                                               params_.ngram_freq_threshold,
                                               params_.ngram_window_words)) {
            reason_ = LoopBrakeReason::WordNgramFreq;
        }
        return tripped();
    }

    /// Feed a generated token id. Returns true if the brake trips.
    bool feed_token(int token_id) {
        if (tripped()) {
            return true;
        }
        tokens_.push_back(token_id);
        if (tokens_.size() > max_tokens_) {
            tokens_.erase(tokens_.begin(),
                          tokens_.begin() +
                              static_cast<std::ptrdiff_t>(tokens_.size() - max_tokens_));
        }
        if (has_consecutive_token_phrase_loop(tokens_,
                                              params_.min_phrase_tokens,
                                              params_.max_phrase_tokens,
                                              params_.token_phrase_repeat_threshold)) {
            reason_ = LoopBrakeReason::PhraseTokens;
            return true;
        }
        return false;
    }

    /// Feed both token and optional text from the same decode step.
    bool feed(int token_id, std::string_view text_chunk) {
        if (feed_token(token_id)) {
            return true;
        }
        if (!text_chunk.empty() && feed_text(text_chunk)) {
            return true;
        }
        return tripped();
    }

private:
    void trim_text_() {
        if (text_.size() > max_text_) {
            text_.erase(0, text_.size() - max_text_);
        }
    }

    void update_lines_(std::string_view chunk) {
        for (char c : chunk) {
            if (c == '\n') {
                commit_line_();
            } else {
                line_buf_.push_back(c);
            }
        }
    }

    void commit_line_() {
        if (!line_buf_.empty() && line_buf_.back() == '\r') {
            line_buf_.pop_back();
        }
        while (!line_buf_.empty() &&
               (line_buf_.back() == ' ' || line_buf_.back() == '\t')) {
            line_buf_.pop_back();
        }

        if (static_cast<int>(line_buf_.size()) >= params_.min_line_chars) {
            if (line_buf_ == last_line_) {
                ++consecutive_same_line_;
            } else {
                last_line_ = line_buf_;
                consecutive_same_line_ = 1;
            }
            if (consecutive_same_line_ >= params_.same_line_threshold) {
                reason_ = LoopBrakeReason::SameLine;
            }
        } else {
            last_line_.clear();
            consecutive_same_line_ = 0;
        }
        line_buf_.clear();
    }

    LoopBrakeParams params_;
    std::size_t max_text_ = 0;
    std::size_t max_tokens_ = 0;
    std::string text_;
    std::vector<int> tokens_;
    std::string line_buf_;
    std::string last_line_;
    int consecutive_same_line_ = 0;
    LoopBrakeReason reason_ = LoopBrakeReason::None;
};

/// One-shot check over a finished generation string (char phrase + line runs).
inline bool has_generation_loop(std::string_view text,
                                const LoopBrakeParams& params = {}) {
    if (has_consecutive_phrase_loop(text,
                                    params.min_phrase_chars,
                                    params.max_phrase_chars,
                                    params.phrase_repeat_threshold)) {
        return true;
    }
    if (has_frequent_word_ngram_loop(text,
                                     params.ngram_min_words,
                                     params.ngram_max_words,
                                     params.ngram_freq_threshold,
                                     params.ngram_window_words)) {
        return true;
    }
    LoopBrake brake(params);
    for (char c : text) {
        if (brake.feed_text(std::string_view(&c, 1))) {
            return true;
        }
    }
    return false;
}

} // namespace mlx_lm
