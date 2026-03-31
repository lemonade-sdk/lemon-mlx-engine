// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/common/wired_limit_guard.h>
#include <mlx/mlx.h>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace mlx_lm {

// Forward declaration.
struct ModelContext;

// Dedicated GPU stream for generation (matches Python's generation_stream).
// Created once, reused across generate() calls.
mlx::core::Stream& generation_stream();

// ---------------------------------------------------------------------------
// LogitSampler — CRTP base, no virtual functions.
// Derived must implement: mlx::core::array sample_impl(const mlx::core::array& logits)
// ---------------------------------------------------------------------------
template <typename Derived>
class LogitSampler {
public:
    mlx::core::array sample(const mlx::core::array& logits) {
        return static_cast<Derived*>(this)->sample_impl(logits);
    }
};

// ArgMax sampler — deterministic, picks most likely token.
class ArgMaxSampler : public LogitSampler<ArgMaxSampler> {
    friend class LogitSampler<ArgMaxSampler>;
    mlx::core::array sample_impl(const mlx::core::array& logits) {
        return mlx::core::argmax(logits, -1);
    }
};

// Compiled function type for JIT-compiled samplers.
using CompiledFn = std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)>;

// Top-p (nucleus) sampler.
// Compiled via mx::compile — matches Python's compiled apply_top_p + categorical_sampling.
class TopPSampler : public LogitSampler<TopPSampler> {
    friend class LogitSampler<TopPSampler>;

    float temperature_;
    float top_p_;
    mutable CompiledFn compiled_top_p_;
    mutable CompiledFn compiled_categorical_;

    mlx::core::array sample_impl(const mlx::core::array& logits);

public:
    TopPSampler(float temperature, float top_p)
        : temperature_(temperature), top_p_(top_p) {}
};

// Categorical sampler with temperature (compiled via mx::compile).
class CategoricalSampler : public LogitSampler<CategoricalSampler> {
    friend class LogitSampler<CategoricalSampler>;

    float temperature_;
    mutable CompiledFn compiled_fn_;

    mlx::core::array sample_impl(const mlx::core::array& logits);

public:
    explicit CategoricalSampler(float temperature) : temperature_(temperature) {}
};

// Type-erased sampler (variant-based, no virtuals).
class AnySampler {
public:
    using SamplerVariant = std::variant<ArgMaxSampler, TopPSampler, CategoricalSampler>;

    explicit AnySampler(SamplerVariant s) : impl_(std::move(s)) {}

    mlx::core::array sample(const mlx::core::array& logits) {
        return std::visit([&](auto& s) { return s.sample(logits); }, impl_);
    }

    // Create sampler from GenerateParameters.
    static AnySampler from_params(const GenerateParameters& params);

private:
    SamplerVariant impl_;
};

// ---------------------------------------------------------------------------
// LogitProcessor — interface for modifying logits before sampling.
//
// The processor is called with the prompt tokens before generation begins,
// then for each step it can adjust the logits, and is informed of the
// sampled token afterwards.
//
// Port of Swift's LogitProcessor protocol.
// ---------------------------------------------------------------------------
class RepetitionProcessor {
    float penalty_;
    int context_size_;
    std::vector<int> tokens_;
    int index_ = 0;

public:
    RepetitionProcessor(float penalty, int context_size)
        : penalty_(penalty), context_size_(context_size) {}

    void prompt(const mlx::core::array& prompt_tokens);
    mlx::core::array process(const mlx::core::array& logits);
    void did_sample(const mlx::core::array& token);
};

// Type-erased logit processor (variant-based, no virtuals).
// Currently only wraps RepetitionProcessor; extend the variant for new processors.
class AnyProcessor {
public:
    using ProcessorVariant = std::variant<RepetitionProcessor>;

    explicit AnyProcessor(ProcessorVariant p) : impl_(std::move(p)) {}

    void prompt(const mlx::core::array& prompt_tokens) {
        std::visit([&](auto& p) { p.prompt(prompt_tokens); }, impl_);
    }

    mlx::core::array process(const mlx::core::array& logits) {
        return std::visit([&](auto& p) { return p.process(logits); }, impl_);
    }

    void did_sample(const mlx::core::array& token) {
        std::visit([&](auto& p) { p.did_sample(token); }, impl_);
    }

    // Create processor from GenerateParameters. Returns nullopt if no processing needed.
    static std::optional<AnyProcessor> from_params(const GenerateParameters& params);

private:
    ProcessorVariant impl_;
};

// ---------------------------------------------------------------------------
// Generation result info — simple legacy struct (kept for backward compat).
// ---------------------------------------------------------------------------
struct GenerateInfo {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double prompt_time_s = 0.0;
    double generation_time_s = 0.0;

    double tokens_per_second() const {
        return generation_time_s > 0.0 ? generated_tokens / generation_time_s : 0.0;
    }
    double prompt_tokens_per_second() const {
        return prompt_time_s > 0.0 ? prompt_tokens / prompt_time_s : 0.0;
    }
};

// ---------------------------------------------------------------------------
// GenerateCompletionInfo — detailed metadata about a completed generation.
// Port of Swift's GenerateCompletionInfo struct.
// ---------------------------------------------------------------------------
struct GenerateCompletionInfo {
    int prompt_token_count = 0;
    int generation_token_count = 0;
    double prompt_time = 0.0;        // seconds
    double generation_time = 0.0;    // seconds

    double prompt_tokens_per_second() const {
        return prompt_time > 0.0 ? static_cast<double>(prompt_token_count) / prompt_time : 0.0;
    }

    double tokens_per_second() const {
        return generation_time > 0.0
            ? static_cast<double>(generation_token_count) / generation_time : 0.0;
    }

    std::string summary() const;
};

// A single chunk of generated output.
struct GenerateChunk {
    std::string text;
    int token_id = 0;
};

// Action returned from a generation callback to control iteration.
// Mirrors Swift's GenerateDisposition.
enum class GenerateDisposition {
    more,   // keep producing tokens
    stop    // stop producing tokens
};

// Callback for streaming generation.
// Returns true to continue, false to stop.
using GenerateCallback = std::function<bool(const GenerateChunk& chunk)>;

// Streaming detokenizer — accumulates tokens and emits decoded text.
class NaiveStreamingDetokenizer {
    std::vector<int> segment_tokens_;
    std::string segment_;

public:
    void append(int token);
    std::optional<std::string> next(
        const std::function<std::string(const std::vector<int>&)>& decode_fn);
    void start_new_segment(
        const std::function<std::string(const std::vector<int>&)>& decode_fn);
};

// ---------------------------------------------------------------------------
// TokenIterator — generates tokens one at a time from a language model.
//
// Port of Swift's TokenIterator. Manages the model forward pass, sampling,
// logit processing, and KV cache internally.
//
// Usage:
//     ModelContext ctx = ...;
//     GenerateParameters params;
//     auto input_tokens = mx::array({...});
//     LMInput input(input_tokens);
//
//     TokenIterator iter(ctx, input, params);
//     while (auto token = iter.next()) {
//         int tok = *token;
//         if (is_eos(tok)) break;
//         // use tok...
//     }
//     auto info = iter.completion_info();
//
// ---------------------------------------------------------------------------
class TokenIterator {
public:
    // Construct from a ModelContext (type-erased model), input, and parameters.
    // This runs the prompt prefill immediately.
    TokenIterator(
        ModelContext& context,
        const LMInput& input,
        const GenerateParameters& params);

    // Construct with explicit cache, sampler, and processor.
    TokenIterator(
        ModelContext& context,
        const LMInput& input,
        std::vector<KVCache> cache,
        AnySampler sampler,
        std::optional<AnyProcessor> processor,
        std::optional<int> max_tokens = std::nullopt,
        int prefill_step_size = 512);

    // Generate the next token. Returns nullopt when max_tokens is reached.
    // The caller is responsible for checking EOS conditions.
    std::optional<int> next();

    // Number of tokens generated so far.
    int token_count() const { return token_count_; }

    // Time spent on prompt prefill (seconds).
    double prompt_prefill_time() const { return prompt_prefill_time_; }

    // Build completion info from current state (call after iteration is done).
    GenerateCompletionInfo completion_info(int prompt_token_count) const;

    // Move the KV cache out of the iterator (for multi-turn reuse).
    std::vector<KVCache> take_cache() { return std::move(cache_); }

private:
    // Add batch dimension to tokens only (mask stays as-is).
    // Equivalent to Swift's `previous[text: .newAxis]`.
    static LMInput::Text add_batch_dim(const LMInput::Text& text);

    // Run model forward for one step, process logits, and sample a token.
    mlx::core::array step(const LMInput::Text& previous);

    // Convert raw logits to a sampled token.
    mlx::core::array convert_to_token(const mlx::core::array& logits);

    // Run the prompt through the model (prefill).
    void prepare(const LMInput& input, int window_size);

    // References (model context must outlive this iterator).
    ModelContext& context_;

    // Current token(s) — the last produced tokens ready for next step.
    LMInput::Text y_;

    // KV cache for this generation session.
    std::vector<KVCache> cache_;

    // Sampler and optional processor.
    AnySampler sampler_;
    std::optional<AnyProcessor> processor_;

    // Model output state (for cross-attention etc.).
    std::optional<LMOutput::State> state_;

    // Generation limits and counters.
    std::optional<int> max_tokens_;
    int token_count_ = 0;

    // KV cache quantization parameters.
    std::optional<int> kv_bits_;
    int kv_group_size_ = 64;
    int quantized_kv_start_ = 0;

    // Timing.
    double prompt_prefill_time_ = 0.0;
    std::chrono::steady_clock::time_point generation_start_;

    // HIP Graph capture state machine for decode acceleration.
    // After warmup tokens, the decode step has a fixed kernel sequence.
    // We capture it into a HIP graph for single-dispatch replay.
    enum class GraphState {
      Warmup,     // First tokens: normal execution, measuring arena size
      Profiling,  // Arena active: run decode to record allocation pattern
      Capturing,  // Arena reset + graph capture: record kernel sequence
      Replaying,  // Arena reset + graph replay each step
      Disabled,   // Capture failed or not supported
    };
    GraphState graph_state_ = GraphState::Warmup;
    int warmup_steps_ = 0;
    static constexpr int kGraphWarmupSteps = 3; // Steps before attempting capture
};

// ---------------------------------------------------------------------------
// Streaming generate() — drives a TokenIterator with a callback.
//
// Calls the callback for each generated text chunk. The callback receives
// GenerateDisposition control: return GenerateDisposition::more to continue,
// GenerateDisposition::stop to halt.
//
// Parameters:
//   context       - ModelContext with type-erased model + tokenizer
//   input         - prepared LMInput (tokens + optional images)
//   params        - generation parameters (temperature, top_p, etc.)
//   eos_token_ids - set of token IDs that signal end of sequence
//   on_token      - callback invoked for each token; receives token ID,
//                   returns GenerateDisposition
//
// Returns: GenerateCompletionInfo with timing and token count statistics.
// ---------------------------------------------------------------------------
GenerateCompletionInfo generate(
    ModelContext& context,
    const LMInput& input,
    const GenerateParameters& params,
    const std::set<int>& eos_token_ids,
    const std::function<GenerateDisposition(int token)>& on_token);

// Streaming generate with text chunks — same as above but decodes tokens
// into text using the context's decode_fn and calls back with text chunks.
//
// Returns: GenerateCompletionInfo with timing and token count statistics.
GenerateCompletionInfo generate_text(
    ModelContext& context,
    const LMInput& input,
    const GenerateParameters& params,
    const std::set<int>& eos_token_ids,
    const std::function<GenerateDisposition(const std::string& text, int token)>& on_text);

} // namespace mlx_lm
