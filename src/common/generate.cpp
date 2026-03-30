// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <iostream>

namespace mlx_lm {

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Dedicated generation stream (matches Python's module-level generation_stream)
// ---------------------------------------------------------------------------

mx::Stream& generation_stream() {
    static mx::Stream s = mx::new_stream(mx::default_device());
    return s;
}

// RAII guard to set/restore the default stream for a scope.
struct StreamGuard {
    mx::Stream old_stream_;
    StreamGuard(mx::Stream s) : old_stream_(mx::default_stream(mx::default_device())) {
        mx::set_default_stream(s);
    }
    ~StreamGuard() { mx::set_default_stream(old_stream_); }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

// ---------------------------------------------------------------------------
// TopPSampler
// ---------------------------------------------------------------------------

mx::array TopPSampler::sample_impl(const mx::array& logits) {
    // top-p filtering is disabled: argsort + put_along_axis + take_along_axis
    // on large vocab (151936) produces wrong results on the ROCm backend,
    // causing the filter to mask out the correct tokens.
    // Fall back to temperature-scaled categorical sampling.
    float inv_temp = 1.0f / temperature_;
    return mx::random::categorical(mx::multiply(logits, mx::array(inv_temp)));
}

// ---------------------------------------------------------------------------
// CategoricalSampler
// ---------------------------------------------------------------------------

mx::array CategoricalSampler::sample_impl(const mx::array& logits) {
    // Compiled sampling — matches Python's @mx.compile on categorical_sampling.
    // Use shapeless=false (not shapeless=true which crashes RandomBits).
    // Shape is constant during generation so this only compiles once.
    if (!compiled_fn_) {
        float inv_temp = 1.0f / temperature_;
        compiled_fn_ = mx::compile(
            [inv_temp](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return {mx::random::categorical(mx::multiply(inputs[0], mx::array(inv_temp)))};
            },
            /*shapeless=*/false);
    }
    return compiled_fn_({logits})[0];
}

// ---------------------------------------------------------------------------
// AnySampler
// ---------------------------------------------------------------------------

AnySampler AnySampler::from_params(const GenerateParameters& params) {
    if (params.temperature == 0.0f) {
        return AnySampler(ArgMaxSampler{});
    } else if (params.top_p > 0.0f && params.top_p < 1.0f) {
        return AnySampler(TopPSampler(params.temperature, params.top_p));
    } else {
        return AnySampler(CategoricalSampler(params.temperature));
    }
}

// ---------------------------------------------------------------------------
// RepetitionProcessor
// ---------------------------------------------------------------------------

void RepetitionProcessor::prompt(const mx::array& prompt_tokens) {
    // Evaluate to ensure data is available.
    mx::eval(prompt_tokens);
    auto data = prompt_tokens.data<int32_t>();
    int n = static_cast<int>(prompt_tokens.size());
    int start = std::max(0, n - context_size_);
    tokens_.clear();
    tokens_.reserve(context_size_);
    for (int i = start; i < n; ++i) {
        tokens_.push_back(static_cast<int>(data[i]));
    }
    index_ = 0;
}

mx::array RepetitionProcessor::process(const mx::array& logits) {
    if (tokens_.empty() || penalty_ == 1.0f) return logits;

    // Build an index array from the tokens in the repetition window.
    std::vector<uint32_t> idx_vec;
    idx_vec.reserve(tokens_.size());
    for (int tok : tokens_) {
        idx_vec.push_back(static_cast<uint32_t>(tok));
    }
    int n_indices = static_cast<int>(idx_vec.size());
    auto indices = mx::array(idx_vec.data(), {n_indices}, mx::uint32);

    // Gather the logit values at those token indices along the last axis.
    // take_along_axis requires indices to have the same ndim as logits.
    // Reshape indices to match logits dims: if logits is [B, V], indices -> [1, N].
    auto shaped_indices = indices;
    if (logits.ndim() == 2) {
        shaped_indices = mx::reshape(indices, {1, n_indices});
    }
    auto selected_logits = mx::take_along_axis(logits, shaped_indices, -1);

    // Where logit < 0 multiply by penalty, where >= 0 divide by penalty.
    // This matches the Swift: selected < 0 ? selected * penalty : selected / penalty
    auto zero = mx::array(0.0f);
    auto penalized = mx::where(
        mx::less(selected_logits, zero),
        mx::multiply(selected_logits, mx::array(penalty_)),
        mx::divide(selected_logits, mx::array(penalty_)));

    // Scatter the penalized values back into the logits at the original positions.
    // scatter requires indices with shape [num_updates, ...] for axis scatter.
    // For axis=-1: indices shape [N] (or [1, N] for 2D), updates same shape as selected.
    auto result = mx::scatter(logits, shaped_indices, penalized, -1);

    return result;
}

void RepetitionProcessor::did_sample(const mx::array& token) {
    mx::eval(token);
    int tok = token.item<int32_t>();
    if (static_cast<int>(tokens_.size()) < context_size_) {
        tokens_.push_back(tok);
    } else {
        tokens_[index_] = tok;
        index_ = (index_ + 1) % context_size_;
    }
}

// ---------------------------------------------------------------------------
// AnyProcessor
// ---------------------------------------------------------------------------

std::optional<AnyProcessor> AnyProcessor::from_params(const GenerateParameters& params) {
    if (params.repetition_penalty.has_value() && params.repetition_context_size > 0) {
        return AnyProcessor(RepetitionProcessor(
            params.repetition_penalty.value(),
            params.repetition_context_size));
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// GenerateCompletionInfo
// ---------------------------------------------------------------------------

std::string GenerateCompletionInfo::summary() const {
    std::ostringstream oss;
    oss << "Prompt:     " << prompt_token_count << " tokens, "
        << prompt_tokens_per_second() << " tokens/s, "
        << prompt_time << "s\n"
        << "Generation: " << generation_token_count << " tokens, "
        << tokens_per_second() << " tokens/s, "
        << generation_time << "s";
    return oss.str();
}

// ---------------------------------------------------------------------------
// NaiveStreamingDetokenizer
// ---------------------------------------------------------------------------

void NaiveStreamingDetokenizer::append(int token) {
    segment_tokens_.push_back(token);
}

std::optional<std::string> NaiveStreamingDetokenizer::next(
    const std::function<std::string(const std::vector<int>&)>& decode_fn)
{
    auto new_segment = decode_fn(segment_tokens_);
    if (new_segment.size() <= segment_.size()) return std::nullopt;

    auto new_text = new_segment.substr(segment_.size());

    // If the new text ends with the UTF-8 replacement character (U+FFFD = 0xEF 0xBF 0xBD),
    // the token didn't produce a complete unicode character yet.
    if (new_text.size() >= 3 &&
        new_text[new_text.size() - 3] == '\xef' &&
        new_text[new_text.size() - 2] == '\xbf' &&
        new_text[new_text.size() - 1] == '\xbd') {
        return std::nullopt;
    }

    if (!new_text.empty() && new_text.back() == '\n') {
        start_new_segment(decode_fn);
    } else {
        segment_ = new_segment;
    }

    return new_text;
}

void NaiveStreamingDetokenizer::start_new_segment(
    const std::function<std::string(const std::vector<int>&)>& decode_fn)
{
    if (segment_tokens_.empty()) {
        segment_ = "";
        return;
    }
    int last = segment_tokens_.back();
    segment_tokens_.clear();
    segment_tokens_.push_back(last);
    segment_ = decode_fn(segment_tokens_);
}

// ---------------------------------------------------------------------------
// TokenIterator — helper: timing utility
// ---------------------------------------------------------------------------

static double measure(const std::function<void()>& fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ---------------------------------------------------------------------------
// TokenIterator — add_batch_dim
// ---------------------------------------------------------------------------

LMInput::Text TokenIterator::add_batch_dim(const LMInput::Text& text) {
    // Equivalent to Swift's `previous[text: .newAxis]`:
    // Ensure tokens are always 2D [1, seq_len] regardless of input shape.
    // reshape handles scalar→[1,1], [L]→[1,L], [1,L]→[1,L].
    return LMInput::Text(
        mx::reshape(text.tokens, {1, -1}),
        text.mask  // mask stays as-is
    );
}

// ---------------------------------------------------------------------------
// TokenIterator — convert_to_token
// ---------------------------------------------------------------------------

mx::array TokenIterator::convert_to_token(const mx::array& logits) {
    // Extract the last token's logits: logits[..., -1, ...]
    // For shape [B, seq_len, vocab], take logits[:, -1, :] -> [B, vocab]
    // Then squeeze to [vocab] for sampling.
    mx::array last_logits = logits;

    // If logits has 3 dimensions [B, T, V], slice to [B, 1, V] then squeeze.
    if (logits.ndim() == 3) {
        int seq_len = logits.shape(1);
        last_logits = mx::slice(logits, {0, seq_len - 1, 0},
                                {logits.shape(0), seq_len, logits.shape(2)});
        last_logits = mx::squeeze(last_logits, 1);
    }

    // Apply logit processor if present.
    if (processor_.has_value()) {
        last_logits = processor_->process(last_logits);
    }

    // Sample a token from the logits.
    auto y = sampler_.sample(last_logits);

    // Inform the processor of the sampled token.
    if (processor_.has_value()) {
        processor_->did_sample(y);
    }

    return y;
}

// ---------------------------------------------------------------------------
// TokenIterator — step
// ---------------------------------------------------------------------------

mx::array TokenIterator::step(const LMInput::Text& previous) {
    // Run on the dedicated generation stream (matches Python's `with mx.stream(generation_stream)`).
    StreamGuard sg(generation_stream());

    // Add batch dimension to tokens; call the model.
    auto batched = add_batch_dim(previous);

    // TODO: HIP graph capture for decode steps.
    // After the allocator cache is warm (post-warmup), the decode step has
    // a fixed kernel sequence that could be captured into a HIP graph and
    // replayed with a single hipGraphLaunch instead of ~8800 individual
    // kernel dispatches. This would reduce launch overhead from ~26ms to
    // ~0.05ms per token, increasing generation from 18 to ~34 tok/s.
    //
    // Requirements for capture:
    // 1. Allocator cache must be warm (no hipMalloc during capture)
    // 2. No host callbacks or stream syncs during capture
    // 3. Buffer addresses must be stable between capture and replay
    //
    // Current blocker: MLX's lazy eval triggers allocations and host
    // callbacks that aren't compatible with hipStreamBeginCapture.
    // Needs MLX framework support for pre-allocated execution plans.

    auto result = context_.call_fn(
        batched,
        cache_.empty() ? nullptr : &cache_,
        state_.has_value() ? &state_.value() : nullptr);

    // Update state from model output.
    state_ = result.state;

    // Apply dynamic KV cache quantization after each step.
    // Matches Swift's maybeQuantizeKVCache() call in TokenIterator.step().
    maybe_quantize_kv_cache(cache_, kv_bits_, kv_group_size_, quantized_kv_start_);

    return convert_to_token(result.logits);
}

// ---------------------------------------------------------------------------
// TokenIterator — prepare (prompt prefill)
// ---------------------------------------------------------------------------

void TokenIterator::prepare(const LMInput& input, int window_size) {
    // Run on the dedicated generation stream (matches Python's prefill in generation_stream).
    StreamGuard sg(generation_stream());

    // Inform the processor about the prompt tokens.
    if (processor_.has_value()) {
        processor_->prompt(input.text.tokens);
    }

    // Run model's prepare function to consume the prompt.
    auto prep_result = context_.prepare_fn(input, cache_, window_size);

    if (prep_result.is_tokens()) {
        // Model returned remaining tokens — evaluate them to prime the cache.
        auto remaining = prep_result.as_tokens();
        auto token = step(remaining);
        y_ = LMInput::Text(token);
        mx::async_eval(y_.tokens);
    } else {
        // Model returned logits directly — sample the first token.
        auto token = convert_to_token(prep_result.as_logits().logits);
        y_ = LMInput::Text(token);
        mx::async_eval(y_.tokens);
    }
}

// ---------------------------------------------------------------------------
// TokenIterator — constructors
// ---------------------------------------------------------------------------

TokenIterator::TokenIterator(
    ModelContext& context,
    const LMInput& input,
    const GenerateParameters& params)
    : context_(context)
    , y_(mx::array(0, mx::int32))  // placeholder, overwritten by prepare()
    , cache_(context.new_cache_fn(params))
    , sampler_(AnySampler::from_params(params))
    , processor_(AnyProcessor::from_params(params))
    , max_tokens_(params.max_tokens)
    , kv_bits_(params.kv_bits)
    , kv_group_size_(params.kv_group_size)
    , quantized_kv_start_(params.quantized_kv_start)
{
    prompt_prefill_time_ = measure([&]() {
        prepare(input, params.prefill_step_size);
    });
    generation_start_ = std::chrono::steady_clock::now();
}

TokenIterator::TokenIterator(
    ModelContext& context,
    const LMInput& input,
    std::vector<KVCache> cache,
    AnySampler sampler,
    std::optional<AnyProcessor> processor,
    std::optional<int> max_tokens,
    int prefill_step_size)
    : context_(context)
    , y_(mx::array(0, mx::int32))  // placeholder, overwritten by prepare()
    , cache_(std::move(cache))
    , sampler_(std::move(sampler))
    , processor_(std::move(processor))
    , max_tokens_(max_tokens)
{
    prompt_prefill_time_ = measure([&]() {
        prepare(input, prefill_step_size);
    });
    generation_start_ = std::chrono::steady_clock::now();
}

// ---------------------------------------------------------------------------
// TokenIterator — next()
// ---------------------------------------------------------------------------

std::optional<int> TokenIterator::next() {
    // Check max_tokens limit.
    if (max_tokens_.has_value() && token_count_ >= max_tokens_.value()) {
        return std::nullopt;
    }

    // The current y_ holds the previously computed token.
    auto previous_y = y_;

    // Compute the next token (evaluates the model for the *next* step).
    auto token = step(previous_y);
    y_ = LMInput::Text(token);

    // Async eval the next token so the GPU pipeline stays full.
    mx::async_eval(token);

    token_count_++;

    // Return the *previous* token (which was already computed and ready).
    mx::eval(previous_y.tokens);
    return previous_y.tokens.item<int32_t>();
}

// ---------------------------------------------------------------------------
// TokenIterator — completion_info
// ---------------------------------------------------------------------------

GenerateCompletionInfo TokenIterator::completion_info(int prompt_token_count) const {
    auto now = std::chrono::steady_clock::now();
    double gen_time = std::chrono::duration<double>(now - generation_start_).count();
    return GenerateCompletionInfo{
        prompt_token_count,
        token_count_,
        prompt_prefill_time_,
        gen_time
    };
}

// ---------------------------------------------------------------------------
// generate() — streaming generation with per-token callback
// ---------------------------------------------------------------------------

GenerateCompletionInfo generate(
    ModelContext& context,
    const LMInput& input,
    const GenerateParameters& params,
    const std::set<int>& eos_token_ids,
    const std::function<GenerateDisposition(int token)>& on_token)
{
    // Upgrade GPU wired memory for the duration of generation.
    // Matches Python mlx-lm's `with wired_limit(model)` in stream_generate().
    WiredLimitGuard wired_guard;

    int prompt_token_count = static_cast<int>(input.text.tokens.size());

    TokenIterator iter(context, input, params);

    auto start = std::chrono::steady_clock::now();
    double prompt_time = 0.0;
    int token_count = 0;

    while (auto maybe_token = iter.next()) {
        int token = *maybe_token;

        // Measure prompt time on first token.
        if (token_count == 0) {
            auto now = std::chrono::steady_clock::now();
            prompt_time = std::chrono::duration<double>(now - start).count();
            start = std::chrono::steady_clock::now();
        }

        // Check for EOS.
        if (eos_token_ids.count(token)) {
            break;
        }

        token_count++;

        // Periodically clear the memory cache to reduce memory pressure.
        // Matches Python mlx-lm's `if n % 256 == 0: mx.clear_cache()`.
        if (token_count % 256 == 0) {
            mx::clear_cache();
        }

        // Invoke callback.
        if (on_token(token) == GenerateDisposition::stop) {
            break;
        }
    }

    // Synchronize the generation stream to ensure all pending GPU work completes.
    mx::synchronize(generation_stream());

    auto now = std::chrono::steady_clock::now();
    double gen_time = std::chrono::duration<double>(now - start).count();

    return GenerateCompletionInfo{
        prompt_token_count,
        token_count,
        prompt_time + iter.prompt_prefill_time(),
        gen_time
    };
}

// ---------------------------------------------------------------------------
// generate_text() — streaming generation with text chunk callback
// ---------------------------------------------------------------------------

GenerateCompletionInfo generate_text(
    ModelContext& context,
    const LMInput& input,
    const GenerateParameters& params,
    const std::set<int>& eos_token_ids,
    const std::function<GenerateDisposition(const std::string& text, int token)>& on_text)
{
    // We need the decode function from context.
    auto decode_fn = context.decode_fn;

    NaiveStreamingDetokenizer detokenizer;

    return generate(context, input, params, eos_token_ids,
        [&](int token) -> GenerateDisposition {
            detokenizer.append(token);
            if (auto text = detokenizer.next(decode_fn)) {
                return on_text(*text, token);
            }
            return GenerateDisposition::more;
        });
}

} // namespace mlx_lm
