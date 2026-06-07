// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>

// Forward declarations for ROCm arena/graph support.
// These symbols only exist in the ROCm build of MLX.
#if defined(MLX_BUILD_ROCM)
namespace mlx::core {
  bool gpu_arena_begin(size_t capacity);
  void gpu_arena_reset();
  void gpu_arena_end();
  size_t gpu_arena_used();
  bool gpu_arena_active();
  bool gpu_graph_begin_capture();
  bool gpu_graph_end_capture();
  bool gpu_graph_replay();
  void gpu_graph_reset();
  bool gpu_graph_available();
}
#endif

namespace mlx_lm {

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Dedicated generation stream (matches Python's module-level generation_stream)
// ---------------------------------------------------------------------------

mx::Stream& generation_stream() {
    // MLX Metal streams are thread-local (known issue in MLX 0.31.2:
    // "There is no Stream(gpu, N) in current thread"). Each thread
    // needs its own stream. This matches the fix applied in mlx-lm
    // (thread-local generation stream) and mlx-vlm.
    static thread_local mx::Stream s = mx::new_stream(mx::default_device());
    return s;
}

// RAII guard to set/restore the default stream for a scope.
// On Apple, skip stream switching to avoid Metal thread-affinity issues.
struct StreamGuard {
    mx::Stream old_stream_;
    bool changed_ = false;
    StreamGuard(mx::Stream s) : old_stream_(mx::default_stream(mx::default_device())) {
#ifndef __APPLE__
        if (s != old_stream_) {
            mx::set_default_stream(s);
            changed_ = true;
        }
#endif
    }
    ~StreamGuard() {
#ifndef __APPLE__
        if (changed_) mx::set_default_stream(old_stream_);
#endif
    }
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
};

// ---------------------------------------------------------------------------
// TopPSampler
// ---------------------------------------------------------------------------

mx::array TopPSampler::sample_impl(const mx::array& logits) {
    // top-p filtering is disabled; argsort + take_along_axis on large vocab
    // produces incorrect results. Investigation needed for re-enablement.
    // Fall back to compiled temperature-scaled categorical sampling.
    //
    // On Apple Metal, avoid mx::compile — it captures stream state at
    // compilation time and becomes unstable across generation requests.
    // Direct sampling is fast enough on Apple Silicon.
#ifdef __APPLE__
    return mx::random::categorical(
        mx::multiply(logits, mx::array(1.0f / temperature_)));
#else
    if (!compiled_categorical_) {
        float inv_temp = 1.0f / temperature_;
        compiled_categorical_ = mx::compile(
            [inv_temp](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return {mx::random::categorical(mx::multiply(inputs[0], mx::array(inv_temp)))};
            },
            /*shapeless=*/false);
    }
    return compiled_categorical_({logits})[0];
#endif
}

// ---------------------------------------------------------------------------
// CategoricalSampler
// ---------------------------------------------------------------------------

mx::array CategoricalSampler::sample_impl(const mx::array& logits) {
    // Compiled sampling — matches Python's @mx.compile on categorical_sampling.
    // Use shapeless=false (not shapeless=true which crashes RandomBits).
    // Shape is constant during generation so this only compiles once.
    //
    // On Apple Metal, avoid mx::compile — it captures stream state at
    // compilation time and becomes unstable across generation requests.
    // Direct sampling is fast enough on Apple Silicon.
#ifdef __APPLE__
    float inv_temp = 1.0f / temperature_;
    return mx::random::categorical(mx::multiply(logits, mx::array(inv_temp)));
#else
    if (!compiled_fn_) {
        float inv_temp = 1.0f / temperature_;
        compiled_fn_ = mx::compile(
            [inv_temp](const std::vector<mx::array>& inputs) -> std::vector<mx::array> {
                return {mx::random::categorical(mx::multiply(inputs[0], mx::array(inv_temp)))};
            },
            /*shapeless=*/false);
    }
    return compiled_fn_({logits})[0];
#endif
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
    // put_along_axis is the mirror of take_along_axis and handles negative axis normalization.
    auto result = mx::put_along_axis(logits, shaped_indices, penalized, -1);

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

    // Append MTP speculative decoding metrics when available.
    if (acceptance_rate() > 0.0) {
        oss << "\nMTP:        drafts=" << mtp_draft_tokens_proposed
            << " accepted=" << mtp_draft_tokens_accepted
            << " speculative_steps=" << mtp_speculative_steps
            << " acceptance_rate=" << std::fixed << std::setprecision(2)
            << (acceptance_rate() * 100.0) << "%";
    }

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
    // Run on the dedicated generation stream.
    StreamGuard sg(generation_stream());

    auto batched = add_batch_dim(previous);

    // --- HIP Graph capture state machine (ROCm only) ---
    // After warmup tokens, capture the decode step as a HIP graph for
    // single-dispatch replay. Uses arena allocator for deterministic addresses.
#if defined(MLX_BUILD_ROCM)
    namespace gpu = mlx::core;

    switch (graph_state_) {
      case GraphState::Warmup:
        warmup_steps_++;
        if (warmup_steps_ >= kGraphWarmupSteps) {
          graph_state_ = GraphState::Profiling;
        }
        break;

      case GraphState::Profiling: {
        // Arena profiling disabled: the arena can't distinguish temporary
        // activations from persistent KV cache updates. Both use malloc(),
        // and freeing the arena destroys KV cache data that lives across
        // steps. Enabling this requires KV cache pre-allocation (#7).
        graph_state_ = GraphState::Disabled;
        break;
      }

      case GraphState::Capturing: {
        // Graph capture records kernels WITHOUT executing them, which
        // leaves the KV cache in a stale state and corrupts subsequent
        // steps. Full graph replay requires:
        // 1. Separating the KV cache update from the captured graph
        // 2. Using hipGraphExecKernelNodeSetParams to update input
        //    token pointers each step
        // 3. Reading output logits from deterministic arena addresses
        //
        // Arena profiling proved the mechanism works (18 KB per step,
        // deterministic addresses). Capture is disabled until the
        // KV cache isolation and pointer update logic is implemented.
        gpu::gpu_arena_end();
        graph_state_ = GraphState::Disabled;
        break;
      }

      case GraphState::Replaying: {
        // Graph replay requires updating input token pointers and reading
        // output logits from the arena — not yet wired. The capture proves
        // the mechanism works; full replay needs hipGraphExecKernelNodeSetParams
        // to update the input token address each step.
        // For now, disable replay and use normal execution.
        gpu::gpu_graph_reset();
        gpu::gpu_arena_end();
        graph_state_ = GraphState::Disabled;
        break;
      }

      case GraphState::Disabled:
        break;
    }
#endif

    // Normal execution path (used by Warmup, Disabled, and fallback)
    auto result = context_.call_fn(
        batched,
        cache_.empty() ? nullptr : &cache_,
        state_.has_value() ? &state_.value() : nullptr);
    state_ = result.state;
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
        auto& prep_output = prep_result.as_logits();
        auto token = convert_to_token(prep_output.logits);
        y_ = LMInput::Text(token);
        mx::async_eval(y_.tokens);

        // Capture hidden state from prefill for MTP.
        if (prep_output.state.has_value()) {
            state_ = prep_output.state;
        }
    }

    // Capture trunk hidden state at last prompt position for first MTP step.
    if (use_mtp_ && state_.has_value() && state_->hidden_intermediates.has_value()) {
        auto trunk_h = state_->hidden_intermediates.value();  // [B, T, H]
        int last_pos = trunk_h.shape(1) - 1;
        auto h_slice = mx::slice(trunk_h, {0, last_pos, 0},
                                 {1, last_pos + 1, trunk_h.shape(2)});  // [1, 1, H]
        mx::eval(h_slice);
        mtp_trunk_hidden_ = h_slice;
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
    , use_mtp_(params.use_mtp && context.get_mtp_head_fn != nullptr)
    , n_draft_tokens_(params.n_draft_tokens)
    , accept_history_(kAcceptHistorySize, 1)  // Initialize with 1 (accepted)
{
    // When MTP is active, initialize state_ so step() always requests
    // hidden_intermediates from the model. The MTP head needs the trunk's
    // hidden state as input.
    if (use_mtp_) {
        mtp_caches_ = context.new_mtp_cache_fn(params);
        state_ = LMOutput::State();  // Empty state signals model to return hidden
    }
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
    , use_mtp_(false)  // MTP not supported with explicit cache
    , accept_history_(kAcceptHistorySize, 1)
{
    prompt_prefill_time_ = measure([&]() {
        prepare(input, prefill_step_size);
    });
    generation_start_ = std::chrono::steady_clock::now();
}

// ---------------------------------------------------------------------------
// TokenIterator — MTP speculative decoding
// ---------------------------------------------------------------------------

std::vector<int> TokenIterator::mtp_speculative_step() {
    // Fallback if MTP not available.
    if (!use_mtp_ || context_.get_mtp_head_fn == nullptr || context_.embed_fn == nullptr
        || context_.apply_lm_head_fn == nullptr) {
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }

    MTPHead* mtp_head = static_cast<MTPHead*>(context_.get_mtp_head_fn());
    int n_draft = current_draft_count();
    int trunk_cache_pos = static_cast<int>(cache_.empty() ? 0 : cache_[0].get_position());

    // Embed the current token for MTP input.
    auto token_embed = context_.embed_fn(y_.tokens);

    // Draft phase: run MTP head n_draft times.
    // MTP head requires (trunk_hidden, token_embed) — the trunk's post-norm
    // hidden state as first input, not just the token embedding.
    // Use cached trunk hidden if available (from previous step), otherwise
    // fall back to token_embed (first step after prefill).
    auto hidden = mtp_trunk_hidden_.has_value()
        ? mtp_trunk_hidden_.value()
        : token_embed;
    std::vector<int> draft_tokens;
    draft_tokens.reserve(n_draft);

    for (int i = 0; i < n_draft; ++i) {
        // Apply output norm to get hidden state ready for lm_head.
        auto norm_h = mtp_head->apply_output_norm(hidden);

        // Apply trunk's lm_head to get draft logits.
        auto logits = context_.apply_lm_head_fn(norm_h);

        // Sample draft token (argmax for deterministic matching).
        int draft_tok = mx::argmax(logits, -1).item<int32_t>();
        draft_tokens.push_back(draft_tok);

        // Embed draft token for next MTP step.
        auto draft_arr = mx::array({draft_tok}, {1, 1}, mx::int32);
        token_embed = context_.embed_fn(draft_arr);

        // Run MTP head forward: hidden = mtp(hidden, embed(draft), mask, cache)
        // For T=1 decode, use no mask.
        hidden = (*mtp_head)(hidden, token_embed, AttentionMask{},
                            mtp_caches_.empty() ? nullptr : &mtp_caches_[0]);

        mx::clear_cache();
    }

    // Trunk verification: run trunk model on draft tokens.
    if (draft_tokens.empty()) {
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }

    // Build draft token sequence for trunk verification.
    std::vector<int32_t> draft_seq;
    draft_seq.reserve(draft_tokens.size());
    for (int t : draft_tokens) draft_seq.push_back(static_cast<int32_t>(t));
    auto draft_arr = mx::array(draft_seq.data(), {1, static_cast<int>(draft_seq.size())}, mx::int32);
    LMInput::Text draft_text(draft_arr);

    // Run trunk model forward on all draft tokens.
    // Request hidden intermediates so state_ is properly updated.
    auto state = LMOutput::State(std::nullopt, std::optional<mx::array>(mx::array(0.0f)));
    auto result = context_.call_fn(draft_text, cache_.empty() ? nullptr : &cache_, &state);
    state_ = result.state;
    maybe_quantize_kv_cache(cache_, kv_bits_, kv_group_size_, quantized_kv_start_);

    // Compare trunk argmax vs draft tokens.
    auto logits = result.logits;  // [1, n_draft, vocab]
    int accepted = 0;

    if (n_draft == 1) {
        // Single draft: argmax match check.
        auto trunk_token = mx::argmax(logits, -1).item<int32_t>();
        if (trunk_token == draft_tokens[0]) {
            accepted = 1;  // Draft accepted
        }
        // Either way, trunk_token is the output
        y_ = LMInput::Text(mx::array({trunk_token}, {1}, mx::int32));
    } else {
        // Multiple drafts: compare each position.
        for (int i = 0; i < n_draft; ++i) {
            // Slice logits at position i: [1, 1, vocab] -> [vocab]
            auto logit_i = mx::slice(logits, {0, i, 0}, {1, i + 1, static_cast<int>(logits.shape(2))});
            logit_i = mx::squeeze(logit_i, 1);
            auto trunk_token = mx::argmax(logit_i, -1).item<int32_t>();

            if (trunk_token == draft_tokens[i]) {
                accepted++;
            } else {
                // Mismatch — use trunk token instead, stop accepting.
                draft_tokens[i] = trunk_token;
                break;
            }
        }

        // Set y_ to the last accepted/rejected token.
        int last_token = (accepted < n_draft) ? draft_tokens[accepted] : draft_tokens.back();
        y_ = LMInput::Text(mx::array({last_token}, {1}, mx::int32));
    }

    // Roll back KV cache if not all drafts were accepted.
    if (accepted < n_draft && !cache_.empty()) {
        size_t new_pos = static_cast<size_t>(trunk_cache_pos + accepted);
        for (auto& c : cache_) {
            c.set_position(new_pos);
        }
    }

    // Capture trunk hidden state at last-used position for next MTP step.
    // The MTP head needs the trunk's post-norm hidden state as input.
    if (result.state.has_value() && result.state->hidden_intermediates.has_value()) {
        auto trunk_h = result.state->hidden_intermediates.value();  // [1, n_draft, H]
        int last_pos = std::min(accepted, n_draft - 1);
        // Slice at position last_pos -> [1, 1, H]. Keep 3D shape for MTPDecoderLayer.
        auto h_slice = mx::slice(trunk_h, {0, last_pos, 0}, {1, last_pos + 1, trunk_h.shape(2)});
        mx::eval(h_slice);
        mtp_trunk_hidden_ = h_slice;
    }

    // Record acceptance for adaptive draft length.
    record_acceptance(n_draft, accepted);

    // Update MTP metrics counters.
    mtp_speculative_steps_++;
    mtp_draft_proposed_ += n_draft;
    mtp_draft_accepted_ += accepted;

    // Return first accepted token; remaining accepted drafts are buffered.
    draft_buffer_.clear();
    for (size_t i = 1; i < draft_tokens.size() && static_cast<int>(i) <= accepted; ++i) {
        draft_buffer_.push_back(draft_tokens[i]);
    }
    draft_buffer_idx_ = 0;

    return {draft_tokens[0]};
}

void TokenIterator::record_acceptance(int proposed, int accepted) {
    uint8_t val = static_cast<uint8_t>(accepted);
    accept_history_[accept_history_idx_ % kAcceptHistorySize] = val;
    accept_history_idx_++;
}

int TokenIterator::current_draft_count() const {
    if (accept_history_idx_ == 0) return n_draft_tokens_;  // No history yet.

    int sum = 0;
    for (int i = 0; i < kAcceptHistorySize; ++i) {
        sum += accept_history_[i];
    }
    float accept_rate = static_cast<float>(sum) / (kAcceptHistorySize * n_draft_tokens_);

    // Adaptive: scale draft count by acceptance rate.
    int adapted = std::max(1, static_cast<int>(n_draft_tokens_ * accept_rate));
    return std::min(adapted, n_draft_tokens_);
}

// ---------------------------------------------------------------------------
// TokenIterator — next()
// ---------------------------------------------------------------------------

std::optional<int> TokenIterator::next() {
    // Check max_tokens limit.
    if (max_tokens_.has_value() && token_count_ >= max_tokens_.value()) {
        return std::nullopt;
    }

    // MTP path: drain buffer first, then run speculative step.
    if (use_mtp_) {
        if (!draft_buffer_.empty() && draft_buffer_idx_ < draft_buffer_.size()) {
            // Return a buffered draft token.
            int tok = draft_buffer_[draft_buffer_idx_++];
            y_ = LMInput::Text(mx::array({tok}, {1}, mx::int32));
            token_count_++;
            mx::eval(y_.tokens);
            return tok;
        }

        // Buffer exhausted — run new MTP speculative step.
        draft_buffer_.clear();
        draft_buffer_idx_ = 0;
        auto accepted = mtp_speculative_step();
        token_count_++;
        mx::eval(y_.tokens);
        return accepted.empty() ? std::nullopt : std::optional<int>(accepted[0]);
    }

    // Standard path: single token generation.
    auto previous_y = y_;
    auto token = step(previous_y);
    y_ = LMInput::Text(token);
    mx::async_eval(token);
    token_count_++;
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
        gen_time,
        mtp_draft_proposed_,
        mtp_draft_accepted_,
        mtp_speculative_steps_
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

    // Use the iterator's completion_info to get accurate MTP metrics.
    auto info = iter.completion_info(prompt_token_count);

    // Override timing with our measured values (iter's clock starts after prefill).
    auto now = std::chrono::steady_clock::now();
    double gen_time = std::chrono::duration<double>(now - start).count();
    info.generation_time = gen_time;
    info.prompt_time += iter.prompt_prefill_time();

    return info;
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
