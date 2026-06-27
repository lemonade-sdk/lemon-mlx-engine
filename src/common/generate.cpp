// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>

#if defined(MLX_BUILD_ROCM)
// Decode-mode toggle (defined in mlx/backend/rocm/eval.cpp; declared here to
// avoid pulling HIP headers into engine code).
#include <mlx-lm/common/graph_decode.h>
namespace mlx::core {
void gpu_set_graph_decode_mode(bool v);
// Build-once pure-relaunch decode + deterministic arena (rocm backend bridge).
void decode_pure_record(int slot);
void decode_pure_replay(int slot);
void decode_pure_off();
size_t decode_pure_chain_len(int slot);
bool decode_arena_begin(size_t capacity, int device, void* stream);
void decode_arena_reset();
void decode_arena_freeze_floor();
void decode_arena_reset_to_floor();
void decode_arena_end();
bool decode_arena_overflowed();
long decode_inline_launch_count();
// Full decode-step stream capture (build-once / replay).
bool decode_capture_begin();
bool decode_capture_end_record(int slot);
bool decode_capture_replay(int slot);
void decode_capture_destroy();
} // namespace mlx::core
#endif

namespace mlx_lm {

namespace mx = mlx::core;

// Dedicated generation stream (thread-local).

mx::Stream& generation_stream() {
#ifdef __APPLE__
    static thread_local mx::Stream s = mx::new_stream(mx::default_device());
    return s;
#else
    if (std::getenv("MLX_GEN_OWN_STREAM")) {
        static thread_local mx::Stream s = mx::new_stream(mx::default_device());
        return s;
    }
    static thread_local mx::Stream s = mx::default_stream(mx::default_device());
    return s;
#endif
}

// RAII guard to set/restore the default stream for a scope.
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
    // top-p filtering disabled; falls back to temperature-scaled categorical.
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
    // Compiled temperature-scaled categorical sampling.
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

    // Index array of the tokens in the repetition window.
    std::vector<uint32_t> idx_vec;
    idx_vec.reserve(tokens_.size());
    for (int tok : tokens_) {
        idx_vec.push_back(static_cast<uint32_t>(tok));
    }
    int n_indices = static_cast<int>(idx_vec.size());
    auto indices = mx::array(idx_vec.data(), {n_indices}, mx::uint32);

    // Gather the logit values at those token indices along the last axis.
    auto shaped_indices = indices;
    if (logits.ndim() == 2) {
        shaped_indices = mx::reshape(indices, {1, n_indices});
    }
    auto selected_logits = mx::take_along_axis(logits, shaped_indices, -1);

    // logit < 0 -> multiply by penalty, else divide by penalty.
    auto zero = mx::array(0.0f);
    auto penalized = mx::where(
        mx::less(selected_logits, zero),
        mx::multiply(selected_logits, mx::array(penalty_)),
        mx::divide(selected_logits, mx::array(penalty_)));

    // Scatter the penalized values back into the logits.
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

    // Incomplete unicode character: new text ends with U+FFFD (EF BF BD).
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
    // Ensure tokens are always 2D [1, seq_len].
    return LMInput::Text(
        mx::reshape(text.tokens, {1, -1}),
        text.mask
    );
}

// ---------------------------------------------------------------------------
// TokenIterator — convert_to_token
// ---------------------------------------------------------------------------

mx::array TokenIterator::convert_to_token(const mx::array& logits) {
    // Extract the last token's logits.
    mx::array last_logits = logits;

    if (logits.ndim() == 3) {
        int seq_len = logits.shape(1);
        last_logits = mx::slice(logits, {0, seq_len - 1, 0},
                                {logits.shape(0), seq_len, logits.shape(2)});
        last_logits = mx::squeeze(last_logits, 1);
    }

    if (processor_.has_value()) {
        last_logits = processor_->process(last_logits);
    }

    auto y = sampler_.sample(last_logits);

    if (processor_.has_value()) {
        processor_->did_sample(y);
    }

    return y;
}

// ---------------------------------------------------------------------------
// TokenIterator — step
// ---------------------------------------------------------------------------

mx::array TokenIterator::step(const LMInput::Text& previous) {
    StreamGuard sg(generation_stream());

    auto batched = add_batch_dim(previous);


    // Normal execution path (used by Warmup, Disabled, and fallback).
    // Decode-mode (single-token forward) tells the ROCm backend to keep the whole
    // forward in one graph and refresh it via ExecUpdate (one launch/token). Stays
    // set through the lazy token eval that happens after this returns.
#if defined(MLX_BUILD_ROCM)
    {
        int Lstep = batched.tokens.shape(batched.tokens.ndim() - 1);
        mlx::core::gpu_set_graph_decode_mode(Lstep == 1);
    }
#endif
    auto result = context_.call_fn(
        batched,
        cache_.empty() ? nullptr : &cache_,
        state_.has_value() ? &state_.value() : nullptr);
    state_ = result.state;
    maybe_quantize_kv_cache(cache_, kv_bits_, kv_group_size_, quantized_kv_start_);

    return convert_to_token(result.logits);
}

#if defined(MLX_BUILD_ROCM)
// Build-once pure-relaunch decode step. Captures the whole forward into a HIP
// graph once, then relaunches the cached exec every token. State machine:
//   0 warmup -> 1 record -> 2 replay   (9 = disabled: arena overflow / capture fail)
// Everything that varies per token lives in FIXED-address buffers so the
// recorded exec's baked pointers stay valid across relaunches: position and input
// token are device buffers injected each step; the GDN recurrent state is updated
// IN PLACE in its cache slots [0]/[1] (the fused kernels alias state-out to
// state-in); KV is written in place at the device position. No scratch, no copy.
mx::array TokenIterator::step_pure_graph(const LMInput::Text& previous) {
    StreamGuard sg(generation_stream());
    namespace mc = mlx::core;

    static const size_t arena_bytes = [] {
        const char* e = std::getenv("MLX_DECODE_ARENA_MB");
        return size_t(e ? std::atoll(e) : 1024) << 20;
    }();
    static const bool noreplay = std::getenv("MLX_PURE_NOREPLAY") != nullptr;

    LMInput::Text in(mlx_lm::graph_decode_input());  // [1,1] int32, fixed addr

    // Feed input + advance position via IMMEDIATE launches (loop-owned, between
    // relaunches) — never recorded graph nodes.
    mc::gpu_set_graph_decode_mode(false);
    mx::array prev_tok = previous.tokens;
    mlx_lm::set_graph_decode_input_from(prev_tok);  // device copy -> fixed buffer
    if (pure_graph_state_ == 0) {
        mlx_lm::set_graph_external_pos(true);
        int off = 0;
        for (auto& c : cache_) off = std::max(off, c.offset());
        mlx_lm::set_graph_decode_pos(off);
        pure_pos_ = off;
        for (auto& c : cache_) c.reserve_to(pure_graph_cap_);
    } else {
        mlx_lm::advance_graph_decode_pos(1);
        pure_pos_ += 1;
    }

    // GDN recurrent state is updated IN PLACE in cache slots [0]/[1] by the fused
    // kernels (state output aliases state input), and KV is written in place at
    // the device position — so there is no scratch slot to copy back between
    // relaunches. One recorded exec suffices: record once (state 1), replay (2).
    const int replay_state = 2;

    auto disable = [&]() {
        mc::decode_capture_destroy();
        mc::decode_arena_end();
        mlx_lm::set_graph_external_pos(false);
        pure_graph_state_ = 9;
    };

    mx::array token = mx::array(0);

    if (!noreplay && pure_graph_state_ == replay_state && pure_logits_.has_value()) {
        // REPLAY: input/pos already set above. Relaunch the recorded exec, then
        // read the freshly-overwritten logits buffer (convert_to_token's sample
        // kernel reads it at launch time).
        mc::decode_arena_reset_to_floor();   // keep recorded buffers; sample above
        if (mc::decode_capture_replay(0)) {
            token = convert_to_token(*pure_logits_);
        } else {
            disable();  // capture lost -> rebuild via the eager fallback below
        }
    }

    const bool is_record =
        !noreplay && pure_graph_state_ >= 1 && pure_graph_state_ < replay_state;
    if (pure_graph_state_ != replay_state || pure_graph_state_ == 9) {
        // WARMUP (0), RECORD (1..replay_state-1), or fallback: run via call_fn.
        if (is_record) {
            if (pure_graph_state_ == 1)
                mc::decode_arena_begin(arena_bytes, 0, nullptr);
            mc::decode_arena_reset();      // record forward allocates from base
            mc::decode_capture_begin();    // capture the eager call_fn that follows
        }
        auto result = context_.call_fn(
            in, cache_.empty() ? nullptr : &cache_,
            state_.has_value() ? &state_.value() : nullptr);
        state_ = result.state;

        if (is_record) {
            // Launch the forward INLINE (async_eval: no blocking sync, which is
            // illegal mid-capture) so every kernel records into the capture. The
            // in-place GDN state slots [0]/[1] are eval'd so their writing kernels
            // are captured.
            std::vector<mx::array> outs{result.logits};
            for (auto& c : cache_) {
                auto* m = c.as_mamba();
                if (!m) continue;
                if ((*m)[0].has_value()) outs.push_back((*m)[0].value());
                if ((*m)[1].has_value()) outs.push_back((*m)[1].value());
            }
            mx::async_eval(outs);
            if (mc::decode_capture_end_record(0)) {
                pure_logits_ = result.logits;  // buffer overwritten by each replay
                // The captured forward's allocations occupy [0, floor); freeze it
                // so replay sampling allocates above the recorded buffers.
                mc::decode_arena_freeze_floor();
            } else {
                disable();
            }
        }
        token = convert_to_token(result.logits);
        // Force-eval token + in-place GDN state (the next relaunch reads them).
        std::vector<mx::array> ev{token};
        for (auto& c : cache_) {
            auto* m = c.as_mamba();
            if (!m) continue;
            if ((*m)[0].has_value()) ev.push_back((*m)[0].value());
            if ((*m)[1].has_value()) ev.push_back((*m)[1].value());
        }
        mx::eval(ev);
    }

    static const bool pure_dbg = std::getenv("MLX_PURE_DEBUG") != nullptr;
    if (pure_dbg) {
        static long prev_inline = 0;
        long now_inline = mc::decode_inline_launch_count();
        fprintf(stderr, "[pure] state=%d pos=%d in=%d sampled=%d inline=%ld(+%ld)\n",
                pure_graph_state_, pure_pos_,
                mlx_lm::graph_decode_input().item<int>(), token.item<int>(),
                now_inline, now_inline - prev_inline);
        prev_inline = now_inline;
    }

    if (pure_graph_state_ == 0) {
        pure_graph_state_ = 1;                       // next token records
    } else if (pure_graph_state_ >= 1 && pure_graph_state_ < replay_state) {
        if (mc::decode_arena_overflowed()) disable();
        else pure_graph_state_ += 1;                 // recorded -> replay
    }
    return token;
}
#endif

// ---------------------------------------------------------------------------
// TokenIterator — prepare (prompt prefill)
// ---------------------------------------------------------------------------

void TokenIterator::prepare(const LMInput& input, int window_size) {
    StreamGuard sg(generation_stream());
#if defined(MLX_BUILD_ROCM)
    // Prefill: large multi-token intermediates — keep the per-graph caps active
    // (decode-mode off) so peak graph memory stays bounded.
    mlx::core::gpu_set_graph_decode_mode(false);
#endif

    if (processor_.has_value()) {
        processor_->prompt(input.text.tokens);
    }

    auto prep_result = context_.prepare_fn(input, cache_, window_size);

    if (prep_result.is_tokens()) {
        // Model returned remaining tokens — prime the cache.
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
    // When MTP is active, request hidden_intermediates from the model.
    if (use_mtp_) {
        mtp_caches_ = context.new_mtp_cache_fn(params);
        state_ = LMOutput::State();  // Empty state signals model to return hidden
    }
    prompt_token_count_ = static_cast<int>(input.text.tokens.size());
    prompt_prefill_time_ = measure([&]() {
        prepare(input, params.prefill_step_size);
    });
    prefill_host_time_ = prompt_prefill_time_;
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
    prompt_token_count_ = static_cast<int>(input.text.tokens.size());
    prompt_prefill_time_ = measure([&]() {
        prepare(input, prefill_step_size);
    });
    prefill_host_time_ = prompt_prefill_time_;
    generation_start_ = std::chrono::steady_clock::now();
}

// External-cache constructor with parameters — MTP over a reused KV cache.
TokenIterator::TokenIterator(
    ModelContext& context,
    const LMInput& input,
    std::vector<KVCache> cache,
    const GenerateParameters& params)
    : context_(context)
    , y_(mx::array(0, mx::int32))  // placeholder, overwritten by prepare()
    , cache_(std::move(cache))
    , sampler_(AnySampler::from_params(params))
    , processor_(AnyProcessor::from_params(params))
    , max_tokens_(params.max_tokens)
    , kv_bits_(params.kv_bits)
    , kv_group_size_(params.kv_group_size)
    , quantized_kv_start_(params.quantized_kv_start)
    , use_mtp_(params.use_mtp && context.get_mtp_head_fn != nullptr
               && context.new_mtp_cache_fn != nullptr)
    , n_draft_tokens_(params.n_draft_tokens)
    , accept_history_(kAcceptHistorySize, 1)
{
    if (use_mtp_) {
        mtp_caches_ = context.new_mtp_cache_fn(params);
        state_ = LMOutput::State();  // Empty state signals model to return hidden
    }
    prompt_token_count_ = static_cast<int>(input.text.tokens.size());
    prompt_prefill_time_ = measure([&]() {
        prepare(input, params.prefill_step_size);
    });
    prefill_host_time_ = prompt_prefill_time_;
    generation_start_ = std::chrono::steady_clock::now();
}

// ---------------------------------------------------------------------------
// TokenIterator — MTP speculative decoding
// ---------------------------------------------------------------------------

std::vector<int> TokenIterator::mtp_speculative_step() {
    // Fallback to plain decode if MTP is not available on this context.
    if (!use_mtp_ || context_.get_mtp_head_fn == nullptr || context_.embed_fn == nullptr
        || context_.apply_lm_head_fn == nullptr) {
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }

    MTPHead* mtp_head = static_cast<MTPHead*>(context_.get_mtp_head_fn());
    // Null when the model carries no MTP head weights — fall back to plain decode.
    if (mtp_head == nullptr) {
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }
    int n_draft = current_draft_count();
    static const bool kMtpTiming = (std::getenv("MTP_TIMING") != nullptr);
    auto t_start = std::chrono::steady_clock::now();
    auto t_draft = t_start, t_verify = t_start;
    // Read the trunk's position from a full-attention (non-Mamba) cache.
    int trunk_cache_pos = 0;
    for (auto& c : cache_) {
        if (!c.as_mamba()) {
            trunk_cache_pos = static_cast<int>(c.get_position());
            break;
        }
    }

    // Reset MTP head cache to drop stale KV from prior speculative steps.
    for (auto& c : mtp_caches_) {
        c.set_position(0);
    }

    // Draft phase. d0 is the trunk's already-computed next token (y_), trusted
    // and never verified; the head drafts d1..d_{K-1}.
    auto hidden = mtp_trunk_hidden_.has_value()
        ? mtp_trunk_hidden_.value()
        : context_.embed_fn(y_.tokens);
    // Ensure hidden is always 3D [1, 1, H].
    if (hidden.ndim() == 2) {
        hidden = mx::reshape(hidden, {1, 1, hidden.shape(-1)});
    }

    std::vector<int> draft_tokens;
    draft_tokens.reserve(n_draft);

    // Keep the draft recurrence on-device; sync once after the chain.
    auto prev_tok_arr = mx::reshape(y_.tokens, {1, 1});  // [1,1] int32, d0
    std::vector<mx::array> draft_tok_arrs;               // d1..d_{n-1}, on-device
    draft_tok_arrs.reserve(n_draft > 1 ? n_draft - 1 : 0);

    for (int i = 1; i < n_draft; ++i) {
        // Advance the hidden state with the previous token, then predict d_i.
        auto prev_embed = context_.embed_fn(prev_tok_arr);
        hidden = (*mtp_head)(hidden, prev_embed, AttentionMask{},
                            mtp_caches_.empty() ? nullptr : &mtp_caches_[0]);

        auto norm_h = mtp_head->apply_output_norm(hidden);
        auto logits = context_.apply_lm_head_fn(norm_h);
        prev_tok_arr = mx::reshape(
            mx::argmax(logits, -1, /*keepdims=*/false), {1, 1});
        prev_tok_arr = mx::astype(prev_tok_arr, mx::int32);
        draft_tok_arrs.push_back(prev_tok_arr);
    }

    // Single sync: concatenate d1..d_{n-1}, eval once with d0, read ints together.
    if (!draft_tok_arrs.empty()) {
        auto drafts_dev = mx::reshape(
            mx::concatenate(draft_tok_arrs, /*axis=*/0),
            {static_cast<int>(draft_tok_arrs.size())});
        mx::eval(y_.tokens, drafts_dev);
        draft_tokens.push_back(y_.tokens.item<int32_t>());  // d0
        for (size_t i = 0; i < draft_tok_arrs.size(); ++i) {
            draft_tokens.push_back(drafts_dev.data<int32_t>()[i]);
        }
    } else {
        mx::eval(y_.tokens);
        draft_tokens.push_back(y_.tokens.item<int32_t>());  // d0 only
    }
    if (kMtpTiming) t_draft = std::chrono::steady_clock::now();

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

    // Enable per-token recurrent-state capture on the linear (Mamba) layers for
    // prefix rollback; snapshot is a safety fallback.
    struct SavedMambaState {
        MambaCache::Snapshot snapshot;
        bool has_mamba = false;
    };
    // MTP_NO_INTERMEDIATES=1 forces the legacy restore+re-run rollback.
    static const bool kUseIntermediates = (std::getenv("MTP_NO_INTERMEDIATES") == nullptr);
    std::vector<SavedMambaState> saved_mamba;
    saved_mamba.reserve(cache_.size());
    bool any_mamba = false;
    for (auto& c : cache_) {
        if (auto* m = c.as_mamba()) {
            SavedMambaState s;
            s.snapshot = m->snapshot();
            s.has_mamba = true;
            saved_mamba.push_back(s);
            if (kUseIntermediates) m->set_capture_spec(true);
            any_mamba = true;
        } else {
            saved_mamba.push_back({});
        }
    }

    // Run trunk model forward on all draft tokens, requesting hidden intermediates.
    auto state = LMOutput::State(std::nullopt, std::optional<mx::array>(mx::array(0.0f)));
    auto result = context_.call_fn(draft_text, cache_.empty() ? nullptr : &cache_, &state);
    state_ = result.state;
    maybe_quantize_kv_cache(cache_, kv_bits_, kv_group_size_, quantized_kv_start_);

    // Compare trunk argmax vs draft tokens: logit[i] predicts draft[i+1].
    // Take argmax over [1, n_draft, vocab] in one op, then scan on host ints.
    auto logits = result.logits;
    auto trunk_argmax = mx::astype(mx::argmax(logits, -1), mx::int32);  // [1, n_draft]
    mx::eval(trunk_argmax);
    const int32_t* trunk_pred = trunk_argmax.data<int32_t>();

    int accepted = 0;
    for (int i = 0; i < n_draft - 1; ++i) {
        int32_t trunk_token = trunk_pred[i];
        if (trunk_token == draft_tokens[i + 1]) {
            accepted++;
        } else {
            // Mismatch — replace with trunk token and stop accepting.
            draft_tokens[i + 1] = trunk_token;
            break;
        }
    }

    // Set y_ to the following token (bonus on full accept, else trunk correction).
    if (accepted == n_draft - 1) {
        int32_t bonus_token = trunk_pred[n_draft - 1];
        y_ = LMInput::Text(mx::array({bonus_token}, {1}, mx::int32));
    } else {
        y_ = LMInput::Text(mx::array({draft_tokens[accepted + 1]}, {1}, mx::int32));
    }
    if (kMtpTiming) { mx::eval(y_.tokens); t_verify = std::chrono::steady_clock::now(); }

    // Commit the accepted prefix without re-running the trunk: trim the caches.

    // Capture the next-step trunk hidden at the position that predicts y_.
    auto capture_hidden_at = [&](int pos) {
        if (result.state.has_value() && result.state->hidden_intermediates.has_value()) {
            auto trunk_h = result.state->hidden_intermediates.value();
            int p = std::min(pos, static_cast<int>(trunk_h.shape(1)) - 1);
            if (p < 0) p = 0;
            auto h_slice = mx::slice(trunk_h, {0, p, 0}, {1, p + 1, trunk_h.shape(2)});
            mx::eval(h_slice);
            mtp_trunk_hidden_ = h_slice;
        }
    };

    // Did every linear layer capture its per-token states this step?
    bool have_spec = any_mamba;
    if (any_mamba) {
        for (auto& c : cache_) {
            if (auto* m = c.as_mamba()) {
                if (!m->has_spec()) { have_spec = false; break; }
            }
        }
    }

    if (accepted == n_draft - 1) {
        // All drafts accepted — caches already hold exactly [d0..d_{n-1}].
        capture_hidden_at(n_draft - 1);
    } else if (any_mamba && !have_spec) {
        // Safety fallback: restore recurrent state and re-run on accepted prefix.
        for (size_t i = 0; i < cache_.size(); ++i) {
            if (saved_mamba[i].has_mamba) {
                auto* m = cache_[i].as_mamba();
                if (m) m->restore(saved_mamba[i].snapshot);
            }
        }
        for (auto& c : cache_) c.set_position(trunk_cache_pos);
        std::vector<int32_t> rerun_seq;
        rerun_seq.reserve(1 + accepted);
        for (int i = 0; i <= accepted; ++i) {
            rerun_seq.push_back(static_cast<int32_t>(draft_tokens[i]));
        }
        auto rerun_arr = mx::array(rerun_seq.data(), {1, static_cast<int>(rerun_seq.size())}, mx::int32);
        LMInput::Text rerun_text(rerun_arr);
        auto rerun_state = LMOutput::State(std::nullopt, std::optional<mx::array>(mx::array(0.0f)));
        result = context_.call_fn(rerun_text, &cache_, &rerun_state);
        state_ = result.state;
        maybe_quantize_kv_cache(cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
        logits = result.logits;
        capture_hidden_at(accepted);
    } else if (accepted < n_draft - 1 && !cache_.empty()) {
        // Fast path: trim caches to [d0..d_accepted].
        capture_hidden_at(accepted);
        int keep_pos = trunk_cache_pos + accepted + 1;
        for (auto& c : cache_) {
            if (auto* m = c.as_mamba()) {
                m->rollback_spec(accepted + 1);
            } else {
                c.set_position(keep_pos);
            }
        }
    } else {
        capture_hidden_at(accepted);
    }

    // Clear the capture flag and any leftover per-token states for next step.
    for (auto& c : cache_) {
        if (auto* m = c.as_mamba()) m->set_capture_spec(false);
    }

    // Record acceptance for adaptive draft length.
    record_acceptance(n_draft, accepted);

    // Update MTP metrics counters (d0 is not counted as a draft).
    mtp_speculative_steps_++;
    mtp_draft_proposed_ += (n_draft > 1 ? n_draft - 1 : 1);
    mtp_draft_accepted_ += accepted;

    static const bool kMtpDebug = (std::getenv("MTP_DEBUG") != nullptr);
    if (kMtpDebug) {
        std::fprintf(stderr, "[mtp] step=%d n_draft=%d accepted=%d drafts=[",
                     mtp_speculative_steps_, n_draft, accepted);
        for (size_t i = 0; i < draft_tokens.size(); ++i)
            std::fprintf(stderr, "%d%s", draft_tokens[i],
                         i + 1 < draft_tokens.size() ? "," : "");
        std::fprintf(stderr, "]\n");
    }
    if (kMtpTiming) {
        auto t_end = std::chrono::steady_clock::now();
        auto us = [](auto a, auto b) {
            return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        std::fprintf(stderr,
            "[mtp-t] step=%d n_draft=%d accepted=%d draft=%ldus verify=%ldus commit=%ldus total=%ldus\n",
            mtp_speculative_steps_, n_draft, accepted,
            us(t_start, t_draft), us(t_draft, t_verify), us(t_verify, t_end),
            us(t_start, t_end));
    }

    // Emit the accepted prefix [d0..d_accepted]: d0 now, d1..d_accepted buffered.
    // y_ holds the following token, emitted as d0 of the next step (not here).
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
    return n_draft_tokens_;
}

// ---------------------------------------------------------------------------
// TokenIterator — next()
// ---------------------------------------------------------------------------

void TokenIterator::measure_prefill_boundary_() {
    if (prefill_measured_) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    prompt_prefill_time_ += std::chrono::duration<double>(now - generation_start_).count();
    generation_start_ = now;
    prefill_measured_ = true;
    if (std::getenv("MLX_PROFILE_PREFILL")) {
        double gpu = prompt_prefill_time_ - prefill_host_time_;
        std::cerr << "[prefill] prompt_tokens=" << prompt_token_count_
                  << " host_build=" << prefill_host_time_ << "s gpu_exec=" << gpu
                  << "s total=" << prompt_prefill_time_ << "s pp/s="
                  << (prompt_prefill_time_ > 0.0
                          ? prompt_token_count_ / prompt_prefill_time_ : 0.0)
                  << std::endl;
    }
}

std::optional<int> TokenIterator::next() {
    if (max_tokens_.has_value() && token_count_ >= max_tokens_.value()) {
        return std::nullopt;
    }

    // MTP path: drain buffer first, then run speculative step.
    if (use_mtp_) {
        if (!draft_buffer_.empty() && draft_buffer_idx_ < draft_buffer_.size()) {
            // Return a buffered accepted token; do NOT touch y_.
            int tok = draft_buffer_[draft_buffer_idx_++];
            token_count_++;
            return tok;
        }

        // Buffer exhausted — run new MTP speculative step.
        draft_buffer_.clear();
        draft_buffer_idx_ = 0;
        auto accepted = mtp_speculative_step();
        token_count_++;
        mx::eval(y_.tokens);
        measure_prefill_boundary_();
        return accepted.empty() ? std::nullopt : std::optional<int>(accepted[0]);
    }

    // Standard path: single token generation.
    static const bool g_sync_decode = std::getenv("MLX_SYNC_DECODE") != nullptr;

#if defined(MLX_BUILD_ROCM)
    // Build-once pure-relaunch graph decode (opt-in, qwen35-moe device-pos path).
    static const bool pure_enabled =
        std::getenv("MLX_DECODE_GRAPH_PURE") != nullptr;
    if (pure_enabled && pure_graph_state_ != 9 && !cache_.empty()) {
        if (pure_graph_cap_ == 0) {
            int off = 0;
            for (auto& c : cache_) off = std::max(off, c.offset());
            int remaining = max_tokens_.has_value()
                ? std::max(0, max_tokens_.value() - token_count_) : 256;
            pure_graph_cap_ = off + remaining + 8;
        }
        auto previous_y = y_;
        auto token = step_pure_graph(previous_y);
        y_ = LMInput::Text(token);
        token_count_++;
        measure_prefill_boundary_();
        // Pure replay is one hipGraphLaunch/token, so there is no per-op host
        // dispatch to overlap — pipelining (return previous) was measured to be
        // a no-op here, so sample this token directly.
        return token.item<int32_t>();
    }
#endif

    auto previous_y = y_;
    auto token = step(previous_y);
    y_ = LMInput::Text(token);
    if (g_sync_decode) {
        // Diagnostic: fully retire each forward before building the next.
        mx::eval(token);
        token_count_++;
        measure_prefill_boundary_();
        return token.item<int32_t>();
    }
    mx::async_eval(token);
    token_count_++;
    mx::eval(previous_y.tokens);
    measure_prefill_boundary_();
    int32_t tid = previous_y.tokens.item<int32_t>();
    return tid;
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
    WiredLimitGuard wired_guard;

    int prompt_token_count = static_cast<int>(input.text.tokens.size());

    TokenIterator iter(context, input, params);

    auto start = std::chrono::steady_clock::now();
    int token_count = 0;

    while (auto maybe_token = iter.next()) {
        int token = *maybe_token;

        // Restart the decode clock after the first token.
        if (token_count == 0) {
            start = std::chrono::steady_clock::now();
        }

        if (eos_token_ids.count(token)) {
            break;
        }

        token_count++;

        // Periodically clear the memory cache to reduce memory pressure.
        if (token_count % 256 == 0) {
            mx::clear_cache();
        }

        if (on_token(token) == GenerateDisposition::stop) {
            break;
        }
    }

    mx::synchronize(generation_stream());

    auto info = iter.completion_info(prompt_token_count);

    // Override timing with our measured values.
    auto now = std::chrono::steady_clock::now();
    double gen_time = std::chrono::duration<double>(now - start).count();
    info.generation_time = gen_time;

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
