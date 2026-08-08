// Copyright © 2024-2025 Apple Inc. — Ported to C++

#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/redline_decode_session.h>
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <iostream>
#include <stdexcept>

#if defined(MLX_BUILD_ROCM)
// Decode-mode toggle (defined in mlx/backend/rocm/eval.cpp; declared here to
// avoid pulling HIP headers into engine code).
#include <mlx-lm/common/graph_decode.h>
namespace mlx::core {
void gpu_set_graph_decode_mode(bool v);
// Deterministic arena (rocm backend bridge).
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

// mlx registers CPU CommandEncoders in thread_local maps at stream creation
// (mlx/backend/cpu/eval.cpp). Streams created on the main/load thread are
// invisible to httplib worker threads — eval then throws
// "There is no Stream(cpu, 0) in current thread" (PR #63 P0-MTP / M1 server).
// Re-bind known CPU streams into this thread's encoder map (try_emplace).
namespace mlx::core::cpu {
void new_stream(Stream s);
} // namespace mlx::core::cpu

namespace mlx_lm {

namespace mx = mlx::core;

static void ensure_thread_cpu_stream_encoders() {
    static thread_local size_t last_n = 0;
    auto streams = mx::get_streams();
    if (streams.size() == last_n && last_n > 0) {
        return;
    }
    for (const auto& s : streams) {
        if (s.device.type == mx::Device::cpu) {
            mlx::core::cpu::new_stream(s);
        }
    }
    // Ensure this thread also has a default CPU stream for future host ops.
    (void)mx::default_stream(mx::Device::cpu);
    last_n = mx::get_streams().size();
}

#if defined(MLX_BUILD_ROCM)
// ---------------------------------------------------------------------------
// MLX_REDLINE_DECODE (exp/redline-kernel-launch): P0 log + P2 session init.
// Default OFF. Exactly "1" → maybe_log_redline_session_status (dlopen smoke).
// XOR with MLX_DECODE_GRAPH_PURE=1 → fail-closed eager.
// Does NOT set MLX_USE_HIP_GRAPHS / MLX_HIP_GRAPH_DECODE / MLX_DECODE_GRAPH_PURE.
// Does NOT replace product forward (P3+).
// ---------------------------------------------------------------------------
static bool env_exact_one_(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1' && v[1] == '\0';
}

static bool redline_decode_env_enabled_() {
    return env_exact_one_("MLX_REDLINE_DECODE");
}

static bool pure_graph_env_enabled_() {
    return env_exact_one_("MLX_DECODE_GRAPH_PURE");
}
#endif

// MLX_KV_OFFSET_LOG=1: stderr KV max offset every MLX_KV_OFFSET_EVERY (default 64).
static void maybe_log_kv_offset_(std::vector<KVCache>& cache, int token_count) {
    static const bool enabled = [] {
        const char* v = std::getenv("MLX_KV_OFFSET_LOG");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    if (!enabled || cache.empty()) return;

    static const int every = [] {
        const char* e = std::getenv("MLX_KV_OFFSET_EVERY");
        int n = e ? std::atoi(e) : 64;
        return n > 0 ? n : 64;
    }();
    static int prev_max_off = -1;

    int max_off = 0;
    for (auto& c : cache) max_off = std::max(max_off, c.offset());
    const bool stall =
        prev_max_off >= 0 && max_off <= prev_max_off && token_count > 0;
    if (stall || (token_count % every) == 0) {
        fprintf(stderr, "[kv] tok=%d max_offset=%d prev=%d layers=%zu%s\n",
                token_count, max_off, prev_max_off, cache.size(),
                stall ? " STALL" : "");
        fflush(stderr);
    }
    prev_max_off = max_off;
}

// Dedicated generation stream (thread-local).

mx::Stream& generation_stream() {
    // Prefer a thread-owned stream so worker threads (server) and MTP draft
    // never depend on a missing default Stream(cpu, 0) TLS binding.
    // Opt out: MLX_GEN_OWN_STREAM=0 → device default stream (legacy).
#ifdef __APPLE__
    static thread_local mx::Stream s = mx::new_stream(mx::default_device());
    return s;
#else
    static thread_local mx::Stream s = [] {
        const char* v = std::getenv("MLX_GEN_OWN_STREAM");
        // Default ON for non-Apple (ROCm/Linux): own stream. Explicit 0 disables.
        if (v && v[0] == '0' && v[1] == '\0') {
            return mx::default_stream(mx::default_device());
        }
        return mx::new_stream(mx::default_device());
    }();
    return s;
#endif
}

// RAII guard to set/restore the default stream for a scope.
struct StreamGuard {
    mx::Stream old_stream_;
    bool changed_ = false;
    StreamGuard(mx::Stream s) : old_stream_(mx::default_stream(mx::default_device())) {
        // Bind CPU stream encoders onto this worker before any eval.
        ensure_thread_cpu_stream_encoders();
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

bool mtp_uses_greedy_spec(const GenerateParameters& params) {
    // Fast argmax draft/verify when sampling contract is greedy-neutral.
    // temperature==0 → AnySampler is ArgMax (top_p is inert: TopPSampler has
    // nucleus filtering disabled, and temp=0 never constructs TopPSampler).
    // Do not force the slow RS path solely because top_p∈(0,1) at temp=0.
    if (params.temperature != 0.0f) return false;
    if (params.repetition_penalty.has_value() &&
        params.repetition_penalty.value() != 1.0f) {
        return false;
    }
    return true;
}

std::string mtp_greedy_only_violation(const GenerateParameters& params) {
    // Legacy name: empty when greedy-spec path applies; otherwise describes
    // why rejection-sampling MTP will be used (not an error).
    if (mtp_uses_greedy_spec(params)) return {};
    std::ostringstream oss;
    oss << "MTP sampled mode (rejection sampling): temperature="
        << params.temperature << " top_p=" << params.top_p;
    if (params.repetition_penalty.has_value()) {
        oss << " repetition_penalty=" << params.repetition_penalty.value();
    }
    return oss.str();
}

MtpEmitPlan mtp_make_emit_plan(const std::vector<int>& draft_tokens, int accepted) {
    MtpEmitPlan plan;
    if (draft_tokens.empty()) {
        return plan;
    }
    plan.d0 = draft_tokens[0];
    const int max_acc = static_cast<int>(draft_tokens.size()) - 1;
    if (accepted < 0) {
        accepted = 0;
    }
    if (accepted > max_acc) {
        accepted = max_acc;
    }
    plan.buffered.reserve(static_cast<size_t>(accepted));
    for (int a = 0; a < accepted; ++a) {
        plan.buffered.push_back(draft_tokens[static_cast<size_t>(a + 1)]);
    }
    return plan;
}

float mtp_accept_ratio(float log_q, float log_p) {
    // min(1, exp(log_q - log_p)); reject non-finite inputs before min/exp
    // (std::min with NaN is implementation-defined and can yield 1 via exp(0)).
    if (!std::isfinite(log_q) || !std::isfinite(log_p)) {
        return 0.0f;
    }
    const float diff = log_q - log_p;
    if (diff >= 0.0f) {
        return 1.0f;
    }
    float ratio = std::exp(diff);
    if (!std::isfinite(ratio) || ratio < 0.0f) {
        return 0.0f;
    }
    if (ratio > 1.0f) {
        ratio = 1.0f;
    }
    return ratio;
}

int mtp_adaptive_n_draft(
    int n_draft_tokens,
    const uint8_t* accept_history,
    int history_len,
    bool fixed) {
    if (n_draft_tokens <= 2) {
        return n_draft_tokens;
    }
    if (fixed) {
        return n_draft_tokens;
    }
    if (history_len <= 0 || accept_history == nullptr) {
        return n_draft_tokens;
    }
    int sum = 0;
    for (int i = 0; i < history_len; ++i) {
        sum += static_cast<int>(accept_history[i]);
    }
    const float mean_acc =
        static_cast<float>(sum) / static_cast<float>(history_len);
    int want = static_cast<int>(mean_acc + 2.0f);  // d0 + drafts slack
    if (want < 2) {
        want = 2;
    }
    if (want > n_draft_tokens) {
        want = n_draft_tokens;
    }
    return want;
}

static mx::array mtp_last_row_logits(const mx::array& logits) {
    mx::array last = logits;
    if (last.ndim() == 3) {
        int seq_len = last.shape(1);
        last = mx::slice(last, {0, seq_len - 1, 0},
                         {last.shape(0), seq_len, last.shape(2)});
        last = mx::squeeze(last, 1);
    }
    if (last.ndim() == 2 && last.shape(0) == 1) {
        last = mx::squeeze(last, 0);
    }
    return last;  // [V]
}

float TokenIterator::mtp_token_logprob(
    const mx::array& logits, int token, float temperature) {
    mx::array last = mtp_last_row_logits(logits);
    float t = temperature;
    if (t <= 0.0f) t = 1.0f;
    if (t != 1.0f) {
        last = mx::multiply(last, mx::array(1.0f / t));
    }
    auto lp = mx::log(mx::softmax(last, /*axis=*/-1));
    auto idx = mx::array(token, mx::int32);
    auto val = mx::take(lp, idx);
    mx::eval(val);
    return val.item<float>();
}

mx::array TokenIterator::mtp_residual_logits(
    const mx::array& target_logits,
    const mx::array& draft_logits,
    int rejected_token,
    float temperature) {
    // Leviathan residual: r = max(0, q - p). Returns logits for bare
    // categorical(sample) — do NOT pass through sampler_.sample() which would
    // divide by temperature again (R-1 double-scale bug → samples r^(1/t)).
    float t = temperature > 0.0f ? temperature : 1.0f;
    auto q_logits = mtp_last_row_logits(target_logits);
    auto p_logits = mtp_last_row_logits(draft_logits);
    if (t != 1.0f) {
        q_logits = mx::multiply(q_logits, mx::array(1.0f / t));
        p_logits = mx::multiply(p_logits, mx::array(1.0f / t));
    }
    auto q = mx::softmax(q_logits, /*axis=*/-1);
    auto p = mx::softmax(p_logits, /*axis=*/-1);
    auto r = mx::maximum(mx::subtract(q, p), mx::array(0.0f));
    auto mass = mx::sum(r);
    mx::eval(mass);
    float m = mass.item<float>();
    if (!(m > 1e-8f)) {
        // Residual mass collapsed — sample from target with rejected token
        // masked; apply temperature once here (bare categorical after).
        auto masked = mtp_last_row_logits(target_logits);
        if (t != 1.0f) {
            masked = mx::multiply(masked, mx::array(1.0f / t));
        }
        auto idx = mx::arange(0, masked.shape(0), mx::int32);
        auto is_rej = mx::equal(idx, mx::array(rejected_token, mx::int32));
        float ninf = -1.0e9f;
        masked = mx::where(is_rej, mx::array(ninf), masked);
        return mx::reshape(masked, {1, masked.shape(0)});
    }
    // log(r) as categorical logits (distribution is already temperatured via q,p).
    auto log_r = mx::log(mx::add(r, mx::array(1e-10f)));
    return mx::reshape(log_r, {1, log_r.shape(0)});
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
        // P0/P2/P6: opt-in session status + graph_decode ptr bind probe on L=1
        // (no forward path change; not gen t/s).
        if (Lstep == 1) {
            maybe_log_redline_session_status();
            maybe_probe_redline_graph_decode_bind();
        }
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
    // Ensure the previous sample is materialised before the in-place device
    // copy; a lazy token here can leave the fixed input buffer stale and
    // desync the autoregressive chain on the first replay.
    mx::eval(prev_tok);
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
        // sample from the logits buffer the exec just overwrote.
        mc::decode_arena_reset_to_floor();   // keep recorded buffers; sample above
        if (mc::decode_capture_replay(0)) {
            token = convert_to_token(*pure_logits_);
            // CRITICAL: materialise the sample before this function returns.
            // The next relaunch overwrites pure_logits_'s device buffer in place
            // without MLX array-versioning; a lazy sample would race that write.
            mx::eval(token);
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
    // Prefill graph mode (opt-in experiment — default OFF / product-safe):
    //
    // Default: graph_decode_mode=false so mid-forward commit caps bound peak
    // graph memory on multi-token chunks (see mlx CommandEncoder::needs_commit).
    //
    // MLX_PREFILL_ONE_GRAPH=1: force graph_decode_mode=true for prepare chunk
    // evals only — whole multi-token forward becomes ONE HIP graph (no split).
    // Requires mlx use_hip_graphs on for decode-mode, i.e. MLX_HIP_GRAPH_DECODE=1
    // or MLX_USE_HIP_GRAPHS=1 (use_hip_graphs picks decode vs prefill flag by mode).
    // Pair with MLX_GRAPH_PREFILL_REPLAY=1 for ExecUpdate on stable topologies.
    // See docs/experiments/prefill-hip-graph/.
    static const bool prefill_one_graph = [] {
        const char* e = std::getenv("MLX_PREFILL_ONE_GRAPH");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    mlx::core::gpu_set_graph_decode_mode(prefill_one_graph);
    if (std::getenv("MLX_PROFILE_PREFILL") || std::getenv("MLX_HIP_GRAPH_PREFILL")
        || std::getenv("MLX_GRAPH_PREFILL_REPLAY")
        || std::getenv("MLX_PREFILL_ONE_GRAPH")
        || std::getenv("MLX_USE_HIP_GRAPHS")) {
        static bool logged = false;
        if (!logged) {
            const char* hp = std::getenv("MLX_HIP_GRAPH_PREFILL");
            const char* hd = std::getenv("MLX_HIP_GRAPH_DECODE");
            const char* ug = std::getenv("MLX_USE_HIP_GRAPHS");
            const char* pr = std::getenv("MLX_GRAPH_PREFILL_REPLAY");
            const char* og = std::getenv("MLX_PREFILL_ONE_GRAPH");
            const char* at = std::getenv("MLX_PREFILL_ABSORB_TAIL");
            const char* ps = std::getenv("MLX_PREFILL_STEP");
            std::cerr << "[prefill-graph] ONE_GRAPH=" << (og ? og : "<unset>")
                      << " ABSORB_TAIL=" << (at ? at : "<unset>")
                      << " HIP_GRAPH_PREFILL=" << (hp ? hp : "<unset>")
                      << " HIP_GRAPH_DECODE=" << (hd ? hd : "<unset>")
                      << " USE_HIP_GRAPHS=" << (ug ? ug : "<unset>")
                      << " PREFILL_REPLAY=" << (pr ? pr : "<unset>")
                      << " PREFILL_STEP=" << (ps ? ps : "<default>")
                      << " prompt_tokens=" << input.text.tokens.size()
                      << " window=" << window_size << "\n";
            logged = true;
        }
    }
#endif

    if (processor_.has_value()) {
        processor_->prompt(input.text.tokens);
    }

    auto prep_result = context_.prepare_fn(input, cache_, window_size);

#if defined(MLX_BUILD_ROCM)
    // Restore split caps before remainder step / decode so product path stays
    // memory-bounded after experimental whole-chunk prefill graphs.
    mlx::core::gpu_set_graph_decode_mode(false);
#endif

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
    // C7: keep slice lazy (materialized with first draft eval).
    if (use_mtp_ && state_.has_value() && state_->hidden_intermediates.has_value()) {
        auto trunk_h = state_->hidden_intermediates.value();  // [B, T, H]
        int last_pos = trunk_h.shape(1) - 1;
        auto h_slice = mx::slice(trunk_h, {0, last_pos, 0},
                                 {1, last_pos + 1, trunk_h.shape(2)});  // [1, 1, H]
        mtp_trunk_hidden_ = h_slice;
    }
}

// ---------------------------------------------------------------------------
// TokenIterator — pure-graph teardown / destructor
// ---------------------------------------------------------------------------

void TokenIterator::teardown_pure_graph_() {
#if defined(MLX_BUILD_ROCM)
    // Idempotent. Must run even after a successful pure generation: leaving the
    // decode arena active makes the next request's prefill allocate out of the
    // frozen capture arena (multi-request server segfault / garbage).
    mlx_lm::set_graph_external_pos(false);
    mlx::core::decode_capture_destroy();
    mlx::core::decode_arena_end();
    pure_logits_.reset();
    pure_graph_state_ = 0;
    pure_graph_cap_ = 0;
    pure_pos_ = 0;
#endif
}

TokenIterator::~TokenIterator() {
    teardown_pure_graph_();
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
    , mtp_greedy_spec_(mtp_uses_greedy_spec(params))
    , mtp_temperature_(params.temperature)
    , n_draft_tokens_(params.n_draft_tokens)
    , accept_history_(kAcceptHistorySize, 1)  // Initialize with 1 (accepted)
{
    // M6 (PR #63): pure-graph and MTP are mutually exclusive. TokenIterator::next
    // short-circuits to MTP when use_mtp_, so pure never runs on the same request —
    // but operators may set both env flags by mistake. Log once so M6 is auditable.
    if (use_mtp_) {
        if (!mtp_greedy_spec_) {
            static bool logged_sample = false;
            if (!logged_sample) {
                std::cerr << "[MTP] sampled mode (rejection sampling): "
                          << mtp_greedy_only_violation(params) << "\n";
                logged_sample = true;
            }
        }
        const char* pure = std::getenv("MLX_DECODE_GRAPH_PURE");
        if (pure && pure[0] == '1' && pure[1] == '\0') {
            static bool logged = false;
            if (!logged) {
                std::cerr << "[MTP] M6 XOR: MLX_DECODE_GRAPH_PURE=1 ignored while "
                             "--use-mtp is active (MTP path takes precedence)\n";
                logged = true;
            }
        }
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
    , mtp_greedy_spec_(mtp_uses_greedy_spec(params))
    , mtp_temperature_(params.temperature)
    , n_draft_tokens_(params.n_draft_tokens)
    , accept_history_(kAcceptHistorySize, 1)
{
    if (use_mtp_) {
        if (!mtp_greedy_spec_) {
            static bool logged_sample_ext = false;
            if (!logged_sample_ext) {
                std::cerr << "[MTP] sampled mode (rejection sampling): "
                          << mtp_greedy_only_violation(params) << "\n";
                logged_sample_ext = true;
            }
        }
        const char* pure = std::getenv("MLX_DECODE_GRAPH_PURE");
        if (pure && pure[0] == '1' && pure[1] == '\0') {
            static bool logged_ext = false;
            if (!logged_ext) {
                std::cerr << "[MTP] M6 XOR: MLX_DECODE_GRAPH_PURE=1 ignored while "
                             "--use-mtp is active (MTP path takes precedence)\n";
                logged_ext = true;
            }
        }
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

std::optional<mx::array> TokenIterator::mtp_run_draft_chain(
    int n_draft, bool async_launch, bool sample_draft,
    std::vector<float>* draft_logprobs,
    std::vector<mx::array>* draft_logits_rows) {
    if (n_draft <= 1 || context_.get_mtp_head_fn == nullptr
        || context_.embed_fn == nullptr || context_.apply_lm_head_fn == nullptr) {
        return std::nullopt;
    }
    MTPHead* mtp_head = static_cast<MTPHead*>(context_.get_mtp_head_fn());
    if (mtp_head == nullptr) return std::nullopt;

    // C7: MTP KV only needed when drafting ≥2 tokens in one chain (n_draft≥3).
    // Default γ≈1 path is n_draft=2 → a single draft step; skip cache reset +
    // update (rope offset 0, no write) to cut draft bandwidth/launches.
    // P0-B: when use_mtp_kv, pass the cache on every draft step including the
    // final one so the last draft can attend to prior draft keys (RoPE offset).
    // Previously i < n_draft-1 nulled the final step and starved self-attn history.
    const bool use_mtp_kv = (n_draft > 2) && !mtp_caches_.empty();
    if (use_mtp_kv) {
        for (auto& c : mtp_caches_) {
            c.set_position(0);
        }
    }

    auto hidden = mtp_trunk_hidden_.has_value()
        ? mtp_trunk_hidden_.value()
        : context_.embed_fn(y_.tokens);
    if (hidden.ndim() == 2) {
        hidden = mx::reshape(hidden, {1, 1, hidden.shape(-1)});
    }

#if defined(MLX_BUILD_ROCM)
    // Draft is T=1 recurrence — allow ROCm single-token graph decode path.
    mlx::core::gpu_set_graph_decode_mode(true);
#endif

    auto prev_tok_arr = mx::reshape(y_.tokens, {1, 1});  // d0
    std::vector<mx::array> draft_tok_arrs;
    draft_tok_arrs.reserve(static_cast<size_t>(n_draft - 1));
    if (draft_logprobs) draft_logprobs->clear();
    if (draft_logits_rows) draft_logits_rows->clear();

    for (int i = 1; i < n_draft; ++i) {
        auto prev_embed = context_.embed_fn(prev_tok_arr);
        // Pass MTP KV for all multi-draft steps (read prior keys on final step too).
        KVCache* mtp_cache = use_mtp_kv ? &mtp_caches_[0] : nullptr;
        hidden = (*mtp_head)(hidden, prev_embed, AttentionMask{}, mtp_cache);
        auto norm_h = mtp_head->apply_output_norm(hidden);
        auto logits = context_.apply_lm_head_fn(norm_h);
        if (sample_draft) {
            // Sample from draft distribution (no trunk processor on draft side).
            auto tok = sampler_.sample(logits);
            mx::eval(tok);
            int tid = tok.item<int32_t>();
            if (draft_logprobs) {
                draft_logprobs->push_back(
                    mtp_token_logprob(logits, tid, mtp_temperature_));
            }
            if (draft_logits_rows) {
                draft_logits_rows->push_back(mtp_last_row_logits(logits));
            }
            prev_tok_arr = mx::reshape(mx::astype(tok, mx::int32), {1, 1});
        } else {
            prev_tok_arr = mx::reshape(
                mx::argmax(logits, -1, /*keepdims=*/false), {1, 1});
            prev_tok_arr = mx::astype(prev_tok_arr, mx::int32);
        }
        draft_tok_arrs.push_back(prev_tok_arr);
    }

    if (draft_tok_arrs.empty()) return std::nullopt;

    auto drafts_dev = mx::reshape(
        mx::concatenate(draft_tok_arrs, /*axis=*/0),
        {static_cast<int>(draft_tok_arrs.size())});
    if (async_launch) {
        mx::async_eval(drafts_dev);
    } else {
        mx::eval(y_.tokens, drafts_dev);
    }
    return drafts_dev;
}

std::vector<int> TokenIterator::mtp_speculative_step_sampled(int n_draft) {
    // Serial draft + sequential T=1 verify with Leviathan-style rejection sampling.
    // Parallel draft / batch verify stay on the greedy-spec path only.
    StreamGuard sg(generation_stream());
    pending_draft_valid_ = false;
    pending_draft_dev_.reset();

    std::vector<float> draft_lps;
    std::vector<mx::array> draft_logit_rows;
    std::vector<int> draft_tokens;
    draft_tokens.reserve(static_cast<size_t>(n_draft));

    mx::eval(y_.tokens);
    draft_tokens.push_back(y_.tokens.item<int32_t>());

    if (n_draft > 1) {
        auto drafts_dev = mtp_run_draft_chain(
            n_draft, /*async_launch=*/false, /*sample_draft=*/true, &draft_lps,
            &draft_logit_rows);
        if (drafts_dev.has_value()) {
            mx::eval(*drafts_dev);
            const int n_extra = static_cast<int>(drafts_dev->size());
            const int32_t* dptr = drafts_dev->data<int32_t>();
            for (int i = 0; i < n_extra; ++i) {
                draft_tokens.push_back(static_cast<int>(dptr[i]));
            }
        }
    }

    if (draft_tokens.size() < 2) {
        // No draft slots — plain sample.
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }

    LMOutput::State want_hidden;
    std::optional<LMOutput::State> last_st;
    int accepted = 0;
    float temp = mtp_temperature_ > 0.0f ? mtp_temperature_ : 1.0f;

    auto note_sample = [&](const mx::array& tok) {
        if (processor_.has_value()) processor_->did_sample(tok);
    };

    for (int i = 0; i < n_draft; ++i) {
#if defined(MLX_BUILD_ROCM)
        mlx::core::gpu_set_graph_decode_mode(true);
#endif
        auto tok_arr = mx::array(
            {static_cast<int32_t>(draft_tokens[i])}, {1, 1}, mx::int32);
        LMInput::Text tok_text(tok_arr);
        auto result = context_.call_fn(
            tok_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
        state_ = result.state;
        last_st = result.state;

        mx::array logits = result.logits;
        if (processor_.has_value()) {
            // process() expects last-token style [B,V] when possible
            if (logits.ndim() == 3) {
                int seq_len = logits.shape(1);
                logits = mx::slice(logits, {0, seq_len - 1, 0},
                                   {logits.shape(0), seq_len, logits.shape(2)});
                logits = mx::squeeze(logits, 1);
            }
            logits = processor_->process(logits);
        }

        if (i < n_draft - 1) {
            const int draft_tok = draft_tokens[i + 1];
            const float log_q = mtp_token_logprob(logits, draft_tok, temp);
            const float log_p =
                (static_cast<size_t>(i) < draft_lps.size())
                    ? draft_lps[static_cast<size_t>(i)]
                    : log_q;
            // Accept with min(1, q/p) via pure helper (golden-tested).
            float ratio = mtp_accept_ratio(log_q, log_p);
            auto uarr = mx::random::uniform(0.0f, 1.0f, {}, mx::float32);
            mx::eval(uarr);
            const float u = uarr.item<float>();
            if (u <= ratio) {
                accepted++;
                note_sample(mx::array(draft_tok, mx::int32));
                continue;
            }
            // Reject: sample from Leviathan residual max(0,q-p) with bare
            // categorical (R-1: no sampler_ temp re-scale on already-temp'd r).
            mx::array tok(0, mx::int32);
            if (static_cast<size_t>(i) < draft_logit_rows.size()) {
                auto resid = mtp_residual_logits(
                    logits, draft_logit_rows[static_cast<size_t>(i)], draft_tok,
                    temp);
                tok = mx::random::categorical(resid);
            } else {
                // No draft logits — fall back to target sampler (temp applied once).
                tok = sampler_.sample(logits);
            }
            mx::eval(tok);
            note_sample(tok);
            y_ = LMInput::Text(mx::reshape(mx::astype(tok, mx::int32), {1}));
            mx::async_eval(y_.tokens);
            break;
        } else {
            // Bonus token after full draft accept.
            auto tok = sampler_.sample(logits);
            mx::eval(tok);
            note_sample(tok);
            y_ = LMInput::Text(mx::reshape(mx::astype(tok, mx::int32), {1}));
            mx::async_eval(y_.tokens);
            accepted = n_draft - 1;
        }
    }

    maybe_quantize_kv_cache(
        cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
    if (last_st.has_value()) {
        if (last_st->hidden_intermediates.has_value()) {
            auto trunk_h = last_st->hidden_intermediates.value();
            int tlen = trunk_h.shape(1);
            int p = std::max(0, tlen - 1);
            mtp_trunk_hidden_ = mx::slice(
                trunk_h, {0, p, 0}, {1, p + 1, trunk_h.shape(2)});
        }
    }

    // Emit protocol (shared pure helper): next() returns only d0; buffer
    // holds d1..d_accepted. y_ is residual/bonus for the next step's d0.
    // Fire-1 bug: returning [d0,d1,…] without filling draft_buffer_ dropped
    // every accepted draft under temp>0 (Maxwell word-garble).
    auto plan = mtp_make_emit_plan(draft_tokens, accepted);
    draft_buffer_ = std::move(plan.buffered);
    draft_buffer_idx_ = 0;

    record_acceptance(n_draft, accepted);
    mtp_draft_proposed_ += std::max(0, n_draft - 1);
    mtp_draft_accepted_ += accepted;
    mtp_speculative_steps_++;
    return {plan.d0};
}

std::vector<int> TokenIterator::mtp_speculative_step() {
    // Same stream discipline as step()/prepare(): MTP draft+verify must not
    // run on an unbound thread default stream. On ROCm, that path historically
    // threw "There is no Stream(cpu, 0) in current thread" under --use-mtp.
    StreamGuard sg(generation_stream());

    // C12: never start a new step with an unfinished pipelined v1.
    if (pending_v1_) finish_pending_v1_();

    // Fallback to plain decode if MTP is not available on this context.
    if (!use_mtp_ || context_.get_mtp_head_fn == nullptr || context_.embed_fn == nullptr
        || context_.apply_lm_head_fn == nullptr) {
        pending_draft_valid_ = false;
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }

    MTPHead* mtp_head = static_cast<MTPHead*>(context_.get_mtp_head_fn());
    // Null when the model carries no MTP head weights — fall back to plain decode.
    if (mtp_head == nullptr) {
        pending_draft_valid_ = false;
        auto token = step(y_);
        y_ = LMInput::Text(token);
        mx::eval(token);
        return {token.item<int32_t>()};
    }
    int n_draft = current_draft_count();

    // Sampled MTP (temp>0 / top_p / rep-penalty): rejection-sampling path.
    if (!mtp_greedy_spec_) {
        return mtp_speculative_step_sampled(n_draft);
    }
    static const bool kMtpTiming = (std::getenv("MTP_TIMING") != nullptr);
    // C4: overlap MTP draft with first trunk verify token (side stream).
    // Disable: MLX_MTP_NO_PARALLEL_DRAFT=1 (serial draft-then-verify).
    static const bool kNoParallelDraft =
        std::getenv("MLX_MTP_NO_PARALLEL_DRAFT") != nullptr;
    // Inter-step async prefetch of next draft. Default OFF: on gfx1150 host
    // emit is too short to hide draft, and post-step draft work is not
    // overlapped — it adds unaccounted wall (~draft_ms per step). Opt in:
    // MLX_MTP_PREFETCH=1.
    static const bool kPrefetch = (std::getenv("MLX_MTP_PREFETCH") != nullptr);
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

    // Side stream for MTP draft so it can run concurrent with trunk verify.
    // Independent of generation_stream_ (different weights path / caches).
    static thread_local mx::Stream mtp_draft_stream =
        mx::new_stream(mx::default_device());

    // Draft phase. d0 is the trunk's already-computed next token (y_), trusted
    // and never verified; the head drafts d1..d_{K-1}.
    std::vector<int> draft_tokens;
    draft_tokens.reserve(static_cast<size_t>(n_draft));

    // Consume inter-step prefetch if still valid for this n_draft.
    bool used_prefetch = false;
    std::optional<mx::array> drafts_dev_pending;
    if (pending_draft_valid_ && pending_draft_n_ == n_draft
        && pending_draft_dev_.has_value() && n_draft > 1) {
        drafts_dev_pending = pending_draft_dev_;
        used_prefetch = true;
        pending_draft_valid_ = false;
        pending_draft_dev_.reset();
    } else {
        pending_draft_valid_ = false;
        pending_draft_dev_.reset();
    }

    // Default (C2): sequential T=1 verify — uses ROCm graph-decode mode and
    // fused GDN T=1 path; early-exit on mismatch (no capture_spec tax, no
    // multi-token gated_delta_update_seq). Field: multi-token verify was the
    // residual after C1 quant draft (~86ms verify vs ~38ms eager T=1).
    // Opt into old batch verify: MLX_MTP_BATCH_VERIFY=1.
    static const bool kBatchVerify =
        std::getenv("MLX_MTP_BATCH_VERIFY") != nullptr;

    int accepted = 0;

    auto stash_hidden_from = [&](const LMOutput::State& st) {
        if (!st.hidden_intermediates.has_value()) return;
        auto trunk_h = st.hidden_intermediates.value();
        int tlen = trunk_h.shape(1);
        int p = std::max(0, tlen - 1);
        auto h_slice = mx::slice(trunk_h, {0, p, 0}, {1, p + 1, trunk_h.shape(2)});
        // C7: leave lazy — next draft's async_eval/eval pulls this; avoid
        // a hard per-step barrier after every speculative step.
        mtp_trunk_hidden_ = h_slice;
    };

    auto fill_draft_tokens_from_dev = [&](const mx::array& drafts_dev) {
        draft_tokens.clear();
        mx::eval(y_.tokens, drafts_dev);
        draft_tokens.push_back(y_.tokens.item<int32_t>());
        const int n_extra = static_cast<int>(drafts_dev.size());
        for (int i = 0; i < n_extra; ++i) {
            draft_tokens.push_back(static_cast<int>(drafts_dev.data<int32_t>()[i]));
        }
    };

    if (!kBatchVerify) {
        // Feed each draft token with L=1; compare trunk next to draft[i+1].
        // Defer KV quant + hidden stash to end of loop (fewer host barriers).
        // Empty State* is only a "return hidden" signal — no dummy array(0.0f).
        LMOutput::State want_hidden;
        std::optional<LMOutput::State> last_st;

        // C4 parallel: when we need a fresh draft, run MTP draft on a side
        // stream while the trunk verifies d0 on the generation stream. Join
        // before the accept decision. Prefetch path skips this (draft ready).
        const bool do_parallel = !kNoParallelDraft && !used_prefetch && n_draft > 1
            && !kBatchVerify;

        if (used_prefetch && drafts_dev_pending.has_value()) {
            fill_draft_tokens_from_dev(*drafts_dev_pending);
            if (kMtpTiming) t_draft = std::chrono::steady_clock::now();
        } else if (do_parallel) {
            // C6: materialize d0 *before* launching the side-stream draft.
            // C4 launched draft then mx::eval(y_.tokens) which can force a
            // device-wide join on ROCm and destroy draft‖verify overlap
            // (joint wall ~55ms vs expected max(~20 draft, ~38 T1)≈38ms).
            mx::eval(y_.tokens);
            auto d0_arr = mx::reshape(y_.tokens, {1, 1});
            if (d0_arr.dtype() != mx::int32) {
                d0_arr = mx::astype(d0_arr, mx::int32);
            }

            // Launch draft on side stream (async). y_/d0 already resident.
            std::optional<mx::array> drafts_dev;
            {
                StreamGuard dsg(mtp_draft_stream);
                drafts_dev = mtp_run_draft_chain(n_draft, /*async_launch=*/true);
            }
            // Concurrent: trunk verify of d0 on generation stream (no extra eval).
#if defined(MLX_BUILD_ROCM)
            mlx::core::gpu_set_graph_decode_mode(true);
#endif
            LMInput::Text tok_text(d0_arr);
            auto result = context_.call_fn(
                tok_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
            state_ = result.state;
            last_st = result.state;
            auto pred = mx::astype(mx::argmax(result.logits, -1), mx::int32);

            // C15: join pred + drafts on device; accept compare from device
            // pointers before building full host draft_tokens. On reject only
            // keep d0 for emit (skip host materialize of rejected drafts).
            static const bool kMtpDebugEarly =
                (std::getenv("MTP_DEBUG") != nullptr);
            // C12: pipeline v1 under host emit of d0 (γ=1 accept only).
            // Measured REGRESS on gfx1150; opt-in MLX_MTP_PIPELINE_V1=1.
            static const bool kPipelineV1 =
                std::getenv("MLX_MTP_PIPELINE_V1") != nullptr &&
                std::getenv("MLX_MTP_NO_PIPELINE_V1") == nullptr;
            bool pipelined_v1 = false;

            draft_tokens.clear();
            bool draft_match = false;
            if (drafts_dev.has_value()) {
                mx::eval(pred, *drafts_dev);
                // d0 already eval'd before side-stream draft (C6).
                const int32_t d0 = y_.tokens.item<int32_t>();
                draft_tokens.push_back(static_cast<int>(d0));
                const int n_extra = static_cast<int>(drafts_dev->size());
                const int32_t* dptr = drafts_dev->data<int32_t>();
                const int32_t trunk_next = pred.data<int32_t>()[0];
                if (n_extra >= 1 && trunk_next == dptr[0]) {
                    draft_match = true;
                    // Accept: host ids for emit/debug (d1..).
                    for (int i = 0; i < n_extra; ++i) {
                        draft_tokens.push_back(static_cast<int>(dptr[i]));
                    }
                } else if (n_extra >= 1 && kMtpDebugEarly) {
                    // Reject but keep draft ids for MTP_DEBUG visibility only.
                    for (int i = 0; i < n_extra; ++i) {
                        draft_tokens.push_back(static_cast<int>(dptr[i]));
                    }
                }
                // else reject: draft_tokens = {d0} only
            } else {
                mx::eval(pred);
                draft_tokens.push_back(
                    static_cast<int>(y_.tokens.item<int32_t>()));
            }
            if (kMtpTiming) t_draft = std::chrono::steady_clock::now();

            if (draft_match && draft_tokens.size() >= 2) {
                accepted = 1;
                // Continue sequential verify from i=1 (feed d1..).
                // Prefer device slices of drafts_dev over host re-upload.
                for (int i = 1; i < n_draft; ++i) {
#if defined(MLX_BUILD_ROCM)
                    mlx::core::gpu_set_graph_decode_mode(true);
#endif
                    // draft_tokens[i] == drafts_dev[i-1]; prefer device slice.
                    mx::array t2 =
                        drafts_dev.has_value()
                            ? mx::reshape(
                                  mx::slice(*drafts_dev, {i - 1}, {i}), {1, 1})
                            : mx::array(
                                  {static_cast<int32_t>(draft_tokens[i])},
                                  {1, 1}, mx::int32);
                    if (t2.dtype() != mx::int32) {
                        t2 = mx::astype(t2, mx::int32);
                    }
                    LMInput::Text t2_text(t2);
                    auto r2 = context_.call_fn(
                        t2_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
                    state_ = r2.state;
                    last_st = r2.state;
                    auto pred2 = mx::astype(mx::argmax(r2.logits, -1), mx::int32);
                    if (i < n_draft - 1) {
                        mx::eval(pred2);
                        int32_t tn = pred2.data<int32_t>()[0];
                        if (tn == static_cast<int32_t>(draft_tokens[i + 1])) {
                            accepted++;
                        } else {
                            y_ = LMInput::Text(mx::reshape(pred2, {1}));
                            break;
                        }
                    } else if (kPipelineV1 && n_draft == 2 && i == 1) {
                        // C12: do not wait for residual pred2 — return d0 now;
                        // finish_pending_v1_() runs when draining buffered d1.
                        pending_v1_pred_ = pred2;
                        pending_v1_state_ = r2.state;
                        pending_v1_ = true;
                        pipelined_v1 = true;
                        accepted = 1;
                    } else {
                        y_ = LMInput::Text(mx::reshape(pred2, {1}));
                        accepted = n_draft - 1;
                    }
                }
            } else {
                // Reject d1 or no draft: y_ is trunk alternative (pred eval'd).
                y_ = LMInput::Text(mx::reshape(pred, {1}));
                accepted = 0;
            }

            if (!pipelined_v1) {
                maybe_quantize_kv_cache(
                    cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
                if (last_st.has_value()) stash_hidden_from(*last_st);
                // C8: kick residual accept-path T=1 (y_/pred2) immediately so it can
                // run under host emit of d0 (+ buffered drafts). Do not force a full
                // mx::eval here — MTP_TIMING used to barrier-sync residual into the
                // step wall and destroy hide-under-emit (opt-in: MTP_TIMING_SYNC=1).
                mx::async_eval(y_.tokens);
            } else {
                // Residual y_ deferred until finish_pending_v1_(); still schedule
                // the in-flight pred graph without a host barrier.
                if (pending_v1_pred_.has_value()) {
                    mx::async_eval(*pending_v1_pred_);
                }
            }
            if (kMtpTiming) {
                static const bool kTimingSync =
                    std::getenv("MTP_TIMING_SYNC") != nullptr;
                if (kTimingSync && !pipelined_v1) mx::eval(y_.tokens);
                t_verify = std::chrono::steady_clock::now();
            }
        } else {
            // Serial draft then sequential verify (legacy / n_draft<=1 / flag).
            if (n_draft > 1) {
                auto drafts_dev = mtp_run_draft_chain(n_draft, /*async_launch=*/false);
                if (drafts_dev.has_value()) {
                    fill_draft_tokens_from_dev(*drafts_dev);
                } else {
                    mx::eval(y_.tokens);
                    draft_tokens.push_back(y_.tokens.item<int32_t>());
                }
            } else {
                mx::eval(y_.tokens);
                draft_tokens.push_back(y_.tokens.item<int32_t>());
            }
            if (kMtpTiming) t_draft = std::chrono::steady_clock::now();

            if (draft_tokens.empty()) {
                auto token = step(y_);
                y_ = LMInput::Text(token);
                mx::eval(token);
                return {token.item<int32_t>()};
            }

            for (int i = 0; i < n_draft; ++i) {
#if defined(MLX_BUILD_ROCM)
                mlx::core::gpu_set_graph_decode_mode(true);
#endif
                auto tok_arr = mx::array(
                    {static_cast<int32_t>(draft_tokens[i])}, {1, 1}, mx::int32);
                LMInput::Text tok_text(tok_arr);
                auto result = context_.call_fn(
                    tok_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
                state_ = result.state;
                last_st = result.state;

                auto pred = mx::astype(mx::argmax(result.logits, -1), mx::int32);
                if (i < n_draft - 1) {
                    mx::eval(pred);
                    int32_t trunk_next = pred.data<int32_t>()[0];
                    if (trunk_next == static_cast<int32_t>(draft_tokens[i + 1])) {
                        accepted++;
                    } else {
                        y_ = LMInput::Text(mx::reshape(pred, {1}));
                        break;
                    }
                } else {
                    y_ = LMInput::Text(mx::reshape(pred, {1}));
                    accepted = n_draft - 1;
                }
            }
            maybe_quantize_kv_cache(
                cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
            if (last_st.has_value()) stash_hidden_from(*last_st);
            // C8: async residual y_ (see parallel path).
            mx::async_eval(y_.tokens);
            if (kMtpTiming) {
                static const bool kTimingSync =
                    std::getenv("MTP_TIMING_SYNC") != nullptr;
                if (kTimingSync) mx::eval(y_.tokens);
                t_verify = std::chrono::steady_clock::now();
            }
        }

        // Prefetch path still needs sequential verify of all tokens.
        if (used_prefetch) {
            if (draft_tokens.empty()) {
                auto token = step(y_);
                y_ = LMInput::Text(token);
                mx::eval(token);
                return {token.item<int32_t>()};
            }
            for (int i = 0; i < n_draft; ++i) {
#if defined(MLX_BUILD_ROCM)
                mlx::core::gpu_set_graph_decode_mode(true);
#endif
                auto tok_arr = mx::array(
                    {static_cast<int32_t>(draft_tokens[i])}, {1, 1}, mx::int32);
                LMInput::Text tok_text(tok_arr);
                auto result = context_.call_fn(
                    tok_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
                state_ = result.state;
                last_st = result.state;

                auto pred = mx::astype(mx::argmax(result.logits, -1), mx::int32);
                if (i < n_draft - 1) {
                    mx::eval(pred);
                    int32_t trunk_next = pred.data<int32_t>()[0];
                    if (trunk_next == static_cast<int32_t>(draft_tokens[i + 1])) {
                        accepted++;
                    } else {
                        y_ = LMInput::Text(mx::reshape(pred, {1}));
                        break;
                    }
                } else {
                    y_ = LMInput::Text(mx::reshape(pred, {1}));
                    accepted = n_draft - 1;
                }
            }
            maybe_quantize_kv_cache(
                cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
            if (last_st.has_value()) stash_hidden_from(*last_st);
            // C8: async residual y_ (see parallel path).
            mx::async_eval(y_.tokens);
            if (kMtpTiming) {
                static const bool kTimingSync =
                    std::getenv("MTP_TIMING_SYNC") != nullptr;
                if (kTimingSync) mx::eval(y_.tokens);
                t_verify = std::chrono::steady_clock::now();
            }
        }
    } else {
        // Batch path needs host draft tokens first (serial draft).
        if (used_prefetch && drafts_dev_pending.has_value()) {
            fill_draft_tokens_from_dev(*drafts_dev_pending);
        } else if (n_draft > 1) {
            auto drafts_dev = mtp_run_draft_chain(n_draft, /*async_launch=*/false);
            if (drafts_dev.has_value()) {
                fill_draft_tokens_from_dev(*drafts_dev);
            } else {
                mx::eval(y_.tokens);
                draft_tokens.push_back(y_.tokens.item<int32_t>());
            }
        } else {
            mx::eval(y_.tokens);
            draft_tokens.push_back(y_.tokens.item<int32_t>());
        }
        if (kMtpTiming) t_draft = std::chrono::steady_clock::now();
        if (draft_tokens.empty()) {
            auto token = step(y_);
            y_ = LMInput::Text(token);
            mx::eval(token);
            return {token.item<int32_t>()};
        }

        // Legacy multi-token batch verify + optional capture_spec rollback.
        std::vector<int32_t> draft_seq;
        draft_seq.reserve(draft_tokens.size());
        for (int t : draft_tokens) draft_seq.push_back(static_cast<int32_t>(t));
        auto draft_arr = mx::array(
            draft_seq.data(), {1, static_cast<int>(draft_seq.size())}, mx::int32);
        LMInput::Text draft_text(draft_arr);

        struct SavedMambaState {
            MambaCache::Snapshot snapshot;
            bool has_mamba = false;
        };
        static const bool kUseIntermediates =
            std::getenv("MLX_MTP_NO_INTERMEDIATES") == nullptr;
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

        LMOutput::State want_hidden;
        auto result =
            context_.call_fn(draft_text, cache_.empty() ? nullptr : &cache_, &want_hidden);
        state_ = result.state;
        maybe_quantize_kv_cache(
            cache_, kv_bits_, kv_group_size_, quantized_kv_start_);

        auto logits = result.logits;
        auto trunk_argmax = mx::astype(mx::argmax(logits, -1), mx::int32);
        mx::eval(trunk_argmax);
        const int32_t* trunk_pred = trunk_argmax.data<int32_t>();

        accepted = 0;
        for (int i = 0; i < n_draft - 1; ++i) {
            int32_t trunk_token = trunk_pred[i];
            if (trunk_token == draft_tokens[i + 1]) {
                accepted++;
            } else {
                draft_tokens[i + 1] = trunk_token;
                break;
            }
        }

        if (accepted == n_draft - 1) {
            int32_t bonus_token = trunk_pred[n_draft - 1];
            y_ = LMInput::Text(mx::array({bonus_token}, {1}, mx::int32));
        } else {
            y_ = LMInput::Text(
                mx::array({draft_tokens[accepted + 1]}, {1}, mx::int32));
        }
        // C8: async residual y_ (batch path).
        mx::async_eval(y_.tokens);
        if (kMtpTiming) {
            static const bool kTimingSync =
                std::getenv("MTP_TIMING_SYNC") != nullptr;
            if (kTimingSync) mx::eval(y_.tokens);
            t_verify = std::chrono::steady_clock::now();
        }

        auto capture_hidden_at = [&](int pos) {
            if (result.state.has_value() &&
                result.state->hidden_intermediates.has_value()) {
                auto trunk_h = result.state->hidden_intermediates.value();
                int p = std::min(pos, static_cast<int>(trunk_h.shape(1)) - 1);
                if (p < 0) p = 0;
                auto h_slice =
                    mx::slice(trunk_h, {0, p, 0}, {1, p + 1, trunk_h.shape(2)});
                // C7: lazy stash (same as sequential path).
                mtp_trunk_hidden_ = h_slice;
            }
        };

        bool have_spec = any_mamba;
        if (any_mamba) {
            for (auto& c : cache_) {
                if (auto* m = c.as_mamba()) {
                    if (!m->has_spec()) {
                        have_spec = false;
                        break;
                    }
                }
            }
        }

        if (accepted == n_draft - 1) {
            capture_hidden_at(n_draft - 1);
        } else if (any_mamba && !have_spec) {
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
            auto rerun_arr = mx::array(
                rerun_seq.data(), {1, static_cast<int>(rerun_seq.size())}, mx::int32);
            LMInput::Text rerun_text(rerun_arr);
            LMOutput::State rerun_want_hidden;
            result = context_.call_fn(rerun_text, &cache_, &rerun_want_hidden);
            state_ = result.state;
            maybe_quantize_kv_cache(
                cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
            capture_hidden_at(accepted);
        } else if (accepted < n_draft - 1 && !cache_.empty()) {
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

        for (auto& c : cache_) {
            if (auto* m = c.as_mamba()) m->set_capture_spec(false);
        }
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
    {
        auto plan = mtp_make_emit_plan(draft_tokens, accepted);
        draft_buffer_ = std::move(plan.buffered);
        draft_buffer_idx_ = 0;
        // plan.d0 == draft_tokens[0] when non-empty (return below).
    }

    // Optional inter-step draft prefetch (MLX_MTP_PREFETCH=1 only).
    if (kPrefetch && mtp_trunk_hidden_.has_value()) {
        const int next_n = current_draft_count();
        if (next_n > 1) {
            auto pref = mtp_run_draft_chain(next_n, /*async_launch=*/true);
            if (pref.has_value()) {
                pending_draft_dev_ = std::move(*pref);
                pending_draft_n_ = next_n;
                pending_draft_valid_ = true;
            } else {
                pending_draft_valid_ = false;
                pending_draft_dev_.reset();
            }
        } else {
            pending_draft_valid_ = false;
            pending_draft_dev_.reset();
        }
    } else {
        pending_draft_valid_ = false;
        pending_draft_dev_.reset();
    }

    return {draft_tokens[0]};
}

void TokenIterator::finish_pending_v1_() {
    if (!pending_v1_ || !pending_v1_pred_.has_value()) {
        pending_v1_ = false;
        pending_v1_pred_.reset();
        pending_v1_state_.reset();
        return;
    }
    // Same stream as generate / mtp_speculative_step (ROCm TLS encoders).
    StreamGuard sg(generation_stream());
#if defined(MLX_BUILD_ROCM)
    mlx::core::gpu_set_graph_decode_mode(true);
#endif
    mx::eval(*pending_v1_pred_);
    y_ = LMInput::Text(mx::reshape(*pending_v1_pred_, {1}));
    if (pending_v1_state_.has_value()) {
        state_ = pending_v1_state_;
        // Same lazy trunk-hidden stash as mtp_speculative_step (C7).
        if (pending_v1_state_->hidden_intermediates.has_value()) {
            auto trunk_h = pending_v1_state_->hidden_intermediates.value();
            int tlen = trunk_h.shape(1);
            int p = std::max(0, tlen - 1);
            mtp_trunk_hidden_ = mx::slice(
                trunk_h, {0, p, 0}, {1, p + 1, trunk_h.shape(2)});
        }
    }
    maybe_quantize_kv_cache(
        cache_, kv_bits_, kv_group_size_, quantized_kv_start_);
    mx::async_eval(y_.tokens);
    pending_v1_ = false;
    pending_v1_pred_.reset();
    pending_v1_state_.reset();
}

void TokenIterator::record_acceptance(int proposed, int accepted) {
    uint8_t val = static_cast<uint8_t>(accepted);
    accept_history_[accept_history_idx_ % kAcceptHistorySize] = val;
    accept_history_idx_++;
}

int TokenIterator::current_draft_count() const {
    // Adaptive block size from recent accepts (C3). Pure helper is golden-tested.
    // Never collapses to "MTP off": min 2 when n_draft_tokens>=2.
    // Disable: MLX_MTP_FIXED_DRAFT=1 → always n_draft_tokens_.
    const bool fixed = (std::getenv("MLX_MTP_FIXED_DRAFT") != nullptr);
    const int n = static_cast<int>(
        std::min(accept_history_idx_, static_cast<size_t>(kAcceptHistorySize)));
    return mtp_adaptive_n_draft(
        n_draft_tokens_,
        accept_history_.data(),
        n,
        fixed);
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
        // Complete any in-flight v1 so KV/cache stay consistent even if we
        // stop without emitting the buffered draft token.
        if (pending_v1_) finish_pending_v1_();
        return std::nullopt;
    }

    // MTP path: drain buffer first, then run speculative step.
    if (use_mtp_) {
        if (!draft_buffer_.empty() && draft_buffer_idx_ < draft_buffer_.size()) {
            // C12: finish deferred v1 before emitting buffered d1 (host already
            // had a chance to emit d0 while v1 ran).
            if (pending_v1_) finish_pending_v1_();
            // Return a buffered accepted token; do NOT touch y_ further.
            int tok = draft_buffer_[draft_buffer_idx_++];
            token_count_++;
            return tok;
        }

        // Buffer exhausted — complete any orphan pending v1, then new step.
        if (pending_v1_) finish_pending_v1_();
        draft_buffer_.clear();
        draft_buffer_idx_ = 0;
        auto accepted = mtp_speculative_step();
        token_count_++;
        // Do not hard-sync y_ here: C4 may have async-prefetched the next draft
        // that depends on y_; a full barrier would collapse the overlap with
        // host emit. y_ is materialised when the next draft/eval needs it.
        // C12: when v1 is still pending, y_ is not set yet — skip async_eval.
        if (!pending_v1_) {
            mx::async_eval(y_.tokens);
        }
        measure_prefill_boundary_();
        return accepted.empty() ? std::nullopt : std::optional<int>(accepted[0]);
    }

    // Standard path: single token generation.
    static const bool g_sync_decode = std::getenv("MLX_SYNC_DECODE") != nullptr;

#if defined(MLX_BUILD_ROCM)
    // Build-once pure-relaunch graph decode. Default OFF — eager decode is the
    // stable/faster path on gfx1151 (4-bit ~68 tok/s eager vs ~64 pure). Enable
    // with MLX_DECODE_GRAPH_PURE=1 when profiling launch-bound dGPUs (e.g. R9700).
    // Opt-in only when exactly "1" — presence of =0 / =false must stay eager.
    // XOR: MLX_REDLINE_DECODE=1 + pure=1 → fail-closed eager (E4 §3).
    static const bool pure_enabled = [] {
        if (redline_decode_env_enabled_() && pure_graph_env_enabled_()) {
            return false;
        }
        return pure_graph_env_enabled_();
    }();
    // P0/P2/P6 status also from next() so XOR / redline-only always one-shot.
    maybe_log_redline_session_status();
    maybe_probe_redline_graph_decode_bind();
    if (pure_enabled && pure_graph_state_ != 9 && !cache_.empty()) {
        if (pure_graph_cap_ == 0) {
            int off = 0;
            for (auto& c : cache_) off = std::max(off, c.offset());
            int remaining = max_tokens_.has_value()
                ? std::max(0, max_tokens_.value() - token_count_) : 256;
            pure_graph_cap_ = off + remaining + 8;
        }
        // Emit-previous pipeline: prepare() sampled token 0 into y_;
        // step_pure_graph materialises the *new* sample (pure_logits_ is
        // overwritten on the next relaunch). previous_y was already eval'd when
        // it was stored (prepare or prior step), so item() is host-side only.
        auto previous_y = y_;
        auto token = step_pure_graph(previous_y);
        y_ = LMInput::Text(std::move(token));
        token_count_++;
        measure_prefill_boundary_();
        return previous_y.tokens.item<int32_t>();
    }
#endif

    auto previous_y = y_;
    auto token = step(previous_y);
    y_ = LMInput::Text(token);
    if (g_sync_decode) {
        // Diagnostic: fully retire each forward before building the next.
        // Still emit previous_y — sync only forces the new sample to complete
        // before we hand control back; it must not change which token is
        // reported to the caller (see pure-graph first-token drop above).
        mx::eval(token);
        token_count_++;
        measure_prefill_boundary_();
        mx::eval(previous_y.tokens);
        maybe_log_kv_offset_(cache_, token_count_);
        return previous_y.tokens.item<int32_t>();
    }
    mx::async_eval(token);
    token_count_++;
    mx::eval(previous_y.tokens);
    measure_prefill_boundary_();
    maybe_log_kv_offset_(cache_, token_count_);
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
    const std::function<GenerateDisposition(int token)>& on_token,
    const GenerateCancelPredicate& should_cancel)
{
    // Upgrade GPU wired memory for the duration of generation.
    WiredLimitGuard wired_guard;

    int prompt_token_count = static_cast<int>(input.text.tokens.size());

    TokenIterator iter(context, input, params);

    auto start = std::chrono::steady_clock::now();
    int token_count = 0;

    while (true) {
        // Poll before the forward pass, so an abandoned request stops without
        // spending another token's compute. on_token cannot be relied on for
        // this: generate_text() skips it whenever the detokenizer has no text.
        if (should_cancel && should_cancel()) {
            break;
        }

        auto maybe_token = iter.next();
        if (!maybe_token) {
            break;
        }
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
    const std::function<GenerateDisposition(const std::string& text, int token)>& on_text,
    const GenerateCancelPredicate& should_cancel)
{
    auto decode_fn = context.decode_fn;

    NaiveStreamingDetokenizer detokenizer;

    return generate(context, input, params, eos_token_ids,
        [&](int token) -> GenerateDisposition {
            detokenizer.append(token);
            if (auto text = detokenizer.next(decode_fn)) {
                return on_text(*text, token);
            }
            // No text this token (incomplete UTF-8, or it did not extend the
            // segment). Cancellation rests entirely on should_cancel, which
            // generate() polls every iteration.
            return GenerateDisposition::more;
        },
        should_cancel);
}

} // namespace mlx_lm
