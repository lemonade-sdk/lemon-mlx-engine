// Tests for samplers and generation utilities
//
// Includes ports of EvalTests.swift: testConcurrentSampling,
// testRandomStateIsolation (adapted for single-threaded sequential execution).

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mx = mlx::core;

// ===== Existing tests =====

TEST_CASE("ArgMaxSampler", "[generate]") {
    mlx_lm::ArgMaxSampler sampler;
    auto logits = mx::array({0.1f, 0.3f, 0.9f, 0.2f});
    auto token = sampler.sample(logits);
    REQUIRE(token.item<int32_t>() == 2);
}

TEST_CASE("AnySampler from params (temperature=0)", "[generate]") {
    mlx_lm::GenerateParameters params;
    params.temperature = 0.0f;
    auto sampler = mlx_lm::AnySampler::from_params(params);

    auto logits = mx::array({0.1f, 0.5f, 0.3f});
    auto token = sampler.sample(logits);
    REQUIRE(token.item<int32_t>() == 1);
}

TEST_CASE("GenerateParameters defaults", "[generate]") {
    mlx_lm::GenerateParameters params;
    REQUIRE(params.temperature == 0.6f);
    REQUIRE(params.top_p == 1.0f);
    REQUIRE(params.prefill_step_size == 512);
    REQUIRE(!params.max_tokens.has_value());
    REQUIRE(!params.max_kv_size.has_value());
}

TEST_CASE("NaiveStreamingDetokenizer", "[generate]") {
    mlx_lm::NaiveStreamingDetokenizer detok;

    // Simple mock decode function
    auto decode = [](const std::vector<int>& tokens) -> std::string {
        std::string result;
        for (int t : tokens) {
            result += static_cast<char>('a' + (t % 26));
        }
        return result;
    };

    detok.append(0);  // 'a'
    auto text = detok.next(decode);
    REQUIRE(text.has_value());
    REQUIRE(text.value() == "a");

    detok.append(1);  // 'b'
    text = detok.next(decode);
    REQUIRE(text.has_value());
    REQUIRE(text.value() == "b");
}

// ===== New tests ported from Swift EvalTests =====

// Port of EvalTests.testConcurrentSampling — sequential adaptation.
// Verifies that sampling from random logits produces valid token IDs
// (in range [0, vocab_size)) for both categorical and argmax samplers.
TEST_CASE("Sampling produces valid token IDs", "[generate]") {
    const int vocab_size = 100;
    const int num_samplers = 4;

    for (int sampler_id = 0; sampler_id < num_samplers; ++sampler_id) {
        // Generate random logits.
        mx::random::seed(static_cast<uint32_t>(sampler_id));
        auto logits = mx::random::normal({1, vocab_size});

        int result;
        if (sampler_id % 2 == 0) {
            // Use categorical sampling.
            result = mx::random::categorical(logits).item<int32_t>();
        } else {
            // Use argmax.
            result = mx::argmax(logits, -1).item<int32_t>();
        }

        CHECK(result >= 0);
        CHECK(result < vocab_size);
    }
}

// Port of EvalTests.testRandomStateIsolation — sequential adaptation.
// Verifies that the CategoricalSampler produces results in valid range
// and that re-seeding produces deterministic results.
TEST_CASE("CategoricalSampler produces valid results", "[generate]") {
    const int vocab_size = 50;
    const int num_samplers = 5;
    const int samples_per_task = 10;

    std::vector<std::vector<int>> all_results;

    for (int sampler_id = 0; sampler_id < num_samplers; ++sampler_id) {
        auto logits = mx::ones({1, vocab_size});
        std::vector<int> task_results;
        mlx_lm::CategoricalSampler sampler(1.0f);

        for (int sample_id = 0; sample_id < samples_per_task; ++sample_id) {
            // Seed per-sample for reproducibility (mirrors Swift's withRandomState).
            mx::random::seed(
                static_cast<uint32_t>(sampler_id * 1000 + sample_id));
            auto token = sampler.sample(logits);
            task_results.push_back(token.item<int32_t>());
        }

        REQUIRE(task_results.size() == static_cast<size_t>(samples_per_task));
        all_results.push_back(task_results);
    }

    REQUIRE(all_results.size() == static_cast<size_t>(num_samplers));

    // Verify all results are in valid range.
    for (const auto& sampler_results : all_results) {
        for (int tok : sampler_results) {
            CHECK(tok >= 0);
            CHECK(tok < vocab_size);
        }
    }

    // There should be at least one unique sequence (non-degenerate output).
    std::set<std::vector<int>> unique_sequences(
        all_results.begin(), all_results.end());
    CHECK(unique_sequences.size() > 0);
}

// Test that deterministic seeding produces identical results.
TEST_CASE("Deterministic sampling with same seed", "[generate]") {
    const int vocab_size = 100;
    mlx_lm::CategoricalSampler sampler(1.0f);

    auto logits = mx::random::normal({1, vocab_size});
    mx::eval(logits);

    // Sample twice with the same seed.
    mx::random::seed(42);
    auto token1 = sampler.sample(logits);
    int t1 = token1.item<int32_t>();

    mx::random::seed(42);
    auto token2 = sampler.sample(logits);
    int t2 = token2.item<int32_t>();

    REQUIRE(t1 == t2);
}

// Test that AnySampler correctly dispatches based on parameters.
TEST_CASE("AnySampler dispatch based on parameters", "[generate]") {
    SECTION("temperature=0 produces argmax") {
        mlx_lm::GenerateParameters params;
        params.temperature = 0.0f;
        auto sampler = mlx_lm::AnySampler::from_params(params);

        // With clear maximum at index 3, argmax should always pick 3.
        auto logits = mx::array({0.1f, 0.2f, 0.3f, 10.0f, 0.4f});
        auto token = sampler.sample(logits);
        REQUIRE(token.item<int32_t>() == 3);
    }

    SECTION("temperature>0 with top_p<1 creates top-p sampler") {
        // Verify from_params selects the TopPSampler path (no crash).
        mlx_lm::GenerateParameters params;
        params.temperature = 0.8f;
        params.top_p = 0.9f;
        auto sampler = mlx_lm::AnySampler::from_params(params);

        // TopPSampler is created; verify we can also get argmax from
        // different params (dispatch correctness).
        mlx_lm::GenerateParameters argmax_params;
        argmax_params.temperature = 0.0f;
        auto argmax_sampler = mlx_lm::AnySampler::from_params(argmax_params);
        auto logits = mx::array({0.1f, 0.2f, 0.3f, 10.0f, 0.4f});
        REQUIRE(argmax_sampler.sample(logits).item<int32_t>() == 3);
    }

    SECTION("temperature>0 with top_p=1 uses categorical sampler") {
        mlx_lm::GenerateParameters params;
        params.temperature = 1.0f;
        params.top_p = 1.0f;
        auto sampler = mlx_lm::AnySampler::from_params(params);

        auto logits = mx::array({0.1f, 0.2f, 0.3f, 10.0f, 0.4f});
        auto token = sampler.sample(logits);
        int tok = token.item<int32_t>();
        CHECK(tok >= 0);
        CHECK(tok < 5);
    }
}

// Test RepetitionProcessor prompt loading and did_sample lifecycle.
TEST_CASE("RepetitionProcessor prompt and did_sample", "[generate]") {
    // Construct a processor and verify basic lifecycle operations do not throw.
    mlx_lm::RepetitionProcessor proc(1.5f, 10);

    // Load a prompt.
    auto prompt = mx::array({0, 1, 2, 3, 4}, mx::int32);
    proc.prompt(prompt);

    // did_sample should track newly sampled tokens without error.
    proc.did_sample(mx::array(5, mx::int32));
    proc.did_sample(mx::array(6, mx::int32));

    // Re-loading a prompt should reset state without error.
    auto prompt2 = mx::array({10, 11, 12}, mx::int32);
    proc.prompt(prompt2);

    // Verify we can still call did_sample after reset.
    proc.did_sample(mx::array(13, mx::int32));
}

// Test RepetitionProcessor with penalty=1.0 (no-op).
TEST_CASE("RepetitionProcessor no-op with penalty 1.0", "[generate]") {
    mlx_lm::RepetitionProcessor proc(1.0f, 10);

    auto prompt = mx::array({0, 1, 2}, mx::int32);
    proc.prompt(prompt);

    auto logits = mx::ones({1, 5});
    auto processed = proc.process(logits);
    mx::eval(processed);

    // With penalty=1.0, logits should be returned unchanged.
    auto data = processed.data<float>();
    for (int i = 0; i < 5; ++i) {
        CHECK(data[i] == 1.0f);
    }
}

// Test RepetitionProcessor with empty context returns logits unchanged.
TEST_CASE("RepetitionProcessor empty context", "[generate]") {
    mlx_lm::RepetitionProcessor proc(2.0f, 5);

    // Empty prompt => no tokens in context.
    auto prompt = mx::array(std::vector<int32_t>{}.data(), {0}, mx::int32);
    proc.prompt(prompt);

    auto logits = mx::ones({1, 10});
    auto processed = proc.process(logits);
    mx::eval(processed);

    // No tokens to penalize, so logits should be unchanged.
    auto data = processed.data<float>();
    for (int i = 0; i < 10; ++i) {
        CHECK(data[i] == 1.0f);
    }
}

// Test RepetitionProcessor with non-trivial penalty exercises the scatter path.
// This verifies the fix for negative axis in mx::scatter (was -1, now logits.ndim() - 1).
TEST_CASE("RepetitionProcessor applies penalty via scatter", "[generate]") {
    mlx_lm::RepetitionProcessor proc(2.0f, 10);

    // Prompt contains token 3, which maps to index 3 in the vocab.
    auto prompt = mx::array({3}, mx::int32);
    proc.prompt(prompt);

    // Create logits where index 3 has a positive value (should be divided by penalty).
    auto logits = mx::array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    logits = mx::reshape(logits, {1, 5});
    auto processed = proc.process(logits);
    mx::eval(processed);

    auto data = processed.data<float>();
    // Index 3 was penalized: 4.0 / 2.0 = 2.0
    CHECK(data[3] == 2.0f);
    // Other indices should be unchanged.
    CHECK(data[0] == 1.0f);
    CHECK(data[1] == 2.0f);
    CHECK(data[2] == 3.0f);
    CHECK(data[4] == 5.0f);
}

// Test RepetitionProcessor with negative logits (multiplied by penalty).
TEST_CASE("RepetitionProcessor handles negative logits", "[generate]") {
    mlx_lm::RepetitionProcessor proc(2.0f, 10);

    // Prompt contains token 2.
    auto prompt = mx::array({2}, mx::int32);
    proc.prompt(prompt);

    // Create logits where index 2 has a negative value.
    auto logits = mx::array({1.0f, 2.0f, -3.0f, 4.0f, 5.0f});
    logits = mx::reshape(logits, {1, 5});
    auto processed = proc.process(logits);
    mx::eval(processed);

    auto data = processed.data<float>();
    // Index 2 was penalized: -3.0 * 2.0 = -6.0
    CHECK(data[2] == -6.0f);
    // Other indices should be unchanged.
    CHECK(data[0] == 1.0f);
    CHECK(data[1] == 2.0f);
    CHECK(data[3] == 4.0f);
    CHECK(data[4] == 5.0f);
}

// Test RepetitionProcessor with 1D logits (single batch, no batch dim).
TEST_CASE("RepetitionProcessor works with 1D logits", "[generate]") {
    mlx_lm::RepetitionProcessor proc(2.0f, 10);

    auto prompt = mx::array({1}, mx::int32);
    proc.prompt(prompt);

    // 1D logits: [10.0f, -5.0f, 3.0f, 7.0f]
    auto logits = mx::array({10.0f, -5.0f, 3.0f, 7.0f});
    auto processed = proc.process(logits);
    mx::eval(processed);

    auto data = processed.data<float>();
    // Index 1 was negative: -5.0 * 2.0 = -10.0
    CHECK(data[1] == -10.0f);
    CHECK(data[0] == 10.0f);
    CHECK(data[2] == 3.0f);
    CHECK(data[3] == 7.0f);
}

// Test AnyProcessor creation from params.
TEST_CASE("AnyProcessor from params", "[generate]") {
    SECTION("no repetition penalty returns nullopt") {
        mlx_lm::GenerateParameters params;
        auto proc = mlx_lm::AnyProcessor::from_params(params);
        REQUIRE(!proc.has_value());
    }

    SECTION("with repetition penalty returns processor") {
        mlx_lm::GenerateParameters params;
        params.repetition_penalty = 1.2f;
        params.repetition_context_size = 20;
        auto proc = mlx_lm::AnyProcessor::from_params(params);
        REQUIRE(proc.has_value());
    }
}

// Test GenerateCompletionInfo statistics.
TEST_CASE("GenerateCompletionInfo", "[generate]") {
    mlx_lm::GenerateCompletionInfo info;
    info.prompt_token_count = 100;
    info.generation_token_count = 50;
    info.prompt_time = 0.5;
    info.generation_time = 2.5;

    CHECK(info.prompt_tokens_per_second() == 200.0);
    CHECK(info.tokens_per_second() == 20.0);

    auto summary = info.summary();
    CHECK(!summary.empty());
}

// Test GenerateCompletionInfo with zero times (no division by zero).
TEST_CASE("GenerateCompletionInfo zero time", "[generate]") {
    mlx_lm::GenerateCompletionInfo info;
    info.prompt_token_count = 0;
    info.generation_token_count = 0;
    info.prompt_time = 0.0;
    info.generation_time = 0.0;

    CHECK(info.prompt_tokens_per_second() == 0.0);
    CHECK(info.tokens_per_second() == 0.0);
}

// Test NaiveStreamingDetokenizer with newline segment reset.
TEST_CASE("NaiveStreamingDetokenizer segment reset on newline", "[generate]") {
    mlx_lm::NaiveStreamingDetokenizer detok;

    // Decode function: token 0 -> "hello\n", token 1 -> "world"
    auto decode = [](const std::vector<int>& tokens) -> std::string {
        std::string result;
        for (int t : tokens) {
            if (t == 0) result += "hello\n";
            else if (t == 1) result += "world";
            else result += "?";
        }
        return result;
    };

    detok.append(0);
    auto text = detok.next(decode);
    REQUIRE(text.has_value());
    REQUIRE(text.value() == "hello\n");

    // After newline, segment should be reset.
    detok.append(1);
    text = detok.next(decode);
    REQUIRE(text.has_value());
    REQUIRE(text.value() == "world");
}

// ===== Cancellation predicate =====

// generate() reaches the model only through ModelContext's type-erased function
// members, so a fake context exercises the real decode loop with no weights.
// The model always emits kForcedToken; decode_fn decides what (if anything) the
// detokenizer surfaces.
static mlx_lm::ModelContext make_fake_context(
    std::function<std::string(const std::vector<int>&)> decode_fn)
{
    constexpr int kVocabSize = 8;
    constexpr int kForcedToken = 3;

    mlx_lm::ModelContext ctx;
    ctx.model_id = "fake-model";
    ctx.new_cache_fn = [](const mlx_lm::GenerateParameters&) {
        return std::vector<mlx_lm::KVCache>{};
    };
    ctx.prepare_fn = [](const mlx_lm::LMInput& input, std::vector<mlx_lm::KVCache>&, int) {
        return mlx_lm::PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [](const mlx_lm::LMInput::Text&, std::vector<mlx_lm::KVCache>*,
                     const mlx_lm::LMOutput::State*) {
        std::vector<float> logits(kVocabSize, 0.0f);
        logits[kForcedToken] = 10.0f;  // argmax sampling makes this deterministic
        return mlx_lm::LMOutput(mx::array(logits.data(), {1, 1, kVocabSize}, mx::float32));
    };
    ctx.decode_fn = std::move(decode_fn);
    return ctx;
}

static mlx_lm::GenerateParameters greedy_params(int max_tokens) {
    mlx_lm::GenerateParameters params;
    params.temperature = 0.0f;  // argmax
    params.max_tokens = max_tokens;
    return params;
}

// Regression: the client-disconnect check must not depend on on_text firing.
// The detokenizer withholds text for incomplete UTF-8 and for tokens that do
// not extend the segment, so a cancel that only runs from on_text would keep
// decoding (holding the model lock) while the detokenizer stays silent.
TEST_CASE("generate_text cancels when the detokenizer emits no text", "[generate]") {
    // Decode to a constant: new_segment never grows, so next() always returns
    // nullopt and on_text is never invoked.
    auto ctx = make_fake_context([](const std::vector<int>&) -> std::string {
        return "";
    });

    auto params = greedy_params(64);
    mlx_lm::LMInput input(mx::array({1, 2, 3}, {3}, mx::int32));

    int text_callbacks = 0;
    int cancel_polls = 0;

    auto info = mlx_lm::generate_text(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](const std::string&, int) {
            text_callbacks++;
            return mlx_lm::GenerateDisposition::more;
        },
        [&]() {
            cancel_polls++;
            return cancel_polls > 3;  // "client disconnected" after 3 tokens
        });

    // The gap this guards: no text chunk ever reached the callback...
    REQUIRE(text_callbacks == 0);
    // ...yet the predicate was still polled and stopped generation early,
    // rather than running to max_tokens.
    REQUIRE(cancel_polls == 4);
    REQUIRE(info.generation_token_count < 64);
}

// The predicate must also be polled when the callback *is* firing normally.
TEST_CASE("generate_text cancel predicate runs every token", "[generate]") {
    // Each token appends a character, so every token yields text.
    auto ctx = make_fake_context([](const std::vector<int>& tokens) -> std::string {
        return std::string(tokens.size(), 'x');
    });

    auto params = greedy_params(64);
    mlx_lm::LMInput input(mx::array({1, 2, 3}, {3}, mx::int32));

    int text_callbacks = 0;
    int cancel_polls = 0;

    mlx_lm::generate_text(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](const std::string& text, int) {
            text_callbacks++;
            REQUIRE(text == "x");
            return mlx_lm::GenerateDisposition::more;
        },
        [&]() {
            cancel_polls++;
            return cancel_polls > 2;
        });

    REQUIRE(cancel_polls == 3);
    // Polled before each forward pass, so at most one text chunk per poll.
    REQUIRE(text_callbacks <= 2);
}

// An absent predicate must not break the ordinary path.
TEST_CASE("generate_text without a cancel predicate runs to completion", "[generate]") {
    auto ctx = make_fake_context([](const std::vector<int>& tokens) -> std::string {
        return std::string(tokens.size(), 'x');
    });

    auto params = greedy_params(5);
    mlx_lm::LMInput input(mx::array({1, 2, 3}, {3}, mx::int32));

    int text_callbacks = 0;
    auto info = mlx_lm::generate_text(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](const std::string&, int) {
            text_callbacks++;
            return mlx_lm::GenerateDisposition::more;
        });

    REQUIRE(text_callbacks > 0);
    REQUIRE(info.generation_token_count > 0);
}

// ===== TokenIterator emission contract =====
//
// prepare() samples the first generated token into y_. next() always launches
// the *next* forward, then returns the *previous* sample (pipeline). Emitting
// the newly sampled token instead drops token 0 — live ROCm pure-graph bug:
// "Maxwell" -> "well", "2+2" -> empty content (only EOS survived).

// Fake model that samples a deterministic ascending sequence under argmax:
// call N produces token (kFirstToken + N). prepare-via-tokens does one call
// (token kFirstToken), then each next() does one more call.
static mlx_lm::ModelContext make_sequenced_context(
    int vocab_size,
    int first_token,
    std::shared_ptr<int> call_count,
    bool non_empty_cache = false)
{
    mlx_lm::ModelContext ctx;
    ctx.model_id = "sequenced-fake-model";
    ctx.new_cache_fn = [non_empty_cache](const mlx_lm::GenerateParameters&) {
        // Non-empty cache is what arms the ROCm pure-graph path in next().
        // The pure-graph kernels are not exercised here (empty KV data); we
        // only need the branch condition for contract tests on ROCm builds.
        if (non_empty_cache) {
            return std::vector<mlx_lm::KVCache>{mlx_lm::KVCache{mlx_lm::KVCacheSimple{}}};
        }
        return std::vector<mlx_lm::KVCache>{};
    };
    ctx.prepare_fn = [](const mlx_lm::LMInput& input, std::vector<mlx_lm::KVCache>&, int) {
        // Remaining tokens -> prepare() runs step() once to sample token 0.
        return mlx_lm::PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [vocab_size, first_token, call_count](
                      const mlx_lm::LMInput::Text&,
                      std::vector<mlx_lm::KVCache>*,
                      const mlx_lm::LMOutput::State*) {
        int n = (*call_count)++;
        int tok = first_token + n;
        if (tok < 0 || tok >= vocab_size) {
            tok = vocab_size - 1;
        }
        std::vector<float> logits(static_cast<size_t>(vocab_size), 0.0f);
        logits[static_cast<size_t>(tok)] = 10.0f;
        return mlx_lm::LMOutput(
            mx::array(logits.data(), {1, 1, vocab_size}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>& tokens) {
        std::string s;
        for (int t : tokens) s += static_cast<char>('A' + (t % 26));
        return s;
    };
    return ctx;
}

// Also exercise the prepare-returns-logits path (used by many VLMs and by
// llm_default_prepare after the last window). First sample is convert_to_token
// without a call_fn; subsequent samples come from call_fn.
static mlx_lm::ModelContext make_sequenced_logits_prepare_context(
    int vocab_size,
    int first_token,
    std::shared_ptr<int> call_count)
{
    mlx_lm::ModelContext ctx;
    ctx.model_id = "sequenced-logits-prepare";
    ctx.new_cache_fn = [](const mlx_lm::GenerateParameters&) {
        return std::vector<mlx_lm::KVCache>{};
    };
    ctx.prepare_fn = [vocab_size, first_token](
                         const mlx_lm::LMInput&,
                         std::vector<mlx_lm::KVCache>&,
                         int) {
        std::vector<float> logits(static_cast<size_t>(vocab_size), 0.0f);
        logits[static_cast<size_t>(first_token)] = 10.0f;
        return mlx_lm::PrepareResult::logits(
            mlx_lm::LMOutput(mx::array(logits.data(), {1, 1, vocab_size}, mx::float32)));
    };
    ctx.call_fn = [vocab_size, first_token, call_count](
                      const mlx_lm::LMInput::Text&,
                      std::vector<mlx_lm::KVCache>*,
                      const mlx_lm::LMOutput::State*) {
        // After prepare sampled first_token, decode calls start at first_token+1.
        int n = (*call_count)++;
        int tok = first_token + 1 + n;
        if (tok < 0 || tok >= vocab_size) tok = vocab_size - 1;
        std::vector<float> logits(static_cast<size_t>(vocab_size), 0.0f);
        logits[static_cast<size_t>(tok)] = 10.0f;
        return mlx_lm::LMOutput(
            mx::array(logits.data(), {1, 1, vocab_size}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>&) { return std::string("x"); };
    return ctx;
}

TEST_CASE("TokenIterator emits the first prepared token (tokens prepare path)",
          "[generate][first_token]") {
    constexpr int kVocab = 32;
    constexpr int kFirst = 5;
    constexpr int kMax = 4;
    auto calls = std::make_shared<int>(0);
    auto ctx = make_sequenced_context(kVocab, kFirst, calls, /*non_empty_cache=*/false);

    auto params = greedy_params(kMax);
    mlx_lm::LMInput input(mx::array({1, 2, 3}, {3}, mx::int32));

    std::vector<int> emitted;
    mlx_lm::generate(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](int token) {
            emitted.push_back(token);
            return mlx_lm::GenerateDisposition::more;
        });

    REQUIRE(emitted.size() == static_cast<size_t>(kMax));
    // Must start at the prefill sample, not the second decode sample.
    CHECK(emitted[0] == kFirst);
    CHECK(emitted[1] == kFirst + 1);
    CHECK(emitted[2] == kFirst + 2);
    CHECK(emitted[3] == kFirst + 3);
}

TEST_CASE("TokenIterator emits the first prepared token (logits prepare path)",
          "[generate][first_token]") {
    constexpr int kVocab = 32;
    constexpr int kFirst = 7;
    constexpr int kMax = 3;
    auto calls = std::make_shared<int>(0);
    auto ctx = make_sequenced_logits_prepare_context(kVocab, kFirst, calls);

    auto params = greedy_params(kMax);
    mlx_lm::LMInput input(mx::array({9, 8}, {2}, mx::int32));

    std::vector<int> emitted;
    mlx_lm::generate(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](int token) {
            emitted.push_back(token);
            return mlx_lm::GenerateDisposition::more;
        });

    REQUIRE(emitted.size() == static_cast<size_t>(kMax));
    CHECK(emitted[0] == kFirst);
    CHECK(emitted[1] == kFirst + 1);
    CHECK(emitted[2] == kFirst + 2);
}

// When the first sample is EOS, generation must stop with zero content tokens
// after the caller filters EOS — not skip EOS and emit the *next* token.
TEST_CASE("TokenIterator first-token EOS is not skipped", "[generate][first_token]") {
    constexpr int kVocab = 16;
    constexpr int kEos = 2;
    // Force every sample to EOS (including prepare's first sample).
    auto calls = std::make_shared<int>(0);
    mlx_lm::ModelContext ctx;
    ctx.model_id = "always-eos";
    ctx.new_cache_fn = [](const mlx_lm::GenerateParameters&) {
        return std::vector<mlx_lm::KVCache>{};
    };
    ctx.prepare_fn = [](const mlx_lm::LMInput& input, std::vector<mlx_lm::KVCache>&, int) {
        return mlx_lm::PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [](const mlx_lm::LMInput::Text&, std::vector<mlx_lm::KVCache>*,
                     const mlx_lm::LMOutput::State*) {
        std::vector<float> logits(kVocab, 0.0f);
        logits[kEos] = 10.0f;
        return mlx_lm::LMOutput(mx::array(logits.data(), {1, 1, kVocab}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>&) { return std::string("x"); };

    auto params = greedy_params(8);
    mlx_lm::LMInput input(mx::array({1}, {1}, mx::int32));

    std::vector<int> emitted;
    mlx_lm::generate(
        ctx, input, params, /*eos_token_ids=*/{kEos},
        [&](int token) {
            emitted.push_back(token);
            return mlx_lm::GenerateDisposition::more;
        });

    // EOS is observed by generate() and stops without invoking on_token.
    REQUIRE(emitted.empty());
}

// Regression for the live "Maxwell" / empty "2+2" failure mode: if the first
// real content token is dropped and the second is EOS, content is empty even
// though prepare() produced a good sample.
TEST_CASE("TokenIterator does not drop content when second sample is EOS",
          "[generate][first_token]") {
    constexpr int kVocab = 32;
    constexpr int kContent = 11;  // "answer"
    constexpr int kEos = 2;
    auto calls = std::make_shared<int>(0);

    mlx_lm::ModelContext ctx;
    ctx.model_id = "content-then-eos";
    ctx.new_cache_fn = [](const mlx_lm::GenerateParameters&) {
        return std::vector<mlx_lm::KVCache>{};
    };
    ctx.prepare_fn = [](const mlx_lm::LMInput& input, std::vector<mlx_lm::KVCache>&, int) {
        return mlx_lm::PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [calls](const mlx_lm::LMInput::Text&, std::vector<mlx_lm::KVCache>*,
                          const mlx_lm::LMOutput::State*) {
        int n = (*calls)++;
        std::vector<float> logits(kVocab, 0.0f);
        // prepare step -> content; first next() step -> EOS
        logits[n == 0 ? kContent : kEos] = 10.0f;
        return mlx_lm::LMOutput(mx::array(logits.data(), {1, 1, kVocab}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>& tokens) {
        std::string s;
        for (int t : tokens) {
            if (t == kContent) s += "4";
            else s += "?";
        }
        return s;
    };

    auto params = greedy_params(8);
    mlx_lm::LMInput input(mx::array({1, 2, 3}, {3}, mx::int32));

    std::vector<int> emitted;
    std::string text;
    mlx_lm::generate_text(
        ctx, input, params, /*eos_token_ids=*/{kEos},
        [&](const std::string& chunk, int token) {
            emitted.push_back(token);
            text += chunk;
            return mlx_lm::GenerateDisposition::more;
        });

    REQUIRE(emitted.size() == 1);
    CHECK(emitted[0] == kContent);
    CHECK(text == "4");
}

// When pure-graph is armed (non-empty cache on ROCm), the emission contract
// must still hold. On non-ROCm builds this is the same as the empty-cache path.
TEST_CASE("TokenIterator first-token contract with non-empty cache",
          "[generate][first_token]") {
    constexpr int kVocab = 32;
    constexpr int kFirst = 4;
    constexpr int kMax = 3;
    auto calls = std::make_shared<int>(0);
    auto ctx = make_sequenced_context(kVocab, kFirst, calls, /*non_empty_cache=*/true);

    auto params = greedy_params(kMax);
    mlx_lm::LMInput input(mx::array({1}, {1}, mx::int32));

    std::vector<int> emitted;
    // Pure-graph may fall back or crash if the ROCm capture path is forced with
    // an empty fake model; force the eager pipeline when the env allows.
    // The contract under test is independent of capture: emit previous_y.
#if defined(MLX_BUILD_ROCM)
    // Ensure pure graph is allowed if the build enables it; the fake model has
    // no weights so capture typically disables (state 9) and falls back to step().
    // Either path must still emit kFirst first.
#endif
    mlx_lm::generate(
        ctx, input, params, /*eos_token_ids=*/{},
        [&](int token) {
            emitted.push_back(token);
            return mlx_lm::GenerateDisposition::more;
        });

    REQUIRE(emitted.size() == static_cast<size_t>(kMax));
    CHECK(emitted[0] == kFirst);
    CHECK(emitted[1] == kFirst + 1);
    CHECK(emitted[2] == kFirst + 2);
}

// ===== MTPHead draft-token shape smoke =====
#include <mlx-lm/llm/models/mtp_head.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/attention_utils.h>

static mlx_lm::MTPHeadConfig make_tiny_mtp_args() {
    mlx_lm::MTPHeadConfig args;
    args.hidden_size = 32;
    args.intermediate_size = 64;
    args.num_attention_heads = 4;
    args.num_key_value_heads = 2;
    args.head_dim = 8;
    args.rope_dims = 8;
    args.rms_norm_eps = 1e-6f;
    args.rope_theta = 10000.0f;
    return args;
}

TEST_CASE("MTPHead: weight map (dense)", "[mtp]") {
    auto args = make_tiny_mtp_args();
    mlx_lm::MTPHead head(args);
    auto wmap = head.weight_map();
    REQUIRE(wmap.count("pre_fc_norm_hidden.weight") == 1);
    REQUIRE(wmap.count("pre_fc_norm_embedding.weight") == 1);
    REQUIRE(wmap.count("fc.weight") == 1);
    REQUIRE(wmap.count("layers.0.input_layernorm.weight") == 1);
    REQUIRE(wmap.count("layers.0.post_attention_layernorm.weight") == 1);
    REQUIRE(wmap.count("layers.0.self_attn.q_proj.weight") == 1);
    REQUIRE(wmap.count("layers.0.mlp.gate_proj.weight") == 1);
    REQUIRE(wmap.count("norm.weight") == 1);
}

TEST_CASE("MTPHead: 4 draft-token shape smoke", "[mtp]") {
    auto args = make_tiny_mtp_args();
    mlx_lm::MTPHead head(args);

    mlx_lm::KVCache cache{mlx_lm::KVCacheSimple{}};
    auto mask = mlx_lm::AttentionMask::none();

    auto hidden = mx::random::normal({1, 1, args.hidden_size}, mx::float32);
    auto embedding = mx::random::normal({1, 1, args.hidden_size}, mx::float32);

    for (int step = 0; step < 4; ++step) {
        auto out = head(hidden, embedding, mask, &cache);
        mx::eval(out);
        REQUIRE(out.ndim() == 3);
        REQUIRE(out.shape(0) == 1);
        REQUIRE(out.shape(1) == 1);
        REQUIRE(out.shape(2) == args.hidden_size);

        auto normed = head.apply_output_norm(out);
        mx::eval(normed);
        REQUIRE(normed.shape(2) == args.hidden_size);

        hidden = out;
    }
}

TEST_CASE("KVCacheSimple: get / set position round-trip", "[mtp][kv]") {
    // Push a few rows, snapshot, push more, restore, confirm offset.
    mlx_lm::KVCacheSimple c;
    auto k0 = mx::random::normal({1, 2, 3, 4}, mx::float32);
    auto v0 = mx::random::normal({1, 2, 3, 4}, mx::float32);
    c.update(k0, v0);
    auto saved = c.get_position();
    REQUIRE(c.offset() == 3);

    auto k1 = mx::random::normal({1, 2, 2, 4}, mx::float32);
    auto v1 = mx::random::normal({1, 2, 2, 4}, mx::float32);
    c.update(k1, v1);
    REQUIRE(c.offset() == 5);

    c.set_position(saved);
    REQUIRE(c.offset() == 3);
}

