// Tests for ChatSession: construction, state management, history, parameters
// Ported from swift/Tests/MLXLMTests/ChatSessionTests.swift
//
// Note: The Swift ChatSessionTests require a real model (Gemma3Text) for
// generate/stream tests. Those are skipped here. We test the parts of
// ChatSession that can be exercised without model inference: construction,
// parameter management, message building, history re-hydration, and clear.

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/chat_session.h>
#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/types.h>
#include <mlx/mlx.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace mlx_lm;
namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Helper: create a minimal ModelContainer with stub functions.
// These stubs are sufficient for testing ChatSession construction and state
// but will NOT produce real model output.
// ---------------------------------------------------------------------------
static std::shared_ptr<ModelContainer> make_stub_container() {
    ModelContext ctx;
    ctx.model_id = "test-stub-model";
    ctx.eos_token_ids = std::vector<int>{2};

    // Stub functions -- they should never be called in these unit tests.
    // If they are called, they will throw, which is fine for testing.
    ctx.encode_fn = [](const std::string&) -> std::vector<int> {
        throw std::runtime_error("stub encode_fn called");
    };
    ctx.decode_fn = [](const std::vector<int>&) -> std::string {
        throw std::runtime_error("stub decode_fn called");
    };
    ctx.apply_chat_template_fn =
        [](const std::vector<std::unordered_map<std::string, std::string>>&,
           const nlohmann::json*) -> std::vector<int> {
        throw std::runtime_error("stub apply_chat_template_fn called");
    };
    ctx.new_cache_fn = [](const GenerateParameters&) -> std::vector<KVCache> {
        return {};
    };

    return std::make_shared<ModelContainer>(std::move(ctx));
}

// ===========================================================================
// Construction Tests
// ===========================================================================

TEST_CASE("ChatSession default construction", "[chat_session]") {
    auto container = make_stub_container();
    ChatSession session(container);

    CHECK(session.message_history().empty());
    CHECK(!session.instructions().has_value());
    CHECK(session.generate_parameters().temperature == 0.6f);
    CHECK(session.generate_parameters().top_p == 1.0f);
}

TEST_CASE("ChatSession construction with instructions", "[chat_session]") {
    auto container = make_stub_container();
    ChatSession session(container, "You are a helpful assistant.");

    CHECK(session.instructions().has_value());
    CHECK(session.instructions().value() == "You are a helpful assistant.");
    CHECK(session.message_history().empty());
}

TEST_CASE("ChatSession construction with custom generate params", "[chat_session]") {
    auto container = make_stub_container();

    GenerateParameters params;
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.max_tokens = 256;

    ChatSession session(container, std::nullopt, params);

    CHECK(session.generate_parameters().temperature == 0.8f);
    CHECK(session.generate_parameters().top_p == 0.95f);
    REQUIRE(session.generate_parameters().max_tokens.has_value());
    CHECK(session.generate_parameters().max_tokens.value() == 256);
}

TEST_CASE("ChatSession construction with history re-hydration", "[chat_session]") {
    auto container = make_stub_container();

    std::vector<chat::ChatMessage> history = {
        chat::ChatMessage::user("What is the capital of France?"),
        chat::ChatMessage::assistant("The capital of France is Paris."),
        chat::ChatMessage::user("What about Germany?"),
        chat::ChatMessage::assistant("The capital of Germany is Berlin."),
    };

    ChatSession session(container, std::move(history), "You are a geography expert.");

    CHECK(session.instructions().has_value());
    CHECK(session.instructions().value() == "You are a geography expert.");
    // After construction with history, message_history reflects that history.
    // (The exact behavior depends on implementation -- it may be in pending_history_
    // until the first generation call, but message_history() should still return it.)
}

// ===========================================================================
// Parameter Management Tests
// ===========================================================================

TEST_CASE("ChatSession set and clear instructions", "[chat_session]") {
    auto container = make_stub_container();
    ChatSession session(container);

    CHECK(!session.instructions().has_value());

    session.set_instructions("Be concise.");
    REQUIRE(session.instructions().has_value());
    CHECK(session.instructions().value() == "Be concise.");

    session.clear_instructions();
    CHECK(!session.instructions().has_value());
}

TEST_CASE("ChatSession set generate parameters", "[chat_session]") {
    auto container = make_stub_container();
    ChatSession session(container);

    // Verify defaults
    CHECK(session.generate_parameters().temperature == 0.6f);
    CHECK(session.generate_parameters().prefill_step_size == 512);

    // Update parameters
    GenerateParameters new_params;
    new_params.temperature = 0.0f;
    new_params.prefill_step_size = 1024;
    new_params.max_tokens = 100;
    session.set_generate_parameters(new_params);

    CHECK(session.generate_parameters().temperature == 0.0f);
    CHECK(session.generate_parameters().prefill_step_size == 1024);
    REQUIRE(session.generate_parameters().max_tokens.has_value());
    CHECK(session.generate_parameters().max_tokens.value() == 100);
}

// ===========================================================================
// Clear Tests
// ===========================================================================

TEST_CASE("ChatSession clear resets history", "[chat_session]") {
    auto container = make_stub_container();
    ChatSession session(container, "System prompt.");

    // Clear should reset history but preserve instructions
    session.clear();
    CHECK(session.message_history().empty());
    // Instructions should be preserved after clear
    CHECK(session.instructions().has_value());
    CHECK(session.instructions().value() == "System prompt.");
}

// ===========================================================================
// Move Semantics Tests
// ===========================================================================

TEST_CASE("ChatSession is movable", "[chat_session]") {
    auto container = make_stub_container();

    ChatSession session1(container, "Instructions.");

    GenerateParameters params;
    params.temperature = 0.9f;
    session1.set_generate_parameters(params);

    // Move construct
    ChatSession session2(std::move(session1));

    CHECK(session2.instructions().has_value());
    CHECK(session2.instructions().value() == "Instructions.");
    CHECK(session2.generate_parameters().temperature == 0.9f);
}

// ===========================================================================
// Chat Message Helpers Tests
// ===========================================================================

TEST_CASE("ChatMessage factory helpers", "[chat_session]") {
    auto user_msg = chat::ChatMessage::user("hello");
    CHECK(user_msg.role == chat::Role::User);
    CHECK(user_msg.content == "hello");

    auto asst_msg = chat::ChatMessage::assistant("hi there");
    CHECK(asst_msg.role == chat::Role::Assistant);
    CHECK(asst_msg.content == "hi there");

    auto sys_msg = chat::ChatMessage::system("You are helpful.");
    CHECK(sys_msg.role == chat::Role::System);
    CHECK(sys_msg.content == "You are helpful.");

    auto tool_msg = chat::ChatMessage::tool("{\"result\": 42}");
    CHECK(tool_msg.role == chat::Role::Tool);
    CHECK(tool_msg.content == "{\"result\": 42}");
}

TEST_CASE("Role string conversion round-trip", "[chat_session]") {
    CHECK(chat::role_to_string(chat::Role::User) == "user");
    CHECK(chat::role_to_string(chat::Role::Assistant) == "assistant");
    CHECK(chat::role_to_string(chat::Role::System) == "system");
    CHECK(chat::role_to_string(chat::Role::Tool) == "tool");

    CHECK(chat::role_from_string("user") == chat::Role::User);
    CHECK(chat::role_from_string("assistant") == chat::Role::Assistant);
    CHECK(chat::role_from_string("system") == chat::Role::System);
    CHECK(chat::role_from_string("tool") == chat::Role::Tool);

    // Unknown roles default to User
    CHECK(chat::role_from_string("unknown") == chat::Role::User);
    CHECK(chat::role_from_string("") == chat::Role::User);
}

// ===========================================================================
// DefaultMessageGenerator Tests
// ===========================================================================

TEST_CASE("DefaultMessageGenerator produces role+content maps", "[chat_session]") {
    DefaultMessageGenerator gen;

    auto msg = gen.generate(chat::ChatMessage::user("hello"));
    CHECK(msg.at("role") == "user");
    CHECK(msg.at("content") == "hello");

    auto msgs = gen.generate({
        chat::ChatMessage::system("sys"),
        chat::ChatMessage::user("usr"),
        chat::ChatMessage::assistant("asst"),
    });
    REQUIRE(msgs.size() == 3);
    CHECK(msgs[0].at("role") == "system");
    CHECK(msgs[1].at("role") == "user");
    CHECK(msgs[2].at("role") == "assistant");
}

TEST_CASE("NoSystemMessageGenerator omits system messages", "[chat_session]") {
    NoSystemMessageGenerator gen;

    auto msgs = gen.generate({
        chat::ChatMessage::system("sys"),
        chat::ChatMessage::user("usr"),
        chat::ChatMessage::assistant("asst"),
    });
    REQUIRE(msgs.size() == 2);
    CHECK(msgs[0].at("role") == "user");
    CHECK(msgs[1].at("role") == "assistant");
}

// ===========================================================================
// GenerateParameters Defaults Tests
// ===========================================================================

TEST_CASE("GenerateParameters defaults", "[chat_session]") {
    GenerateParameters params;
    CHECK(params.temperature == 0.6f);
    CHECK(params.top_p == 1.0f);
    CHECK(params.prefill_step_size == 512);
    CHECK(!params.max_tokens.has_value());
    CHECK(!params.max_kv_size.has_value());
    CHECK(!params.kv_bits.has_value());
    CHECK(params.kv_group_size == 64);
    CHECK(params.quantized_kv_start == 0);
    CHECK(!params.repetition_penalty.has_value());
    CHECK(params.repetition_context_size == 20);
}

// ===========================================================================
// Multi-turn prefill regression (double-prefill guard)
// ===========================================================================
//
// Old bug: turn 2+ re-used non-empty KV while (mis)templating only the new
// user turn — double-prefill / wrong framing. Fix: every turn templates the
// full history into a FRESH cache. This test fails if:
//   - turn2 prefill token count <= turn1 (history not re-included), or
//   - new_cache_fn is not called again on turn2 (residual KV reuse).

TEST_CASE("ChatSession multi-turn full re-prefill uses fresh cache",
        "[chat_session][multi_turn]") {
    constexpr int kVocabSize = 8;
    constexpr int kForcedToken = 3;  // non-EOS
    constexpr int kEosToken = 2;

    int cache_creates = 0;
    std::vector<int> prefill_token_counts;
    std::vector<size_t> template_message_counts;

    ModelContext ctx;
    ctx.model_id = "fake-chat-session";
    ctx.eos_token_ids = std::vector<int>{kEosToken};

    ctx.new_cache_fn = [&](const GenerateParameters&) {
        cache_creates++;
        return std::vector<KVCache>{};
    };
    ctx.prepare_fn = [&](const LMInput& input, std::vector<KVCache>&, int) {
        // tokens are 1D [seq]
        prefill_token_counts.push_back(static_cast<int>(input.text.tokens.size()));
        return PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [](const LMInput::Text&, std::vector<KVCache>*,
                     const LMOutput::State*) {
        // First decode step: emit a visible token; second would be EOS if we
        // kept going — max_tokens=1 stops after one generated token.
        std::vector<float> logits(kVocabSize, 0.0f);
        logits[kForcedToken] = 10.0f;
        return LMOutput(mx::array(logits.data(), {1, 1, kVocabSize}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>& tokens) -> std::string {
        return std::string(tokens.size(), 'x');
    };
    ctx.apply_chat_template_fn =
        [&](const std::vector<std::unordered_map<std::string, std::string>>& messages,
            const nlohmann::json*) -> std::vector<int> {
        template_message_counts.push_back(messages.size());
        // Deterministic length: 10 tokens per message + 2 framing.
        const int n = static_cast<int>(messages.size()) * 10 + 2;
        std::vector<int> toks(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            toks[static_cast<size_t>(i)] = 100 + i;
        }
        return toks;
    };

    auto container = std::make_shared<ModelContainer>(std::move(ctx));

    GenerateParameters params;
    params.temperature = 0.0f;
    params.max_tokens = 1;

    ChatSession session(container, std::nullopt, params);

    // Turn 1: single user message → template sees 1 msg → 12 tokens.
    (void)session.respond("My name is Ada.");
    REQUIRE(template_message_counts.size() == 1);
    CHECK(template_message_counts[0] == 1);
    REQUIRE(prefill_token_counts.size() == 1);
    CHECK(prefill_token_counts[0] == 12);
    const int caches_after_turn1 = cache_creates;
    REQUIRE(caches_after_turn1 >= 1);

    // Turn 2: history has user+assistant, plus new user → 3 messages → 32 tokens.
    (void)session.respond("What is my name?");
    REQUIRE(template_message_counts.size() == 2);
    CHECK(template_message_counts[1] == 3);
    REQUIRE(prefill_token_counts.size() == 2);
    CHECK(prefill_token_counts[1] == 32);
    // Fresh cache allocated for turn 2 (not residual reuse only).
    CHECK(cache_creates > caches_after_turn1);

    // History recorded both turns.
    REQUIRE(session.message_history().size() == 4);
    CHECK(session.message_history()[0].content == "My name is Ada.");
    CHECK(session.message_history()[2].content == "What is my name?");
}

TEST_CASE("ChatSession history re-hydration folds into messages_ for later turns",
          "[chat_session][multi_turn][rehydrate]") {
    constexpr int kVocabSize = 8;
    constexpr int kForcedToken = 3;
    constexpr int kEosToken = 2;

    int cache_creates = 0;
    std::vector<size_t> template_message_counts;

    ModelContext ctx;
    ctx.model_id = "fake-rehydrate";
    ctx.eos_token_ids = std::vector<int>{kEosToken};
    ctx.new_cache_fn = [&](const GenerateParameters&) {
        cache_creates++;
        return std::vector<KVCache>{};
    };
    ctx.prepare_fn = [](const LMInput& input, std::vector<KVCache>&, int) {
        return PrepareResult::tokens(input.text);
    };
    ctx.call_fn = [](const LMInput::Text&, std::vector<KVCache>*,
                     const LMOutput::State*) {
        std::vector<float> logits(kVocabSize, 0.0f);
        logits[kForcedToken] = 10.0f;
        return LMOutput(mx::array(logits.data(), {1, 1, kVocabSize}, mx::float32));
    };
    ctx.decode_fn = [](const std::vector<int>& tokens) -> std::string {
        return std::string(tokens.size(), 'y');
    };
    ctx.apply_chat_template_fn =
        [&](const std::vector<std::unordered_map<std::string, std::string>>& messages,
            const nlohmann::json*) -> std::vector<int> {
        template_message_counts.push_back(messages.size());
        return std::vector<int>(messages.size() * 10 + 2, 1);
    };

    std::vector<chat::ChatMessage> history = {
        chat::ChatMessage::user("Capital of France?"),
        chat::ChatMessage::assistant("Paris."),
    };

    auto container = std::make_shared<ModelContainer>(std::move(ctx));
    GenerateParameters params;
    params.temperature = 0.0f;
    params.max_tokens = 1;
    ChatSession session(container, std::move(history), std::nullopt, params);

    // Visible before first generate (pending history).
    REQUIRE(session.message_history().size() == 2);
    CHECK(session.message_history()[0].content == "Capital of France?");

    // Turn 1: system none + 2 history + new user = 3 messages.
    (void)session.respond("And Germany?");
    REQUIRE(template_message_counts.size() == 1);
    CHECK(template_message_counts[0] == 3);

    // After turn1: history folded + new user/assistant → 4 messages.
    REQUIRE(session.message_history().size() == 4);
    CHECK(session.message_history()[0].content == "Capital of France?");
    CHECK(session.message_history()[2].content == "And Germany?");

    // Turn 2 must still see original history (not drop it).
    // messages_: 4 prior + new user = 5 templated messages.
    (void)session.respond("Summarize in one word.");
    REQUIRE(template_message_counts.size() == 2);
    CHECK(template_message_counts[1] == 5);
    REQUIRE(session.message_history().size() == 6);
    CHECK(session.message_history()[0].content == "Capital of France?");
    CHECK(cache_creates >= 2);
}
