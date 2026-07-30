// Unit tests for generation loop brake (no model load).
#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/loop_brake.h>

#include <string>
#include <vector>

using namespace mlx_lm;

namespace {

std::string repeat(const std::string& s, int n) {
    std::string out;
    out.reserve(s.size() * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out += s;
    }
    return out;
}

} // namespace

TEST_CASE("loop brake empty / short text is no-op", "[loop_brake]") {
    CHECK_FALSE(has_generation_loop(""));
    CHECK_FALSE(has_generation_loop("The capital of France is Paris."));
    CHECK_FALSE(has_generation_loop(repeat("abc", 20))); // short phrase < 40 chars
}

TEST_CASE("loop brake consecutive char phrase >= 4 trips", "[loop_brake]") {
    // ~44-char phrase (within 40–80), repeated 4 times packed.
    const std::string phrase =
        "Fact Check: The capital of France is Paris. "; // 44 chars
    REQUIRE(phrase.size() >= 32);
    REQUIRE(phrase.size() <= 120);

    CHECK_FALSE(has_generation_loop(repeat(phrase, 3)));
    CHECK(has_generation_loop(repeat(phrase, 4)));
    CHECK(has_generation_loop("prefix noise " + repeat(phrase, 4)));
}

TEST_CASE("loop brake streaming feed_text trips on phrase", "[loop_brake]") {
    const std::string phrase =
        "Wait, actually, there is a specific type of radar array. "; // ~56
    REQUIRE(phrase.size() >= 40);

    LoopBrake brake;
    const std::string full = repeat(phrase, 4);
    bool tripped = false;
    for (std::size_t i = 0; i < full.size(); i += 7) {
        auto chunk = std::string_view(full).substr(
            i, std::min<std::size_t>(7, full.size() - i));
        if (brake.feed_text(chunk)) {
            tripped = true;
            break;
        }
    }
    CHECK(tripped);
    CHECK(brake.reason() == LoopBrakeReason::PhraseChars);
}

TEST_CASE("loop brake same line >= 6 trips", "[loop_brake]") {
    // Disable char-phrase path so we isolate SameLine (line+\\n is a packed
    // phrase of length line.size()+1 and would trip PhraseChars at 4 repeats).
    LoopBrakeParams p;
    p.min_phrase_chars = 10000;
    p.max_phrase_chars = 10000;

    const std::string line =
        "    *   Fact Check: The capital of France is Paris.";
    REQUIRE(line.size() >= 20);

    std::string text;
    for (int i = 0; i < 6; ++i) {
        text += line;
        text += '\n';
    }
    CHECK(has_generation_loop(text, p));

    LoopBrake brake(p);
    for (int i = 0; i < 5; ++i) {
        CHECK_FALSE(brake.feed_text(line + "\n"));
    }
    CHECK(brake.feed_text(line + "\n"));
    CHECK(brake.reason() == LoopBrakeReason::SameLine);
}

TEST_CASE("loop brake alternating long lines does not trip same-line early",
          "[loop_brake]") {
    const std::string a =
        "    *   Wait, I need to check the museum in the capital city.";
    const std::string b =
        "    *   Actually, I need to check the museum in capital city.";
    LoopBrake brake;
    // 5 pairs alternate — never 6 consecutive same lines.
    for (int i = 0; i < 5; ++i) {
        CHECK_FALSE(brake.feed_text(a + "\n"));
        CHECK_FALSE(brake.feed_text(b + "\n"));
    }
    CHECK_FALSE(brake.tripped());
}

TEST_CASE("loop brake token n-gram phrase >= 4 trips", "[loop_brake]") {
    std::vector<int> phrase = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    REQUIRE(phrase.size() >= 8);
    REQUIRE(phrase.size() <= 16);

    LoopBrake brake;
    for (int r = 0; r < 3; ++r) {
        for (int t : phrase) {
            CHECK_FALSE(brake.feed_token(t));
        }
    }
    for (std::size_t i = 0; i < phrase.size(); ++i) {
        bool trip = brake.feed_token(phrase[i]);
        if (i + 1 == phrase.size()) {
            CHECK(trip);
            CHECK(brake.reason() == LoopBrakeReason::PhraseTokens);
        } else {
            CHECK_FALSE(trip);
        }
    }
}

TEST_CASE("loop brake does not trip on normal prose", "[loop_brake]") {
    const std::string prose =
        "Thinking Process:\n"
        "1. Analyze the request: the user asks for the capital of France.\n"
        "2. Retrieve knowledge: the capital city of France is Paris.\n"
        "3. Formulate answer: state the answer clearly.\n"
        "4. Final check: ensure accuracy and tone.\n"
        "\n"
        "The capital of France is Paris.\n";
    CHECK_FALSE(has_generation_loop(prose));

    LoopBrake brake;
    CHECK_FALSE(brake.feed_text(prose));
    for (int i = 0; i < 64; ++i) {
        CHECK_FALSE(brake.feed_token(1000 + i));
    }
}

TEST_CASE("loop brake feed combines token and text", "[loop_brake]") {
    LoopBrake brake;
    const std::string phrase =
        "Fact Check: The capital of France is Paris!! "; // 44
    const std::string full = repeat(phrase, 4);
    bool tripped = false;
    int tok = 1;
    for (std::size_t i = 0; i < full.size(); ++i) {
        if (brake.feed(tok++, std::string_view(full).substr(i, 1))) {
            tripped = true;
            break;
        }
    }
    CHECK(tripped);
}

TEST_CASE("loop brake ignores pure punctuation stacks", "[loop_brake]") {
    CHECK_FALSE(has_generation_loop(repeat("----", 40)));
}
