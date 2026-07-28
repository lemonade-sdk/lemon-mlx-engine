// OpenAI stop suffix matching (no model load).
#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/stop_sequences.h>

#include <string>
#include <vector>

using namespace mlx_lm;

TEST_CASE("stop sequences empty list is no-op", "[stop_sequences]") {
    std::string text = "hello world";
    CHECK_FALSE(apply_stop_sequences(text, {}));
    CHECK(text == "hello world");
}

TEST_CASE("stop sequences empty accumulated is no-op", "[stop_sequences]") {
    std::string text;
    CHECK_FALSE(apply_stop_sequences(text, {"END"}));
    CHECK(text.empty());
}

TEST_CASE("stop sequences ignores empty stop string", "[stop_sequences]") {
    std::string text = "hello";
    CHECK_FALSE(apply_stop_sequences(text, {"", ""}));
    CHECK(text == "hello");
}

TEST_CASE("stop sequences strips suffix match", "[stop_sequences]") {
    std::string text = "count 1,2,3,4,5";
    CHECK(apply_stop_sequences(text, {"5"}));
    CHECK(text == "count 1,2,3,4,");
}

TEST_CASE("stop sequences multi-candidate first match wins", "[stop_sequences]") {
    std::string text = "foo###END###";
    // List order: first suffix match wins ("END###" before "###END###").
    CHECK(apply_stop_sequences(text, {"END###", "###END###"}));
    CHECK(text == "foo###");
}

TEST_CASE("stop sequences multi-candidate second if first not suffix",
          "[stop_sequences]") {
    std::string text = "answer###STOP###";
    CHECK(apply_stop_sequences(text, {"NOPE", "###STOP###"}));
    CHECK(text == "answer");
}

TEST_CASE("stop sequences no match leaves text", "[stop_sequences]") {
    std::string text = "still going";
    CHECK_FALSE(apply_stop_sequences(text, {"###"}));
    CHECK(text == "still going");
}

TEST_CASE("stop sequences stop longer than text ignored", "[stop_sequences]") {
    std::string text = "hi";
    CHECK_FALSE(apply_stop_sequences(text, {"longer-than-text"}));
    CHECK(text == "hi");
}
