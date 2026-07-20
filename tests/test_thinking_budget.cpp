// Unit tests for thinking budget floor (no model load).
#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/thinking_budget.h>

#include <optional>

using namespace mlx_lm;

TEST_CASE("thinking budget floor no-op when thinking off", "[thinking_budget]") {
    std::optional<int> m = 256;
    CHECK_FALSE(apply_thinking_budget_floor(m, /*thinking_on=*/false));
    REQUIRE(m.has_value());
    CHECK(*m == 256);
}

TEST_CASE("thinking budget floor leaves explicit low budget alone",
          "[thinking_budget]") {
    std::optional<int> m = 5;
    CHECK_FALSE(apply_thinking_budget_floor(m, /*thinking_on=*/true));
    REQUIRE(m.has_value());
    CHECK(*m == 5);
}

TEST_CASE("thinking budget floor raises missing budget when thinking on",
          "[thinking_budget]") {
    std::optional<int> m;
    CHECK(apply_thinking_budget_floor(m, /*thinking_on=*/true));
    REQUIRE(m.has_value());
    CHECK(*m == kThinkingBudgetRecommend);
}

TEST_CASE("thinking budget floor leaves high budget alone", "[thinking_budget]") {
    std::optional<int> m = 8192;
    CHECK_FALSE(apply_thinking_budget_floor(m, /*thinking_on=*/true));
    REQUIRE(m.has_value());
    CHECK(*m == 8192);
}

TEST_CASE("thinking budget floor leaves exact recommend alone",
          "[thinking_budget]") {
    std::optional<int> m = kThinkingBudgetRecommend;
    CHECK_FALSE(apply_thinking_budget_floor(m, /*thinking_on=*/true));
    REQUIRE(m.has_value());
    CHECK(*m == kThinkingBudgetRecommend);
}

TEST_CASE("thinking budget floor no-op on nullopt when thinking off",
          "[thinking_budget]") {
    std::optional<int> m;
    CHECK_FALSE(apply_thinking_budget_floor(m, /*thinking_on=*/false));
    CHECK_FALSE(m.has_value());
}
