// Copyright (C) 2024-2025 — thinking budget policy for CoT / enable_thinking
#pragma once

#include <optional>

namespace mlx_lm {

/// Soft floor for thinking/CoT (raise missing or low max_tokens to this).
inline constexpr int kThinkingBudgetRecommend = 4096;

/// If thinking_on and max_tokens is nullopt or below floor, set floor.
/// Never lowers a higher explicit budget. Returns true if changed.
inline bool apply_thinking_budget_floor(std::optional<int>& max_tokens,
                                        bool thinking_on) {
    if (!thinking_on) {
        return false;
    }
    const int current = max_tokens.value_or(0);
    if (max_tokens.has_value() && current >= kThinkingBudgetRecommend) {
        return false;
    }
    if (!max_tokens.has_value() || current < kThinkingBudgetRecommend) {
        max_tokens = kThinkingBudgetRecommend;
        return true;
    }
    return false;
}

} // namespace mlx_lm
