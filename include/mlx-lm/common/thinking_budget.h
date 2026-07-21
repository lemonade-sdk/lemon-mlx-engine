// Copyright (C) 2024-2025 — thinking budget policy for CoT / enable_thinking
#pragma once

#include <optional>

namespace mlx_lm {

/// Default max_tokens when thinking is on and the optional budget is unset.
inline constexpr int kThinkingBudgetRecommend = 4096;

/// If thinking_on and max_tokens is nullopt, set kThinkingBudgetRecommend.
/// Explicit values are never overwritten. Returns true if changed.
inline bool apply_thinking_budget_floor(std::optional<int>& max_tokens,
                                        bool thinking_on) {
    if (!thinking_on) {
        return false;
    }
    if (max_tokens.has_value()) {
        return false;
    }
    max_tokens = kThinkingBudgetRecommend;
    return true;
}

} // namespace mlx_lm
