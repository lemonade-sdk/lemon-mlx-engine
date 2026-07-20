// Copyright (C) 2024-2025 — thinking budget policy for CoT / enable_thinking
#pragma once

#include <optional>

namespace mlx_lm {

/// Soft target when thinking/CoT is on. Clients that send lower max_tokens
/// often hit finish_reason=length with little final answer (Discord UX).
inline constexpr int kThinkingBudgetRecommend = 4096;

/// When thinking is on and max_tokens is missing or below the recommend floor,
/// raise it to kThinkingBudgetRecommend. Never lowers an explicit higher budget.
/// Returns true if max_tokens was changed.
inline bool apply_thinking_budget_floor(std::optional<int>& max_tokens,
                                        bool thinking_on) {
    if (!thinking_on) {
        return false;
    }
    const int current = max_tokens.value_or(0);
    // Missing budget (nullopt) or below floor → raise.
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
