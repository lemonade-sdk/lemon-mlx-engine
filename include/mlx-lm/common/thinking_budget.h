// Copyright (C) 2024-2025 — thinking budget policy for CoT / enable_thinking
#pragma once

#include <optional>

namespace mlx_lm {

/// Soft target when thinking/CoT is on and the client omitted max_tokens.
/// Explicit client max_tokens is never overwritten.
inline constexpr int kThinkingBudgetRecommend = 4096;

/// When thinking is on and max_tokens is unset, set it to the recommend floor.
/// Returns true if max_tokens was changed. Explicit budgets always win.
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
