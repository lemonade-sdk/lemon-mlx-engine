// Copyright (C) 2024-2025 — OpenAI-compatible stop string matching
#pragma once

#include <string>
#include <vector>

namespace mlx_lm {

/// OpenAI `stop`: if `accumulated` ends with any stop string, strip that
/// suffix and return true (caller should halt generation).
///
/// Tier-1 behavior (suffix-only): matches after full detokenized text so far.
/// Empty stop strings are ignored. First matching stop wins (order in vector).
/// Never lowers or mutates the stop list.
inline bool apply_stop_sequences(std::string& accumulated,
                                 const std::vector<std::string>& stops) {
    if (stops.empty() || accumulated.empty()) {
        return false;
    }
    for (const auto& s : stops) {
        if (s.empty() || accumulated.size() < s.size()) {
            continue;
        }
        if (accumulated.compare(accumulated.size() - s.size(), s.size(), s) == 0) {
            accumulated.resize(accumulated.size() - s.size());
            return true;
        }
    }
    return false;
}

} // namespace mlx_lm
