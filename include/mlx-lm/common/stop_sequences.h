// Copyright (C) 2024-2025 — OpenAI-compatible stop string matching
#pragma once

#include <string>
#include <vector>

namespace mlx_lm {

/// If `accumulated` ends with any stop string, strip that suffix and return true.
/// Empty stops ignored; first match in list order wins. Does not mutate `stops`.
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
