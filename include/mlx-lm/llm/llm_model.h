// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/language_model.h>
#include <mlx-lm/common/types.h>
#include <mlx/mlx.h>
#include <algorithm>
#include <cstdlib>
#include <vector>

namespace mlx_lm {

// Default prepare step for LLM models.
// Evaluates the prompt in chunks until there is a small number
// of tokens left to feed into the TokenIterator.
//
// This is a free function that any LLM model can call from its
// prepare_impl.
//
// MLX_PREFILL_ABSORB_TAIL=1 (opt-in, experiment only): keep chunking while
// more than one token remains, taking min(step, n-1) each time so all T>1
// multi-token prefill runs inside prepare (under any prepare-time HIP graph
// mode). Leaves a single token for TokenIterator::step first-token sample.
// Default OFF — product keeps classic "leave remainder ≤ step for step()".
template <typename Model>
PrepareResult llm_default_prepare(
    Model& model,
    const LMInput& input,
    std::vector<KVCache>& cache,
    int window_size)
{
    int prefill_step_size = (window_size > 0) ? window_size : 512;
    auto text = input.text;

    static const bool absorb_tail = [] {
        const char* e = std::getenv("MLX_PREFILL_ABSORB_TAIL");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    // Classic: stop when remaining ≤ step (remainder may still be multi-token).
    // Absorb: stop only when a single token remains for first-token sampling.
    const int stop_above = absorb_tail ? 1 : prefill_step_size;

    // Prepare the prompt in chunks if larger than the prefill size.
    // Tokens are 1D [seq_len]; add batch dim [1, seq_len] for model calls.
    while (text.tokens.shape(0) > stop_above) {
        const int n = text.tokens.shape(0);
        int take = prefill_step_size;
        if (absorb_tail) {
            // Never consume the final token here — leave it for step().
            take = std::min(prefill_step_size, n - 1);
            if (take <= 0) {
                break;
            }
        }

        auto chunk_tokens = mlx::core::slice(
            text.tokens,
            {0},
            {take});

        // Add batch dimension for model call (matches Swift's newAxis)
        LMInput::Text chunk_text(mlx::core::expand_dims(chunk_tokens, 0));
        model(chunk_text, &cache, nullptr);

        // Eval the actual cache state arrays so the GPU materializes
        // the forward pass. Matches Python's mx.eval([c.state for c in cache]).
        // Without this, the computation graph keeps growing across chunks.
        {
            std::vector<mlx::core::array> to_eval;
            for (auto& c : cache) {
                auto s = c.state();
                to_eval.insert(to_eval.end(), s.begin(), s.end());
            }
            mlx::core::eval(to_eval);
        }
        mlx::core::clear_cache();

        text.tokens = mlx::core::slice(
            text.tokens,
            {take},
            {text.tokens.shape(0)});
    }

    return PrepareResult::tokens(std::move(text));
}

} // namespace mlx_lm
