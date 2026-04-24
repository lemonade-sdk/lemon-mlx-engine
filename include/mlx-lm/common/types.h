// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx/mlx.h>
#include <optional>
#include <variant>
#include <vector>

namespace mlx_lm {

// Time/Height/Width struct for input image dimensions.
struct THW {
    int t;
    int h;
    int w;

    THW(int t, int h, int w) : t(t), h(h), w(w) {}

    int product() const { return t * h * w; }
};

// Representation of LanguageModel input.
struct LMInput {

    // Tokenized input text.
    struct Text {
        mlx::core::array tokens;
        std::optional<mlx::core::array> mask;

        Text() = default;
        Text(mlx::core::array tokens, std::optional<mlx::core::array> mask = std::nullopt)
            : tokens(std::move(tokens)), mask(std::move(mask)) {}
    };

    // Prepared input image(s).
    struct ProcessedImage {
        mlx::core::array pixels;
        std::optional<std::vector<THW>> frames;

        ProcessedImage(mlx::core::array pixels, std::optional<std::vector<THW>> frames = std::nullopt)
            : pixels(std::move(pixels)), frames(std::move(frames)) {}
    };

    // Prepared input video(s).
    struct ProcessedVideo {
        mlx::core::array pixels;
        std::optional<std::vector<THW>> frames;

        ProcessedVideo(mlx::core::array pixels, std::optional<std::vector<THW>> frames = std::nullopt)
            : pixels(std::move(pixels)), frames(std::move(frames)) {}
    };

    Text text;
    std::optional<ProcessedImage> image;
    std::optional<ProcessedVideo> video;

    LMInput() = default;

    LMInput(mlx::core::array tokens, std::optional<mlx::core::array> mask = std::nullopt)
        : text(std::move(tokens), std::move(mask)) {}

    LMInput(Text text,
            std::optional<ProcessedImage> image = std::nullopt,
            std::optional<ProcessedVideo> video = std::nullopt)
        : text(std::move(text)), image(std::move(image)), video(std::move(video)) {}
};

// LanguageModel step output.
struct LMOutput {

    struct State {
        std::optional<mlx::core::array> cross_attention_states;

        State() = default;
        explicit State(mlx::core::array cas)
            : cross_attention_states(std::move(cas)) {}
    };

    mlx::core::array logits;
    std::optional<State> state;

    LMOutput(mlx::core::array logits, std::optional<State> state = std::nullopt)
        : logits(std::move(logits)), state(std::move(state)) {}
};

// Result of the call to LanguageModel::prepare().
struct PrepareResult {
    enum Tag { kTokens, kLogits };

    Tag tag;
    std::variant<LMInput::Text, LMOutput> value;

    static PrepareResult tokens(LMInput::Text text) {
        return {kTokens, std::move(text)};
    }

    static PrepareResult logits(LMOutput output) {
        return {kLogits, std::move(output)};
    }

    bool is_tokens() const { return tag == kTokens; }
    bool is_logits() const { return tag == kLogits; }

    const LMInput::Text& as_tokens() const { return std::get<LMInput::Text>(value); }
    const LMOutput& as_logits() const { return std::get<LMOutput>(value); }

    LMInput::Text& as_tokens() { return std::get<LMInput::Text>(value); }
    LMOutput& as_logits() { return std::get<LMOutput>(value); }
};

} // namespace mlx_lm
