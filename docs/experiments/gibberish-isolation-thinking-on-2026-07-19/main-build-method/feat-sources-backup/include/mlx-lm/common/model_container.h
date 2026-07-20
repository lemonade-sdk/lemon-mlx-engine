// Copyright © 2024-2025 Apple Inc. — Ported to C++
#pragma once

#include <mlx-lm/common/generate_params.h>
#include <mlx-lm/common/types.h>
#include <nlohmann/json.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mlx_lm {

// Forward declarations
class KVCache;

// ModelContext holds the model, tokenizer, and processor together.
// Template-free — uses type-erased function objects for model operations.
struct ModelContext {
    // Model operations (type-erased via std::function to avoid virtuals).
    std::function<PrepareResult(const LMInput&, std::vector<KVCache>&, int)> prepare_fn;
    std::function<LMOutput(const LMInput::Text&, std::vector<KVCache>*, const LMOutput::State*)> call_fn;
    std::function<mlx::core::array(const mlx::core::array&, std::vector<KVCache>*)> forward_fn;
    std::function<std::vector<KVCache>(const GenerateParameters&)> new_cache_fn;
    std::function<std::unordered_map<std::string, mlx::core::array>(
        std::unordered_map<std::string, mlx::core::array>)> sanitize_fn;

    // Token embedding lookup (for MTP speculative decoding).
    // Maps token IDs to embedding vectors [B, T, H].
    std::function<mlx::core::array(const mlx::core::array&)> embed_fn;

    // Apply lm_head to hidden states (for MTP speculative decoding).
    // For tied embeddings: matmul(x, embed_tokens.T).
    // For untied: linear_fwd(x, lm_head_weight).
    std::function<mlx::core::array(const mlx::core::array&)> apply_lm_head_fn;

    // MTP head access (returns nullptr if MTP not available).
    // These function pointers are set only when the model has MTP support.
    std::function<void*()> get_mtp_head_fn;  // Returns MTPHead*
    std::function<std::vector<KVCache>(const GenerateParameters&)> new_mtp_cache_fn;  // Returns single-layer KVCache

    // Tokenizer operations (type-erased).
    std::function<std::vector<int>(const std::string&)> encode_fn;
    std::function<std::string(const std::vector<int>&)> decode_fn;
    // messages + optional tools JSON array (nullptr = no tools injection).
    std::function<std::vector<int>(
        const std::vector<std::unordered_map<std::string, std::string>>&,
        const nlohmann::json* /*tools*/)> apply_chat_template_fn;

    // Configuration
    std::string model_id;
    // model_type from config.json (e.g. "qwen3") for tool-call format inference.
    std::string model_type;
    std::optional<std::vector<int>> eos_token_ids;

    // Extra context for chat template rendering (e.g., enable_thinking=false).
    // Shared with the apply_chat_template_fn lambda so mutations propagate.
    std::shared_ptr<nlohmann::json> template_extra_context;

    // Bind a concrete model into this context (non-owning reference).
    template <typename Model>
    static ModelContext from_model(Model& model) {
        ModelContext ctx;
        ctx.prepare_fn = [&model](const LMInput& input, std::vector<KVCache>& cache, int ws) {
            return model.prepare(input, cache, ws);
        };
        ctx.call_fn = [&model](const LMInput::Text& input, std::vector<KVCache>* cache,
                               const LMOutput::State* state) {
            return model(input, cache, state);
        };
        ctx.forward_fn = [&model](const mlx::core::array& inputs, std::vector<KVCache>* cache) {
            return model.forward(inputs, cache);
        };
        ctx.new_cache_fn = [&model](const GenerateParameters& params) {
            return model.new_cache(params);
        };
        ctx.sanitize_fn = [&model](std::unordered_map<std::string, mlx::core::array> w) {
            return model.sanitize(std::move(w));
        };
        if constexpr (requires { model.embed_as_linear(std::declval<mlx::core::array>()); }) {
            ctx.embed_fn = [&model](const mlx::core::array& tokens) {
                return model.embed_as_linear(tokens);
            };
        }
        if constexpr (requires { model.apply_lm_head(std::declval<mlx::core::array>()); }) {
            ctx.apply_lm_head_fn = [&model](const mlx::core::array& hidden) {
                return model.apply_lm_head(hidden);
            };
        }
        // MTP bindings (if model has MTP support).
        if constexpr (requires { model.get_mtp_head(); }) {
            ctx.get_mtp_head_fn = [&model]() -> void* {
                return static_cast<void*>(model.get_mtp_head());
            };
        }
        if constexpr (requires { model.new_mtp_cache(std::declval<const GenerateParameters&>()); }) {
            ctx.new_mtp_cache_fn = [&model](const GenerateParameters& p) {
                return model.new_mtp_cache(p);
            };
        }
        return ctx;
    }

    // Bind an owned model via shared_ptr (model lifetime tied to context).
    template <typename Model>
    static ModelContext from_model_owned(std::shared_ptr<Model> model) {
        ModelContext ctx;
        ctx.prepare_fn = [model](const LMInput& input, std::vector<KVCache>& cache, int ws) {
            return model->prepare(input, cache, ws);
        };
        ctx.call_fn = [model](const LMInput::Text& input, std::vector<KVCache>* cache,
                               const LMOutput::State* state) {
            return (*model)(input, cache, state);
        };
        ctx.forward_fn = [model](const mlx::core::array& inputs, std::vector<KVCache>* cache) {
            return model->forward(inputs, cache);
        };
        ctx.new_cache_fn = [model](const GenerateParameters& params) {
            return model->new_cache(params);
        };
        ctx.sanitize_fn = [model](std::unordered_map<std::string, mlx::core::array> w) {
            return model->sanitize(std::move(w));
        };
        if constexpr (requires { model->embed_as_linear(std::declval<mlx::core::array>()); }) {
            ctx.embed_fn = [model](const mlx::core::array& tokens) {
                return model->embed_as_linear(tokens);
            };
        }
        if constexpr (requires { model->apply_lm_head(std::declval<mlx::core::array>()); }) {
            ctx.apply_lm_head_fn = [model](const mlx::core::array& hidden) {
                return model->apply_lm_head(hidden);
            };
        }
        // MTP bindings (if model has MTP support).
        if constexpr (requires { model->get_mtp_head(); }) {
            ctx.get_mtp_head_fn = [model]() -> void* {
                return static_cast<void*>(model->get_mtp_head());
            };
        }
        if constexpr (requires { model->new_mtp_cache(std::declval<const GenerateParameters&>()); }) {
            ctx.new_mtp_cache_fn = [model](const GenerateParameters& p) {
                return model->new_mtp_cache(p);
            };
        }
        return ctx;
    }
};

// Thread-safe container for a ModelContext.
// Replaces Swift's actor-based ModelContainer.
class ModelContainer {
public:
    explicit ModelContainer(ModelContext context)
        : context_(std::make_shared<ModelContext>(std::move(context))) {}

    // Perform an action with exclusive access to the model context.
    template <typename Func>
    auto perform(Func&& action) -> decltype(action(std::declval<ModelContext&>())) {
        std::lock_guard<std::mutex> lock(mutex_);
        return action(*context_);
    }

    // Read-only access.
    template <typename Func>
    auto perform_read(Func&& action) const -> decltype(action(std::declval<const ModelContext&>())) {
        std::lock_guard<std::mutex> lock(mutex_);
        return action(*context_);
    }

    const std::string& model_id() const { return context_->model_id; }

private:
    std::shared_ptr<ModelContext> context_;
    mutable std::mutex mutex_;
};

} // namespace mlx_lm
