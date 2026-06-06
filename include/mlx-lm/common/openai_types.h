// OpenAI-compatible API types with automatic JSON serialization.
// Uses NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT for struct <-> JSON.
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace mlx_lm {
namespace openai {

// ---------------------------------------------------------------------------
// std::optional<T> adapter for nlohmann/json
// Serializes as value or null, deserializes from value/null/missing.
// ---------------------------------------------------------------------------
template <typename T>
void optional_to_json(nlohmann::json& j, const char* key, const std::optional<T>& opt) {
    if (opt.has_value()) {
        j[key] = *opt;
    }
}

template <typename T>
void optional_from_json(const nlohmann::json& j, const char* key, std::optional<T>& opt) {
    if (j.contains(key) && !j.at(key).is_null()) {
        opt = j.at(key).get<T>();
    } else {
        opt = std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
struct Usage {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Usage,
    prompt_tokens, completion_tokens, total_tokens)

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------
struct ErrorDetail {
    std::string message;
    std::string type = "invalid_request_error";
    std::string code;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ErrorDetail,
    message, type, code)

struct ErrorResponse {
    ErrorDetail error;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ErrorResponse, error)

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
struct ModelObject {
    std::string id;
    std::string object = "model";
    int64_t created = 0;
    std::string owned_by = "local";
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ModelObject,
    id, object, created, owned_by)

struct ModelList {
    std::string object = "list";
    std::vector<ModelObject> data;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ModelList, object, data)

// ---------------------------------------------------------------------------
// Chat Completions — Request
// ---------------------------------------------------------------------------
struct ChatMessage {
    std::string role;
    std::string content;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatMessage, role, content)

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 0.6f;
    float top_p = 1.0f;
    int max_tokens = 2048;
    float repetition_penalty = 0.0f;
    bool stream = false;
    bool use_mtp = false;
    // stop sequences (optional, parsed manually)
    std::vector<std::string> stop;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionRequest,
    model, messages, temperature, top_p, max_tokens,
    repetition_penalty, stream, use_mtp, stop)

// ---------------------------------------------------------------------------
// Chat Completions — Response (non-streaming)
// ---------------------------------------------------------------------------
struct ChatCompletionChoiceMessage {
    std::string role = "assistant";
    std::string content;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionChoiceMessage,
    role, content)

struct ChatCompletionChoice {
    int index = 0;
    ChatCompletionChoiceMessage message;
    std::string finish_reason = "stop";
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionChoice,
    index, message, finish_reason)

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChoice> choices;
    Usage usage;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionResponse,
    id, object, created, model, choices, usage)

// ---------------------------------------------------------------------------
// Chat Completions — Streaming chunks (SSE)
// ---------------------------------------------------------------------------

// Delta in a streaming chunk. Fields are optional — first chunk has role,
// subsequent chunks have content, final chunk is empty with finish_reason.
// We serialize manually to handle optional fields properly.
struct ChatCompletionChunkDelta {
    std::string role;
    std::string content;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionChunkDelta,
    role, content)

struct ChatCompletionChunkChoice {
    int index = 0;
    ChatCompletionChunkDelta delta;
    // Empty string = still generating, "stop" = done, "length" = max_tokens hit
    std::string finish_reason;
};

// Custom serialization: omit finish_reason when empty (OpenAI sends null)
inline void to_json(nlohmann::json& j, const ChatCompletionChunkChoice& c) {
    j["index"] = c.index;
    j["delta"] = c.delta;
    if (c.finish_reason.empty()) {
        j["finish_reason"] = nullptr;
    } else {
        j["finish_reason"] = c.finish_reason;
    }
}
inline void from_json(const nlohmann::json& j, ChatCompletionChunkChoice& c) {
    j.at("index").get_to(c.index);
    j.at("delta").get_to(c.delta);
    if (j.contains("finish_reason") && !j.at("finish_reason").is_null()) {
        j.at("finish_reason").get_to(c.finish_reason);
    }
}

struct ChatCompletionChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChunkChoice> choices;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionChunk,
    id, object, created, model, choices)

// ---------------------------------------------------------------------------
// Text Completions — Request
// ---------------------------------------------------------------------------
struct CompletionRequest {
    std::string model;
    std::string prompt;
    float temperature = 0.6f;
    float top_p = 1.0f;
    int max_tokens = 2048;
    float repetition_penalty = 0.0f;
    bool stream = false;
    bool use_mtp = false;
    std::vector<std::string> stop;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CompletionRequest,
    model, prompt, temperature, top_p, max_tokens,
    repetition_penalty, stream, use_mtp, stop)

// ---------------------------------------------------------------------------
// Text Completions — Response
// ---------------------------------------------------------------------------
struct CompletionChoice {
    int index = 0;
    std::string text;
    std::string finish_reason = "stop";
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CompletionChoice,
    index, text, finish_reason)

struct CompletionResponse {
    std::string id;
    std::string object = "text_completion";
    int64_t created = 0;
    std::string model;
    std::vector<CompletionChoice> choices;
    Usage usage;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CompletionResponse,
    id, object, created, model, choices, usage)

// ---------------------------------------------------------------------------
// Embeddings — Request
// ---------------------------------------------------------------------------
struct EmbeddingRequest {
    std::string model;
    // Input can be a single string or array of strings.
    // For simplicity, always normalize to vector<string> in handler.
    std::vector<std::string> input;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EmbeddingRequest, model, input)

// ---------------------------------------------------------------------------
// Embeddings — Response
// ---------------------------------------------------------------------------
struct EmbeddingObject {
    std::string object = "embedding";
    int index = 0;
    std::vector<float> embedding;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EmbeddingObject,
    object, index, embedding)

struct EmbeddingResponse {
    std::string object = "list";
    std::vector<EmbeddingObject> data;
    std::string model;
    Usage usage;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EmbeddingResponse,
    object, data, model, usage)

// ---------------------------------------------------------------------------
// Utility: generate a unique request ID
// ---------------------------------------------------------------------------
std::string generate_request_id();

// Utility: current unix timestamp
int64_t unix_timestamp();

} // namespace openai
} // namespace mlx_lm
