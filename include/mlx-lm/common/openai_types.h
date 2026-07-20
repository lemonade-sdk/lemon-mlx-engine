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

// tool_choice: "none" | "auto" | "required" | {"type":"function","function":{"name":"..."}}
struct ToolChoice {
    enum class Mode { Auto, None, Required, Function };
    Mode mode = Mode::Auto;
    std::optional<std::string> function_name;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 0.6f;
    float top_p = 1.0f;
    // Default 4096: Qwen thinking-on often needs CoT headroom. Explicit client
    // max_tokens always wins (server floor only applies when the field is omitted
    // from the request body and GenerateParameters still has nullopt).
    int max_tokens = 4096;
    float repetition_penalty = 0.0f;
    bool stream = false;
    bool use_mtp = false;
    // stop sequences (optional, parsed manually)
    std::vector<std::string> stop;
    // OpenAI tools: raw array of tool objects (null/missing = no tools).
    // Kept as JSON for zero-loss schema passthrough into chat templates.
    std::optional<nlohmann::json> tools;
    ToolChoice tool_choice;
    // Optional request-level thinking override (null = use policy default).
    // Precedence: request > tools-inject auto-off > process --no-think / load default.
    std::optional<bool> enable_thinking;
};

inline void to_json(nlohmann::json& j, const ChatCompletionRequest& r) {
    j = nlohmann::json{
        {"model", r.model},
        {"messages", r.messages},
        {"temperature", r.temperature},
        {"top_p", r.top_p},
        {"max_tokens", r.max_tokens},
        {"repetition_penalty", r.repetition_penalty},
        {"stream", r.stream},
        {"use_mtp", r.use_mtp},
        {"stop", r.stop},
    };
    if (r.tools.has_value()) {
        j["tools"] = *r.tools;
    }
    if (r.enable_thinking.has_value()) {
        j["enable_thinking"] = *r.enable_thinking;
    }
    switch (r.tool_choice.mode) {
        case ToolChoice::Mode::None:
            j["tool_choice"] = "none";
            break;
        case ToolChoice::Mode::Required:
            j["tool_choice"] = "required";
            break;
        case ToolChoice::Mode::Function:
            j["tool_choice"] = {
                {"type", "function"},
                {"function", {{"name", r.tool_choice.function_name.value_or("")}}},
            };
            break;
        case ToolChoice::Mode::Auto:
        default:
            // omit or "auto"
            break;
    }
}

inline void from_json(const nlohmann::json& j, ChatCompletionRequest& r) {
    if (j.contains("model") && !j.at("model").is_null()) {
        j.at("model").get_to(r.model);
    }
    if (j.contains("messages") && !j.at("messages").is_null()) {
        j.at("messages").get_to(r.messages);
    }
    if (j.contains("temperature") && !j.at("temperature").is_null()) {
        j.at("temperature").get_to(r.temperature);
    }
    if (j.contains("top_p") && !j.at("top_p").is_null()) {
        j.at("top_p").get_to(r.top_p);
    }
    if (j.contains("max_tokens") && !j.at("max_tokens").is_null()) {
        j.at("max_tokens").get_to(r.max_tokens);
    }
    if (j.contains("repetition_penalty") && !j.at("repetition_penalty").is_null()) {
        j.at("repetition_penalty").get_to(r.repetition_penalty);
    }
    if (j.contains("stream") && !j.at("stream").is_null()) {
        j.at("stream").get_to(r.stream);
    }
    if (j.contains("use_mtp") && !j.at("use_mtp").is_null()) {
        j.at("use_mtp").get_to(r.use_mtp);
    }
    if (j.contains("stop") && !j.at("stop").is_null()) {
        j.at("stop").get_to(r.stop);
    }
    r.tools = std::nullopt;
    if (j.contains("tools") && !j.at("tools").is_null()) {
        r.tools = j.at("tools");
    }
    r.enable_thinking = std::nullopt;
    if (j.contains("enable_thinking") && j.at("enable_thinking").is_boolean()) {
        r.enable_thinking = j.at("enable_thinking").get<bool>();
    } else if (j.contains("chat_template_kwargs") &&
               j.at("chat_template_kwargs").is_object() &&
               j.at("chat_template_kwargs").contains("enable_thinking") &&
               j.at("chat_template_kwargs").at("enable_thinking").is_boolean()) {
        r.enable_thinking =
            j.at("chat_template_kwargs").at("enable_thinking").get<bool>();
    }
    r.tool_choice = ToolChoice{};
    if (j.contains("tool_choice") && !j.at("tool_choice").is_null()) {
        const auto& tc = j.at("tool_choice");
        if (tc.is_string()) {
            const auto s = tc.get<std::string>();
            if (s == "none") {
                r.tool_choice.mode = ToolChoice::Mode::None;
            } else if (s == "required") {
                r.tool_choice.mode = ToolChoice::Mode::Required;
            } else {
                r.tool_choice.mode = ToolChoice::Mode::Auto;
            }
        } else if (tc.is_object()) {
            r.tool_choice.mode = ToolChoice::Mode::Function;
            if (tc.contains("function") && tc["function"].is_object() &&
                tc["function"].contains("name")) {
                r.tool_choice.function_name = tc["function"]["name"].get<std::string>();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Chat Completions — Response (non-streaming)
// ---------------------------------------------------------------------------
struct ChatCompletionToolCallFunction {
    std::string name;
    // OpenAI wire format: arguments is a STRING containing JSON object text.
    std::string arguments;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionToolCallFunction,
    name, arguments)

struct ChatCompletionToolCall {
    std::string id;
    std::string type = "function";
    ChatCompletionToolCallFunction function;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ChatCompletionToolCall,
    id, type, function)

struct ChatCompletionChoiceMessage {
    std::string role = "assistant";
    std::string content;
    std::vector<ChatCompletionToolCall> tool_calls;
};

// Custom: omit empty tool_calls; allow null content when tools present.
inline void to_json(nlohmann::json& j, const ChatCompletionChoiceMessage& m) {
    j = nlohmann::json{{"role", m.role}};
    if (!m.tool_calls.empty()) {
        j["content"] = m.content.empty() ? nlohmann::json(nullptr) : nlohmann::json(m.content);
        j["tool_calls"] = m.tool_calls;
    } else {
        j["content"] = m.content;
    }
}
inline void from_json(const nlohmann::json& j, ChatCompletionChoiceMessage& m) {
    if (j.contains("role") && !j.at("role").is_null()) {
        j.at("role").get_to(m.role);
    }
    if (j.contains("content") && !j.at("content").is_null()) {
        j.at("content").get_to(m.content);
    } else {
        m.content.clear();
    }
    m.tool_calls.clear();
    if (j.contains("tool_calls") && j.at("tool_calls").is_array()) {
        j.at("tool_calls").get_to(m.tool_calls);
    }
}

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
// subsequent chunks have content, tool_calls, final chunk has finish_reason.
struct ChatCompletionChunkDelta {
    std::string role;
    std::string content;
    std::vector<ChatCompletionToolCall> tool_calls;
};

inline void to_json(nlohmann::json& j, const ChatCompletionChunkDelta& d) {
    j = nlohmann::json::object();
    if (!d.role.empty()) {
        j["role"] = d.role;
    }
    if (!d.content.empty()) {
        j["content"] = d.content;
    }
    if (!d.tool_calls.empty()) {
        j["tool_calls"] = d.tool_calls;
    }
}
inline void from_json(const nlohmann::json& j, ChatCompletionChunkDelta& d) {
    d.role.clear();
    d.content.clear();
    d.tool_calls.clear();
    if (j.contains("role") && !j.at("role").is_null()) {
        j.at("role").get_to(d.role);
    }
    if (j.contains("content") && !j.at("content").is_null()) {
        j.at("content").get_to(d.content);
    }
    if (j.contains("tool_calls") && j.at("tool_calls").is_array()) {
        j.at("tool_calls").get_to(d.tool_calls);
    }
}

struct ChatCompletionChunkChoice {
    int index = 0;
    ChatCompletionChunkDelta delta;
    // Empty string = still generating, "stop"/"length"/"tool_calls" = done
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
    int max_tokens = 4096;
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

// Generate a tool call id (e.g. call_<12 hex>).
std::string generate_tool_call_id();

} // namespace openai
} // namespace mlx_lm
