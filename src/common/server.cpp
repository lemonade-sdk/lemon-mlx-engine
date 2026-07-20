// OpenAI-compatible HTTP server implementation.
// Supports single-model and multi-model (auto-load) modes.

#include <mlx-lm/common/server.h>
#include <mlx-lm/common/openai_types.h>
#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/thinking_budget.h>
#include <mlx-lm/common/stop_sequences.h>
#include <mlx-lm/common/tool_calling.h>
#include <mlx-lm/common/types.h>
#include <mlx-lm/common/wired_limit_guard.h>
#include <mlx/mlx.h>

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

namespace mlx_lm {

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

namespace openai {

std::string generate_request_id() {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    std::uniform_int_distribution<uint64_t> dist;
    uint64_t val = dist(rng);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "chatcmpl-%016llx",
                  static_cast<unsigned long long>(val));
    return std::string(buf);
}

int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string generate_tool_call_id() {
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t val = dist(rng);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "call_%012llx",
                  static_cast<unsigned long long>(val & 0xffffffffffffULL));
    return std::string(buf);
}

} // namespace openai

namespace {

// Security caps (tools plan §E5) — parse/emit only, never execute.
constexpr size_t kMaxToolsCount = 64;
constexpr size_t kMaxToolsJsonBytes = 256 * 1024;
constexpr int kMaxToolsNesting = 12;
constexpr size_t kMaxToolCallsEmit = 16;

int json_nesting_depth(const nlohmann::json& j, int depth = 0) {
    if (depth > kMaxToolsNesting) return depth;
    if (j.is_object()) {
        int max_d = depth;
        for (auto it = j.begin(); it != j.end(); ++it) {
            max_d = std::max(max_d, json_nesting_depth(it.value(), depth + 1));
        }
        return max_d;
    }
    if (j.is_array()) {
        int max_d = depth;
        for (const auto& el : j) {
            max_d = std::max(max_d, json_nesting_depth(el, depth + 1));
        }
        return max_d;
    }
    return depth;
}

// Returns error message if invalid; empty string if OK.
// Also normalizes whether tools should be injected into the template.
struct ToolsGate {
    bool inject = false;
    std::string error;
};

ToolsGate validate_and_gate_tools(const openai::ChatCompletionRequest& req) {
    ToolsGate g;
    if (!req.tools.has_value() || req.tools->is_null()) {
        return g;
    }
    if (!req.tools->is_array()) {
        g.error = "tools must be a JSON array";
        return g;
    }
    if (req.tools->size() > kMaxToolsCount) {
        g.error = "tools array exceeds maximum of " + std::to_string(kMaxToolsCount);
        return g;
    }
    const auto dumped = req.tools->dump();
    if (dumped.size() > kMaxToolsJsonBytes) {
        g.error = "tools payload exceeds maximum size";
        return g;
    }
    if (json_nesting_depth(*req.tools) > kMaxToolsNesting) {
        g.error = "tools schema nesting too deep";
        return g;
    }
    for (const auto& t : *req.tools) {
        if (!t.is_object()) {
            g.error = "each tool must be an object";
            return g;
        }
        if (t.value("type", "function") != "function") {
            g.error = "only tools with type \"function\" are supported";
            return g;
        }
        if (!t.contains("function") || !t["function"].is_object()) {
            g.error = "tool.function must be an object";
            return g;
        }
        if (!t["function"].contains("name") ||
            !t["function"]["name"].is_string() ||
            t["function"]["name"].get<std::string>().empty()) {
            g.error = "tool.function.name is required";
            return g;
        }
    }
    if (req.tool_choice.mode == openai::ToolChoice::Mode::None) {
        g.inject = false;
        return g;
    }
    if (req.tools->empty()) {
        if (req.tool_choice.mode != openai::ToolChoice::Mode::Auto) {
            g.error = "tool_choice requires a non-empty tools array";
            return g;
        }
        return g;
    }
    if (req.tool_choice.mode == openai::ToolChoice::Mode::Function) {
        const auto& want = req.tool_choice.function_name.value_or("");
        if (want.empty()) {
            g.error = "tool_choice function name is required";
            return g;
        }
        bool found = false;
        for (const auto& t : *req.tools) {
            if (t["function"]["name"].get<std::string>() == want) {
                found = true;
                break;
            }
        }
        if (!found) {
            g.error = "tool_choice function name not found in tools";
            return g;
        }
    }
    g.inject = true;
    return g;
}

openai::ChatCompletionToolCall to_openai_tool_call(const ToolCall& tc) {
    openai::ChatCompletionToolCall out;
    out.id = openai::generate_tool_call_id();
    out.type = "function";
    out.function.name = tc.function.name;
    // OpenAI wire: arguments must be a stringified JSON object.
    out.function.arguments =
        tc.function.arguments.is_string()
            ? tc.function.arguments.get<std::string>()
            : tc.function.arguments.dump();
    return out;
}

std::vector<openai::ChatCompletionToolCall>
to_openai_tool_calls(const std::vector<ToolCall>& calls) {
    std::vector<openai::ChatCompletionToolCall> out;
    out.reserve(std::min(calls.size(), kMaxToolCallsEmit));
    for (size_t i = 0; i < calls.size() && i < kMaxToolCallsEmit; ++i) {
        out.push_back(to_openai_tool_call(calls[i]));
    }
    return out;
}

ToolCallFormat resolve_tool_format(const ModelContext& ctx) {
    if (!ctx.model_type.empty()) {
        if (auto fmt = infer_tool_call_format(ctx.model_type)) {
            return *fmt;
        }
    }
    return ToolCallFormat::json;
}

// Per-request thinking policy (tools × enable_thinking).
// Precedence: request explicit enable_thinking >
//             tools inject (non-empty tools && tool_choice != none) ⇒ OFF >
//             process/load default already in template_extra_context.
// Mutates shared template_extra_context under ModelContainer::perform mutex.
// Caller MUST restore previous value after apply_chat_template_fn (RAII below)
// so tools_auto does not sticky-disable thinking for later non-tools requests.
//
struct ThinkingContextGuard {
    nlohmann::json* ctx = nullptr;
    bool had_prev = false;
    bool prev_value = true;
    bool thinking_on = true;  // resolved polarity for this request

    ThinkingContextGuard(ModelContext& model_ctx,
                         bool inject_tools,
                         const std::optional<bool>& request_enable_thinking) {
        if (!model_ctx.template_extra_context) {
            // No template context — still resolve polarity for budget warnings.
            if (request_enable_thinking.has_value()) {
                thinking_on = *request_enable_thinking;
            } else if (inject_tools) {
                thinking_on = false;
            } else {
                thinking_on = true;
            }
            return;
        }
        ctx = model_ctx.template_extra_context.get();
        if (ctx->contains("enable_thinking") &&
            (*ctx)["enable_thinking"].is_boolean()) {
            had_prev = true;
            prev_value = (*ctx)["enable_thinking"].get<bool>();
        }
        bool thinking;
        const char* reason = "process_default";
        if (request_enable_thinking.has_value()) {
            thinking = *request_enable_thinking;
            reason = "client";
        } else if (inject_tools) {
            thinking = false;
            reason = "tools_auto";
        } else if (had_prev) {
            thinking = prev_value;
            reason = "process_default";
        } else {
            thinking = true;
            reason = "process_default";
        }
        thinking_on = thinking;
        (*ctx)["enable_thinking"] = thinking;
        std::cerr << "[server] effective_thinking=" << (thinking ? "on" : "off")
                  << " reason=" << reason << std::endl;
    }

    ~ThinkingContextGuard() {
        if (!ctx) {
            return;
        }
        if (had_prev) {
            (*ctx)["enable_thinking"] = prev_value;
        } else {
            ctx->erase("enable_thinking");
        }
    }

    ThinkingContextGuard(const ThinkingContextGuard&) = delete;
    ThinkingContextGuard& operator=(const ThinkingContextGuard&) = delete;
};

// Soft floor wrapper with logging (policy in thinking_budget.h).
static void apply_thinking_budget_floor_logged(
    GenerateParameters& params,
    bool thinking_on)
{
    const int before = params.max_tokens.value_or(-1);
    if (apply_thinking_budget_floor(params.max_tokens, thinking_on)) {
        std::cerr << "[server] thinking_budget_floor: max_tokens "
                  << before << " → " << kThinkingBudgetRecommend
                  << " (thinking=on; CoT often exhausts lower budgets)\n";
    }
}

} // namespace


// ---------------------------------------------------------------------------
// Server::Impl
// ---------------------------------------------------------------------------

struct Server::Impl {
    std::shared_ptr<ModelManager> manager;
    httplib::Server svr;

    explicit Impl(std::shared_ptr<ModelManager> mgr)
        : manager(std::move(mgr)) {}

    // --- Resolve model from request ---

    std::shared_ptr<ModelContainer> resolve_model(const std::string& requested_model) {
        // If a specific model is requested, load it.
        if (!requested_model.empty()) {
            return manager->get_or_load(requested_model);
        }

        // Fall back to the default (first loaded) model.
        auto default_id = manager->default_model_id();
        if (default_id.empty()) {
            throw std::runtime_error(
                "No model specified in request and no default model loaded. "
                "Send a request with a \"model\" field, or pre-load a model.");
        }
        return manager->get_or_load(default_id);
    }

    // --- Route handlers ---

    void handle_health(const httplib::Request& /*req*/, httplib::Response& res) {
        nlohmann::json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    }

    void handle_models(const httplib::Request& /*req*/, httplib::Response& res) {
        auto available = manager->list_available();
        auto ts = openai::unix_timestamp();

        openai::ModelList list;
        for (const auto& m : available) {
            openai::ModelObject obj;
            obj.id = m.model_id;
            obj.created = ts;
            obj.owned_by = m.loaded ? "local (loaded)" : "local";
            list.data.push_back(std::move(obj));
        }

        // If no models discovered but we have loaded models, list those.
        if (list.data.empty()) {
            for (const auto& id : manager->list_loaded()) {
                openai::ModelObject obj;
                obj.id = id;
                obj.created = ts;
                obj.owned_by = "local (loaded)";
                list.data.push_back(std::move(obj));
            }
        }

        nlohmann::json j = list;
        res.set_content(j.dump(), "application/json");
    }

    void handle_load(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        auto model_id = body.value("model", std::string{});
        if (model_id.empty()) {
            send_error(res, 400, "\"model\" field is required");
            return;
        }

        try {
            manager->get_or_load(model_id);
            nlohmann::json j = {
                {"status", "ok"},
                {"model", model_id},
                {"message", "Model loaded successfully"}
            };
            res.set_content(j.dump(), "application/json");
        } catch (const std::exception& e) {
            send_error(res, 500, std::string("Failed to load model: ") + e.what());
        }
    }

    void handle_unload(const httplib::Request& req, httplib::Response& res) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        auto model_id = body.value("model", std::string{});
        if (model_id.empty()) {
            send_error(res, 400, "\"model\" field is required");
            return;
        }

        manager->unload(model_id);
        nlohmann::json j = {
            {"status", "ok"},
            {"model", model_id},
            {"message", "Model unloaded"}
        };
        res.set_content(j.dump(), "application/json");
    }

    void handle_chat_completions(const httplib::Request& req,
                                 httplib::Response& res,
                                 const GenerateParameters& defaults) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        openai::ChatCompletionRequest chat_req;
        try {
            chat_req = body.get<openai::ChatCompletionRequest>();
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid request: ") + e.what());
            return;
        }

        if (chat_req.messages.empty()) {
            send_error(res, 400, "messages array must not be empty");
            return;
        }

        // v1: multi-turn tool history not supported (MASTER freeze).
        // OWUI Memory / native tools often inject role=tool turns; return a clear
        // 400 (not 200 soup) so operators can disable Memory/tools rather than
        // blame decode. Full multi-turn tools product is deferred.
        for (const auto& m : chat_req.messages) {
            if (m.role == "tool") {
                send_error(res, 400,
                    "role \"tool\" messages are not supported in this version. "
                    "Disable OpenWebUI Memory/RAG and native function-tools "
                    "(tool follow-up turns require multi-turn tools support). "
                    "Plain chat: send only user/assistant messages.");
                return;
            }
        }

        auto tools_gate = validate_and_gate_tools(chat_req);
        if (!tools_gate.error.empty()) {
            send_error(res, 400, tools_gate.error);
            return;
        }

        // Resolve the model from the request.
        std::shared_ptr<ModelContainer> model;
        try {
            model = resolve_model(chat_req.model);
        } catch (const std::exception& e) {
            send_error(res, 404, std::string("Model error: ") + e.what());
            return;
        }

        // Build generation parameters from request + defaults.
        GenerateParameters params = defaults;
        params.temperature = chat_req.temperature;
        params.top_p = chat_req.top_p;
        params.max_tokens = chat_req.max_tokens;
        if (chat_req.repetition_penalty > 0.0f) {
            params.repetition_penalty = chat_req.repetition_penalty;
        }
        if (chat_req.use_mtp) {
            params.use_mtp = chat_req.use_mtp;
        }

        if (chat_req.stream) {
            handle_chat_stream(res, model, chat_req, params, tools_gate.inject);
        } else {
            handle_chat_blocking(res, model, chat_req, params, tools_gate.inject);
        }
    }

    void handle_completions(const httplib::Request& req,
                            httplib::Response& res,
                            const GenerateParameters& defaults) {
        nlohmann::json body;
        try {
            body = nlohmann::json::parse(req.body);
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid JSON: ") + e.what());
            return;
        }

        openai::CompletionRequest comp_req;
        try {
            comp_req = body.get<openai::CompletionRequest>();
        } catch (const std::exception& e) {
            send_error(res, 400, std::string("Invalid request: ") + e.what());
            return;
        }

        if (comp_req.prompt.empty()) {
            send_error(res, 400, "prompt must not be empty");
            return;
        }

        // Resolve the model from the request.
        std::shared_ptr<ModelContainer> model;
        try {
            model = resolve_model(comp_req.model);
        } catch (const std::exception& e) {
            send_error(res, 404, std::string("Model error: ") + e.what());
            return;
        }

        GenerateParameters params = defaults;
        params.temperature = comp_req.temperature;
        params.top_p = comp_req.top_p;
        params.max_tokens = comp_req.max_tokens;
        if (comp_req.repetition_penalty > 0.0f) {
            params.repetition_penalty = comp_req.repetition_penalty;
        }
        if (comp_req.use_mtp) {
            params.use_mtp = comp_req.use_mtp;
        }

        handle_completion_blocking(res, model, comp_req, params);
    }

    // --- Blocking chat completion ---

    void handle_chat_blocking(httplib::Response& res,
                              std::shared_ptr<ModelContainer> model,
                              const openai::ChatCompletionRequest& chat_req,
                              const GenerateParameters& params,
                              bool inject_tools) {
        auto request_id = openai::generate_request_id();
        auto created = openai::unix_timestamp();

        try {
            model->perform([&](ModelContext& ctx) {
                WiredLimitGuard wired_guard;

                std::vector<int> tokens;
                bool thinking_on = true;
                {
                    ThinkingContextGuard thinking_guard(
                        ctx, inject_tools, chat_req.enable_thinking);
                    thinking_on = thinking_guard.thinking_on;
                    auto raw_messages = to_raw_messages(chat_req.messages);
                    const nlohmann::json* tools_ptr =
                        (inject_tools && chat_req.tools.has_value())
                            ? &*chat_req.tools
                            : nullptr;

                    if (ctx.apply_chat_template_fn) {
                        tokens = ctx.apply_chat_template_fn(raw_messages, tools_ptr);
                    } else {
                        tokens = ctx.encode_fn(chat_req.messages.back().content);
                    }
                } // restore process thinking default after template apply
                // params is local to this request; floor after template so
                // tools_auto thinking-off is not over-budgeted.
                GenerateParameters gen_params = params;
                apply_thinking_budget_floor_logged(gen_params, thinking_on);

                if (tokens.empty()) {
                    throw std::runtime_error("tokenization produced no tokens");
                }

                auto token_array = mx::array(
                    tokens.data(),
                    {static_cast<int>(tokens.size())},
                    mx::int32);
                LMInput lm_input(token_array);

                auto eos_set = build_eos_set(ctx);

                std::string output_text;
                int generated_count = 0;
                bool stopped_on_string = false;

                auto info = generate_text(
                    ctx, lm_input, gen_params, eos_set,
                    [&](const std::string& text, int /*token*/) {
                        output_text += text;
                        generated_count++;
                        if (apply_stop_sequences(output_text, chat_req.stop)) {
                            stopped_on_string = true;
                            return GenerateDisposition::stop;
                        }
                        return GenerateDisposition::more;
                    });

                std::cerr << "[TPS] " << info.summary() << std::endl;

                openai::ChatCompletionResponse response;
                response.id = request_id;
                response.created = created;
                response.model = ctx.model_id;

                openai::ChatCompletionChoice choice;
                choice.index = 0;

                // Parse/emit tool_calls only — never Tool::execute (security).
                if (inject_tools) {
                    std::optional<nlohmann::json> tools_opt =
                        chat_req.tools.has_value() ? chat_req.tools : std::nullopt;
                    auto try_parse = [&](ToolCallFormat fmt)
                        -> std::pair<std::string, std::vector<openai::ChatCompletionToolCall>> {
                        ToolCallProcessor processor(fmt, tools_opt);
                        auto display = processor.process_chunk(output_text);
                        return {display.value_or(""),
                                to_openai_tool_calls(processor.tool_calls())};
                    };

                    auto fmt = resolve_tool_format(ctx);
                    auto [display_text, openai_calls] = try_parse(fmt);
                    if (openai_calls.empty()) {
                        // Fallback between JSON-tag and XML-function styles.
                        const ToolCallFormat alt =
                            (fmt == ToolCallFormat::xml_function)
                                ? ToolCallFormat::json
                                : ToolCallFormat::xml_function;
                        auto alt_result = try_parse(alt);
                        display_text = std::move(alt_result.first);
                        openai_calls = std::move(alt_result.second);
                    }

                    if (!openai_calls.empty()) {
                        choice.message.content = display_text;
                        choice.message.tool_calls = std::move(openai_calls);
                        choice.finish_reason = "tool_calls";
                    } else {
                        choice.message.content = output_text;
                        choice.finish_reason =
                            (!stopped_on_string && gen_params.max_tokens.has_value() &&
                             generated_count >= *gen_params.max_tokens)
                                ? "length"
                                : "stop";
                    }
                } else {
                    choice.message.content = output_text;
                    choice.finish_reason =
                        (!stopped_on_string && gen_params.max_tokens.has_value() &&
                         generated_count >= *gen_params.max_tokens)
                            ? "length"
                            : "stop";
                }
                response.choices.push_back(std::move(choice));

                response.usage.prompt_tokens = info.prompt_token_count;
                response.usage.completion_tokens = info.generation_token_count;
                response.usage.total_tokens =
                    info.prompt_token_count + info.generation_token_count;

                nlohmann::json j = response;
                res.set_content(j.dump(), "application/json");
            });
        } catch (const std::exception& e) {
            send_error(res, 500, std::string("Generation error: ") + e.what());
        }
    }

    // --- Streaming chat completion (SSE) ---

    void handle_chat_stream(httplib::Response& res,
                            std::shared_ptr<ModelContainer> model,
                            const openai::ChatCompletionRequest& chat_req,
                            const GenerateParameters& params,
                            bool inject_tools) {
        auto request_id = openai::generate_request_id();
        auto created = openai::unix_timestamp();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [model, chat_req, params, request_id, created, inject_tools]
            (size_t /*offset*/, httplib::DataSink& sink) -> bool {
                bool client_gone = false;
                try {
                    model->perform([&](ModelContext& ctx) {
                        WiredLimitGuard wired_guard;

                        std::vector<int> tokens;
                        bool thinking_on = true;
                        {
                            ThinkingContextGuard thinking_guard(
                                ctx, inject_tools, chat_req.enable_thinking);
                            thinking_on = thinking_guard.thinking_on;
                            auto raw_messages = to_raw_messages(chat_req.messages);
                            const nlohmann::json* tools_ptr =
                                (inject_tools && chat_req.tools.has_value())
                                    ? &*chat_req.tools
                                    : nullptr;

                            if (ctx.apply_chat_template_fn) {
                                tokens =
                                    ctx.apply_chat_template_fn(raw_messages, tools_ptr);
                            } else {
                                tokens =
                                    ctx.encode_fn(chat_req.messages.back().content);
                            }
                        } // restore process thinking default after template apply
                        GenerateParameters gen_params = params;
                        apply_thinking_budget_floor_logged(gen_params, thinking_on);

                        if (tokens.empty()) {
                            throw std::runtime_error("tokenization produced no tokens");
                        }

                        auto token_array = mx::array(
                            tokens.data(),
                            {static_cast<int>(tokens.size())},
                            mx::int32);
                        LMInput lm_input(token_array);

                        auto eos_set = build_eos_set(ctx);

                        // Send initial chunk with role.
                        {
                            openai::ChatCompletionChunk chunk;
                            chunk.id = request_id;
                            chunk.created = created;
                            chunk.model = ctx.model_id;

                            openai::ChatCompletionChunkChoice choice;
                            choice.index = 0;
                            choice.delta.role = "assistant";
                            chunk.choices.push_back(std::move(choice));

                            if (!send_sse(sink, nlohmann::json(chunk).dump())) {
                                client_gone = true;
                                return;
                            }
                        }

                        std::optional<nlohmann::json> tools_opt =
                            inject_tools && chat_req.tools.has_value()
                                ? chat_req.tools
                                : std::nullopt;
                        std::unique_ptr<ToolCallProcessor> processor;
                        if (inject_tools) {
                            processor = std::make_unique<ToolCallProcessor>(
                                resolve_tool_format(ctx), tools_opt);
                        }
                        int generated_count = 0;
                        std::string stream_accum;
                        bool stopped_on_string = false;

                        // Generate tokens and stream as SSE. Tier-1 tools:
                        // suppress tool markup in content; emit complete
                        // tool_calls after generation (not true arg streaming).
                        auto info = generate_text(
                            ctx, lm_input, gen_params, eos_set,
                            [&](const std::string& text, int /*token*/) {
                                generated_count++;
                                std::string out_text = text;
                                if (processor) {
                                    auto display = processor->process_chunk(text);
                                    if (!display.has_value() || display->empty()) {
                                        return GenerateDisposition::more;
                                    }
                                    out_text = *display;
                                }

                                // Honor request stop strings. Strip the match from
                                // the tail of this chunk when possible; stop gen.
                                const size_t before = stream_accum.size();
                                stream_accum += out_text;
                                if (apply_stop_sequences(stream_accum, chat_req.stop)) {
                                    stopped_on_string = true;
                                    if (stream_accum.size() > before) {
                                        out_text = stream_accum.substr(before);
                                    } else {
                                        // Entire chunk was part of the stop tail.
                                        return GenerateDisposition::stop;
                                    }
                                }

                                openai::ChatCompletionChunk chunk;
                                chunk.id = request_id;
                                chunk.created = created;
                                chunk.model = ctx.model_id;

                                openai::ChatCompletionChunkChoice choice;
                                choice.index = 0;
                                choice.delta.content = out_text;
                                chunk.choices.push_back(std::move(choice));

                                if (!send_sse(sink, nlohmann::json(chunk).dump())) {
                                    client_gone = true;
                                    return GenerateDisposition::stop;
                                }
                                if (stopped_on_string) {
                                    return GenerateDisposition::stop;
                                }
                                return GenerateDisposition::more;
                            },
                            [&]() {
                                if (client_writable(sink)) {
                                    return false;
                                }
                                client_gone = true;
                                return true;
                            });

                        if (client_gone) {
                            std::cerr << "[server] client disconnected mid-stream; "
                                      << "cancelled generation for " << request_id
                                      << "\n";
                            return;
                        }

                        std::cerr << "[TPS] " << info.summary() << std::endl;

                        std::string finish = "stop";
                        if (!stopped_on_string && gen_params.max_tokens.has_value() &&
                            generated_count >= *gen_params.max_tokens) {
                            finish = "length";
                        }

                        // Emit complete tool_calls (Tier-1) then finish_reason.
                        if (processor && !processor->tool_calls().empty()) {
                            auto openai_calls =
                                to_openai_tool_calls(processor->tool_calls());
                            // OpenAI stream: one delta per tool call index.
                            for (size_t i = 0; i < openai_calls.size(); ++i) {
                                openai::ChatCompletionChunk chunk;
                                chunk.id = request_id;
                                chunk.created = created;
                                chunk.model = ctx.model_id;
                                openai::ChatCompletionChunkChoice choice;
                                choice.index = 0;
                                // Use index field inside tool_calls array for multi.
                                auto tc = openai_calls[i];
                                // Encode index as first tool_calls entry (OpenAI uses
                                // delta.tool_calls[].index — add if needed).
                                nlohmann::json delta = {
                                    {"tool_calls", nlohmann::json::array({
                                        {
                                            {"index", static_cast<int>(i)},
                                            {"id", tc.id},
                                            {"type", tc.type},
                                            {"function",
                                             {{"name", tc.function.name},
                                              {"arguments", tc.function.arguments}}},
                                        },
                                    })},
                                };
                                nlohmann::json choice_j = {
                                    {"index", 0},
                                    {"delta", delta},
                                    {"finish_reason", nullptr},
                                };
                                nlohmann::json chunk_j = {
                                    {"id", request_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", created},
                                    {"model", ctx.model_id},
                                    {"choices", nlohmann::json::array({choice_j})},
                                };
                                if (!send_sse(sink, chunk_j.dump())) {
                                    client_gone = true;
                                    return;
                                }
                            }
                            finish = "tool_calls";
                        }

                        {
                            openai::ChatCompletionChunk chunk;
                            chunk.id = request_id;
                            chunk.created = created;
                            chunk.model = ctx.model_id;

                            openai::ChatCompletionChunkChoice choice;
                            choice.index = 0;
                            choice.finish_reason = finish;
                            chunk.choices.push_back(std::move(choice));

                            if (!send_sse(sink, nlohmann::json(chunk).dump())) {
                                client_gone = true;
                                return;
                            }
                        }

                        send_sse(sink, "[DONE]");
                    });
                } catch (const std::exception& e) {
                    if (client_writable(sink)) {
                        nlohmann::json err = {{"error", e.what()}};
                        send_sse(sink, err.dump());
                    }
                }

                if (!client_gone) {
                    sink.done();
                }
                // false tells httplib the connection is finished/aborted.
                return !client_gone;
            });
    }

    // --- Blocking text completion ---

    void handle_completion_blocking(httplib::Response& res,
                                    std::shared_ptr<ModelContainer> model,
                                    const openai::CompletionRequest& comp_req,
                                    const GenerateParameters& params) {
        auto request_id = openai::generate_request_id();
        auto created = openai::unix_timestamp();

        try {
            model->perform([&](ModelContext& ctx) {
                WiredLimitGuard wired_guard;

                auto tokens = ctx.encode_fn(comp_req.prompt);
                if (tokens.empty()) {
                    throw std::runtime_error("tokenization produced no tokens");
                }

                auto token_array = mx::array(
                    tokens.data(),
                    {static_cast<int>(tokens.size())},
                    mx::int32);
                LMInput lm_input(token_array);

                auto eos_set = build_eos_set(ctx);

                std::string output_text;
                int generated_count = 0;
                bool stopped_on_string = false;

                auto info = generate_text(
                    ctx, lm_input, params, eos_set,
                    [&](const std::string& text, int /*token*/) {
                        output_text += text;
                        generated_count++;
                        if (apply_stop_sequences(output_text, comp_req.stop)) {
                            stopped_on_string = true;
                            return GenerateDisposition::stop;
                        }
                        return GenerateDisposition::more;
                    });

                std::cerr << "[TPS] " << info.summary() << std::endl;

                openai::CompletionResponse response;
                response.id = request_id;
                response.created = created;
                response.model = ctx.model_id;

                openai::CompletionChoice choice;
                choice.index = 0;
                choice.text = output_text;
                choice.finish_reason =
                    (!stopped_on_string && params.max_tokens.has_value() &&
                     generated_count >= *params.max_tokens)
                        ? "length"
                        : "stop";
                response.choices.push_back(std::move(choice));

                response.usage.prompt_tokens = info.prompt_token_count;
                response.usage.completion_tokens = info.generation_token_count;
                response.usage.total_tokens =
                    info.prompt_token_count + info.generation_token_count;

                nlohmann::json j = response;
                res.set_content(j.dump(), "application/json");
            });
        } catch (const std::exception& e) {
            send_error(res, 500, std::string("Generation error: ") + e.what());
        }
    }

    // --- Helpers ---

    static std::vector<std::unordered_map<std::string, std::string>>
    to_raw_messages(const std::vector<openai::ChatMessage>& messages) {
        std::vector<std::unordered_map<std::string, std::string>> raw;
        raw.reserve(messages.size());
        for (const auto& msg : messages) {
            raw.push_back({{"role", msg.role}, {"content", msg.content}});
        }
        return raw;
    }

    static std::set<int> build_eos_set(const ModelContext& ctx) {
        std::set<int> eos_set;
        if (ctx.eos_token_ids.has_value()) {
            for (int id : *ctx.eos_token_ids) {
                eos_set.insert(id);
            }
        }
        return eos_set;
    }

    // apply_stop_sequences — see mlx-lm/common/stop_sequences.h

    // Returns false if the client disconnected (or write failed) so callers
    // can stop generation and release the model lock promptly.
    static bool client_writable(httplib::DataSink& sink) {
        if (sink.is_writable && !sink.is_writable()) {
            return false;
        }
        return true;
    }

    static bool send_sse(httplib::DataSink& sink, const std::string& data) {
        if (!client_writable(sink)) {
            return false;
        }
        std::string event = "data: " + data + "\n\n";
        if (!sink.write(event.data(), event.size())) {
            return false;
        }
        return client_writable(sink);
    }

    static void send_error(httplib::Response& res, int status,
                           const std::string& message) {
        openai::ErrorResponse err;
        err.error.message = message;
        if (status == 400) {
            err.error.type = "invalid_request_error";
            err.error.code = "invalid_request";
        } else if (status == 404) {
            err.error.type = "not_found_error";
            err.error.code = "model_not_found";
        } else {
            err.error.type = "server_error";
            err.error.code = "internal_error";
        }
        nlohmann::json j = err;
        res.status = status;
        res.set_content(j.dump(), "application/json");
    }
};

// ---------------------------------------------------------------------------
// Server public API
// ---------------------------------------------------------------------------

Server::Server(std::shared_ptr<ModelContainer> model, ServerConfig config)
    : config_(std::move(config))
{
    // Wrap single model in a ModelManager for unified handling.
    auto manager = std::make_shared<ModelManager>();
    manager->add_loaded(model->model_id(), std::move(model));
    impl_ = std::make_unique<Impl>(std::move(manager));
    setup_routes();
}

Server::Server(std::shared_ptr<ModelManager> manager, ServerConfig config)
    : impl_(std::make_unique<Impl>(std::move(manager))),
      config_(std::move(config))
{
    setup_routes();
}

Server::~Server() = default;

void Server::setup_routes() {
    auto& svr = impl_->svr;
    auto* impl = impl_.get();
    auto defaults = config_.default_params;

    // CORS headers for browser clients.
    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
        {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
    });

    // Health check.
    svr.Get("/health", [impl](const httplib::Request& req, httplib::Response& res) {
        impl->handle_health(req, res);
    });

    // List models (returns all available MLX models from HF cache).
    svr.Get("/v1/models", [impl](const httplib::Request& req, httplib::Response& res) {
        impl->handle_models(req, res);
    });

    // Load a model explicitly.
    svr.Post("/load", [impl](const httplib::Request& req, httplib::Response& res) {
        impl->handle_load(req, res);
    });

    // Unload a model.
    svr.Post("/unload", [impl](const httplib::Request& req, httplib::Response& res) {
        impl->handle_unload(req, res);
    });

    // Chat completions (auto-loads model from request "model" field).
    svr.Post("/v1/chat/completions",
        [impl, defaults](const httplib::Request& req, httplib::Response& res) {
            impl->handle_chat_completions(req, res, defaults);
        });

    // Text completions.
    svr.Post("/v1/completions",
        [impl, defaults](const httplib::Request& req, httplib::Response& res) {
            impl->handle_completions(req, res, defaults);
        });

    // OPTIONS preflight for CORS.
    svr.Options(".*", [](const httplib::Request& /*req*/, httplib::Response& res) {
        res.status = 204;
    });
}

void Server::start() {
    std::cerr << "Server listening on http://" << config_.host
              << ":" << config_.port << std::endl;
    impl_->svr.listen(config_.host, config_.port);
}

void Server::stop() {
    impl_->svr.stop();
}

} // namespace mlx_lm
