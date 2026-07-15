// OpenAI-compatible HTTP server implementation.
// Supports single-model and multi-model (auto-load) modes.

#include <mlx-lm/common/server.h>
#include <mlx-lm/common/openai_types.h>
#include <mlx-lm/common/chat.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/kv_cache.h>
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

} // namespace openai

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
            handle_chat_stream(res, model, chat_req, params);
        } else {
            handle_chat_blocking(res, model, chat_req, params);
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
                              const GenerateParameters& params) {
        auto request_id = openai::generate_request_id();
        auto created = openai::unix_timestamp();

        try {
            model->perform([&](ModelContext& ctx) {
                WiredLimitGuard wired_guard;

                auto raw_messages = to_raw_messages(chat_req.messages);

                std::vector<int> tokens;
                if (ctx.apply_chat_template_fn) {
                    tokens = ctx.apply_chat_template_fn(raw_messages);
                } else {
                    tokens = ctx.encode_fn(chat_req.messages.back().content);
                }

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

                auto info = generate_text(
                    ctx, lm_input, params, eos_set,
                    [&](const std::string& text, int /*token*/) {
                        output_text += text;
                        generated_count++;
                        return GenerateDisposition::more;
                    });

                std::cerr << "[TPS] " << info.summary() << std::endl;

                openai::ChatCompletionResponse response;
                response.id = request_id;
                response.created = created;
                response.model = ctx.model_id;

                openai::ChatCompletionChoice choice;
                choice.index = 0;
                choice.message.content = output_text;
                choice.finish_reason = (params.max_tokens.has_value() &&
                    generated_count >= *params.max_tokens) ? "length" : "stop";
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
                            const GenerateParameters& params) {
        auto request_id = openai::generate_request_id();
        auto created = openai::unix_timestamp();

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        res.set_chunked_content_provider(
            "text/event-stream",
            [model, chat_req, params, request_id, created]
            (size_t /*offset*/, httplib::DataSink& sink) -> bool {
                bool client_gone = false;
                try {
                    model->perform([&](ModelContext& ctx) {
                        WiredLimitGuard wired_guard;

                        auto raw_messages = to_raw_messages(chat_req.messages);

                        std::vector<int> tokens;
                        if (ctx.apply_chat_template_fn) {
                            tokens = ctx.apply_chat_template_fn(raw_messages);
                        } else {
                            tokens = ctx.encode_fn(chat_req.messages.back().content);
                        }

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

                        // Generate tokens and stream as SSE. Returning
                        // GenerateDisposition::stop aborts the decode loop
                        // (and frees the model mutex) when the client drops.
                        auto info = generate_text(
                            ctx, lm_input, params, eos_set,
                            [&](const std::string& text, int /*token*/) {
                                if (!client_writable(sink)) {
                                    client_gone = true;
                                    return GenerateDisposition::stop;
                                }
                                openai::ChatCompletionChunk chunk;
                                chunk.id = request_id;
                                chunk.created = created;
                                chunk.model = ctx.model_id;

                                openai::ChatCompletionChunkChoice choice;
                                choice.index = 0;
                                choice.delta.content = text;
                                chunk.choices.push_back(std::move(choice));

                                if (!send_sse(sink, nlohmann::json(chunk).dump())) {
                                    client_gone = true;
                                    return GenerateDisposition::stop;
                                }
                                return GenerateDisposition::more;
                            });

                        if (client_gone) {
                            std::cerr << "[server] client disconnected mid-stream; "
                                      << "cancelled generation for " << request_id
                                      << "\n";
                            return;
                        }

                        std::cerr << "[TPS] " << info.summary() << std::endl;

                        // Send final chunk with finish_reason.
                        {
                            openai::ChatCompletionChunk chunk;
                            chunk.id = request_id;
                            chunk.created = created;
                            chunk.model = ctx.model_id;

                            openai::ChatCompletionChunkChoice choice;
                            choice.index = 0;
                            choice.finish_reason = "stop";
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

                auto info = generate_text(
                    ctx, lm_input, params, eos_set,
                    [&](const std::string& text, int /*token*/) {
                        output_text += text;
                        generated_count++;
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
                choice.finish_reason = (params.max_tokens.has_value() &&
                    generated_count >= *params.max_tokens) ? "length" : "stop";
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
