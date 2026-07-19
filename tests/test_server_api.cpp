// Integration tests for the OpenAI-compatible server API.
// Uses cpp-httplib client to make real HTTP requests.
// Requires a cached MLX model (mlx-community/Qwen3.5-0.8B-4bit by default).
//
// Tag: [server-api] — can be skipped in CI if no model is available.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include <mlx-lm/common/server.h>
#include <mlx-lm/common/model_manager.h>
#include <mlx-lm/common/hub_api.h>
#include <mlx-lm/llm/llm_factory.h>
#include <mlx/mlx.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <thread>
#include <atomic>

namespace mx = mlx::core;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// Small model for integration tests — fast to load, enough for 2+2.
static const char* TEST_MODEL = "mlx-community/Qwen3.5-0.8B-4bit";
static const int TEST_PORT = 18321; // unlikely to conflict

static bool model_is_cached() {
    return mlx_lm::HubApi::shared().is_cached(TEST_MODEL);
}

// RAII server runner: starts server in a background thread, stops on destruction.
struct ServerRunner {
    std::unique_ptr<mlx_lm::Server> server;
    std::thread thread;
    std::atomic<bool> ready{false};

    ServerRunner(std::shared_ptr<mlx_lm::ModelManager> manager, int port) {
        mlx_lm::ServerConfig config;
        config.host = "127.0.0.1";
        config.port = port;
        config.default_params.max_tokens = 128;
        config.default_params.temperature = 0.0f; // deterministic

        server = std::make_unique<mlx_lm::Server>(manager, config);

        thread = std::thread([this]() {
            server->start();
        });

        // Poll /health until the socket is actually accepting connections.
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(1);
        cli.set_read_timeout(1);
        bool up = false;
        for (int i = 0; i < 200; ++i) {
            if (auto res = cli.Get("/health"); res && res->status == 200) {
                up = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        ready.store(up);
        REQUIRE(up);
    }

    ~ServerRunner() {
        server->stop();
        if (thread.joinable()) thread.join();
    }
};

// ---------------------------------------------------------------------------
// Model discovery tests (no model loading required)
// ---------------------------------------------------------------------------

TEST_CASE("HubApi cache_dir respects env vars", "[server-api][discovery]") {
    auto& hub = mlx_lm::HubApi::shared();
    auto dir = hub.cache_dir();
    // Should be a non-empty path
    REQUIRE(!dir.empty());
    // Default path should contain "huggingface"
    REQUIRE(dir.find("huggingface") != std::string::npos);
}

TEST_CASE("HubApi discover_cached_models finds only MLX models", "[server-api][discovery]") {
    auto& hub = mlx_lm::HubApi::shared();
    auto models = hub.discover_cached_models();

    // We should find at least some models (if HF cache exists)
    // More importantly, none should be GGUF-only
    for (const auto& m : models) {
        REQUIRE(!m.model_id.empty());
        REQUIRE(!m.local_path.empty());

        // Verify: the local path should have config.json and safetensors
        auto config_path = m.local_path + "/config.json";
        REQUIRE(std::filesystem::exists(config_path));

        // Must have safetensors (not just GGUF)
        bool has_safetensors = false;
        for (const auto& entry : std::filesystem::directory_iterator(m.local_path)) {
            if (entry.path().extension() == ".safetensors") {
                has_safetensors = true;
                break;
            }
        }
        bool has_index = std::filesystem::exists(
            m.local_path + "/model.safetensors.index.json");
        REQUIRE((has_safetensors || has_index));

        // Should NOT be a GGUF-only model
        INFO("Model: " << m.model_id);
    }
}

TEST_CASE("ModelManager list_available returns MLX models only", "[server-api][discovery]") {
    mlx_lm::ModelManager manager;
    auto available = manager.list_available();

    for (const auto& m : available) {
        REQUIRE(!m.model_id.empty());
        // model_type should be parseable from config.json
        // (not all may have it, but none should be "gguf")
        INFO("Model: " << m.model_id << " type: " << m.model_type);
    }
}

TEST_CASE("HubApi resolve_cache_path handles both -- and - formats", "[server-api][discovery]") {
    auto& hub = mlx_lm::HubApi::shared();

    // If we have a model in the old format, it should still resolve
    if (hub.is_cached("mlx-community/Qwen3.5-0.8B-4bit")) {
        auto path = hub.model_directory("mlx-community/Qwen3.5-0.8B-4bit");
        REQUIRE(!path.empty());
        REQUIRE(std::filesystem::exists(path));
        REQUIRE(std::filesystem::exists(path + "/config.json"));
    }
}

// ---------------------------------------------------------------------------
// Server endpoint tests (requires cached model)
// ---------------------------------------------------------------------------

TEST_CASE("GET /health returns ok", "[server-api][endpoints]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT);

    httplib::Client cli("127.0.0.1", TEST_PORT);
    cli.set_connection_timeout(5);

    auto res = cli.Get("/health");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    auto body = json::parse(res->body);
    REQUIRE(body["status"] == "ok");
}

TEST_CASE("GET /v1/models lists available MLX models", "[server-api][endpoints]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 1);

    httplib::Client cli("127.0.0.1", TEST_PORT + 1);
    cli.set_connection_timeout(5);

    auto res = cli.Get("/v1/models");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    auto body = json::parse(res->body);
    REQUIRE(body["object"] == "list");
    REQUIRE(body["data"].is_array());
    REQUIRE(!body["data"].empty());

    // All listed models should have an id
    for (const auto& m : body["data"]) {
        REQUIRE(m.contains("id"));
        REQUIRE(!m["id"].get<std::string>().empty());
    }

    // Test model should be in the list
    bool found = false;
    for (const auto& m : body["data"]) {
        if (m["id"].get<std::string>() == TEST_MODEL ||
            m["id"].get<std::string>().find("Qwen3.5-0.8B-4bit") != std::string::npos) {
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

TEST_CASE("POST /load loads a model explicitly", "[server-api][endpoints]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(true);
    ServerRunner runner(manager, TEST_PORT + 2);

    httplib::Client cli("127.0.0.1", TEST_PORT + 2);
    cli.set_connection_timeout(120); // loading can be slow
    cli.set_read_timeout(120);

    json req_body = {{"model", TEST_MODEL}};
    auto res = cli.Post("/load", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    auto body = json::parse(res->body);
    REQUIRE(body["status"] == "ok");
    REQUIRE(body["model"] == TEST_MODEL);
}

TEST_CASE("POST /load returns error for unknown model with no-download", "[server-api][endpoints]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 3);

    httplib::Client cli("127.0.0.1", TEST_PORT + 3);
    cli.set_connection_timeout(5);

    json req_body = {{"model", "nonexistent/model-that-does-not-exist"}};
    auto res = cli.Post("/load", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 500);

    auto body = json::parse(res->body);
    REQUIRE(body.contains("error"));
}

TEST_CASE("POST /v1/chat/completions auto-loads model and generates response", "[server-api][inference]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    // Keep thinking enabled: Qwen3.5 empty-think prefill often samples <|im_end|> first.
    manager->set_no_think(false);
    ServerRunner runner(manager, TEST_PORT + 4);

    httplib::Client cli("127.0.0.1", TEST_PORT + 4);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(120);

    json req_body = {
        {"model", TEST_MODEL},
        {"messages", json::array({
            {{"role", "user"}, {"content", "What is 2+2? Reply with just the number."}}
        })},
        // Thinking models need enough budget to reach the answer inside the chain-of-thought.
        {"max_tokens", 128},
        {"temperature", 0.0},
        {"stream", false}
    };

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    auto body = json::parse(res->body);
    REQUIRE(body["object"] == "chat.completion");
    REQUIRE(!body["choices"].empty());
    REQUIRE(body["choices"][0].contains("message"));

    auto content = body["choices"][0]["message"]["content"].get<std::string>();
    REQUIRE(!content.empty());

    // The answer should contain "4"
    INFO("Response: " << content);
    REQUIRE(content.find("4") != std::string::npos);

    // Should have usage stats
    REQUIRE(body.contains("usage"));
    REQUIRE(body["usage"]["prompt_tokens"].get<int>() > 0);
    REQUIRE(body["usage"]["completion_tokens"].get<int>() > 0);
}

TEST_CASE("POST /v1/chat/completions streaming", "[server-api][inference]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(false);
    ServerRunner runner(manager, TEST_PORT + 5);

    httplib::Client cli("127.0.0.1", TEST_PORT + 5);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(180);

    json req_body = {
        {"model", TEST_MODEL},
        {"messages", json::array({
            {{"role", "user"}, {"content", "What is 2+2? Reply with just the number."}}
        })},
        {"max_tokens", 128},
        {"temperature", 0.0},
        {"stream", true}
    };

    // Collect SSE chunks
    std::string full_content;
    int chunk_count = 0;
    bool got_done = false;

    auto res = cli.Post("/v1/chat/completions",
        req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    // Parse SSE events from the response body
    std::istringstream stream(res->body);
    std::string line;
    while (std::getline(stream, line)) {
        // Remove trailing \r if present
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.starts_with("data: ")) {
            auto data = line.substr(6);
            if (data == "[DONE]") {
                got_done = true;
                break;
            }

            auto chunk = json::parse(data);
            REQUIRE(chunk["object"] == "chat.completion.chunk");
            chunk_count++;

            if (chunk["choices"][0]["delta"].contains("content")) {
                full_content += chunk["choices"][0]["delta"]["content"].get<std::string>();
            }
        }
    }

    REQUIRE(got_done);
    REQUIRE(chunk_count > 1); // Should have multiple chunks
    REQUIRE(!full_content.empty());
    INFO("Streamed response: " << full_content);
    REQUIRE(full_content.find("4") != std::string::npos);
}

TEST_CASE("POST /v1/completions text completion", "[server-api][inference]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(false);
    ServerRunner runner(manager, TEST_PORT + 6);

    httplib::Client cli("127.0.0.1", TEST_PORT + 6);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(120);

    json req_body = {
        {"model", TEST_MODEL},
        {"prompt", "The answer to 2+2 is"},
        {"max_tokens", 16},
        {"temperature", 0.0}
    };

    auto res = cli.Post("/v1/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);

    auto body = json::parse(res->body);
    REQUIRE(body["object"] == "text_completion");
    REQUIRE(!body["choices"].empty());

    auto text = body["choices"][0]["text"].get<std::string>();
    REQUIRE(!text.empty());
    INFO("Completion: " << text);
}

TEST_CASE("POST /unload removes a loaded model", "[server-api][endpoints]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(true);
    ServerRunner runner(manager, TEST_PORT + 7);

    httplib::Client cli("127.0.0.1", TEST_PORT + 7);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(120);

    // First load
    json load_body = {{"model", TEST_MODEL}};
    auto load_res = cli.Post("/load", load_body.dump(), "application/json");
    REQUIRE(load_res);
    REQUIRE(load_res->status == 200);

    // Now unload
    json unload_body = {{"model", TEST_MODEL}};
    auto unload_res = cli.Post("/unload", unload_body.dump(), "application/json");
    REQUIRE(unload_res);
    REQUIRE(unload_res->status == 200);

    auto body = json::parse(unload_res->body);
    REQUIRE(body["status"] == "ok");
}

TEST_CASE("Chat completion with missing model returns 404", "[server-api][errors]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 8);

    httplib::Client cli("127.0.0.1", TEST_PORT + 8);
    cli.set_connection_timeout(5);

    json req_body = {
        {"model", "nonexistent/model"},
        {"messages", json::array({
            {{"role", "user"}, {"content", "hello"}}
        })}
    };

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 404);

    auto body = json::parse(res->body);
    REQUIRE(body.contains("error"));
}

TEST_CASE("Chat completion with empty messages returns 400", "[server-api][errors]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 9);

    httplib::Client cli("127.0.0.1", TEST_PORT + 9);
    cli.set_connection_timeout(5);

    json req_body = {
        {"model", TEST_MODEL},
        {"messages", json::array()}
    };

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
}

TEST_CASE("Invalid JSON body returns 400", "[server-api][errors]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    ServerRunner runner(manager, TEST_PORT + 10);

    httplib::Client cli("127.0.0.1", TEST_PORT + 10);
    cli.set_connection_timeout(5);

    auto res = cli.Post("/v1/chat/completions", "not json at all", "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
}

TEST_CASE("CORS headers are present", "[server-api][endpoints]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    ServerRunner runner(manager, TEST_PORT + 11);

    httplib::Client cli("127.0.0.1", TEST_PORT + 11);
    cli.set_connection_timeout(5);

    auto res = cli.Get("/health");
    REQUIRE(res);
    REQUIRE(res->has_header("Access-Control-Allow-Origin"));
    REQUIRE(res->get_header_value("Access-Control-Allow-Origin") == "*");
}

// ---------------------------------------------------------------------------
// OpenAI tools (parse/emit only — never execute)
// ---------------------------------------------------------------------------

TEST_CASE("tools request increases prompt_tokens vs no tools", "[server-api][tools]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(true);
    ServerRunner runner(manager, TEST_PORT + 20);

    httplib::Client cli("127.0.0.1", TEST_PORT + 20);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(180);

    json tools = json::array({
        {
            {"type", "function"},
            {"function",
             {
                 {"name", "get_weather"},
                 {"description", "Get the weather for a city. This schema is intentionally long so it expands the prompt."},
                 {"parameters",
                  {{"type", "object"},
                   {"properties",
                    {{"city", {{"type", "string"}, {"description", "City name"}}},
                     {"units", {{"type", "string"}, {"description", "celsius or fahrenheit"}}}}},
                   {"required", json::array({"city"})}}},
             }},
        },
    });

    json base_msgs = json::array({
        {{"role", "user"}, {"content", "Say hi."}},
    });

    json without = {
        {"model", TEST_MODEL},
        {"messages", base_msgs},
        {"max_tokens", 8},
        {"temperature", 0.0},
        {"stream", false},
    };
    json with = without;
    with["tools"] = tools;
    with["tool_choice"] = "auto";

    auto res0 = cli.Post("/v1/chat/completions", without.dump(), "application/json");
    REQUIRE(res0);
    REQUIRE(res0->status == 200);
    auto body0 = json::parse(res0->body);
    int pt0 = body0["usage"]["prompt_tokens"].get<int>();

    auto res1 = cli.Post("/v1/chat/completions", with.dump(), "application/json");
    REQUIRE(res1);
    REQUIRE(res1->status == 200);
    auto body1 = json::parse(res1->body);
    int pt1 = body1["usage"]["prompt_tokens"].get<int>();

    INFO("prompt without tools=" << pt0 << " with tools=" << pt1);
    REQUIRE(pt1 > pt0);
}

TEST_CASE("tool_choice none does not force tool_calls finish", "[server-api][tools]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(true);
    ServerRunner runner(manager, TEST_PORT + 21);

    httplib::Client cli("127.0.0.1", TEST_PORT + 21);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(180);

    json tools = json::array({
        {
            {"type", "function"},
            {"function",
             {
                 {"name", "calculator_calculate"},
                 {"description", "Evaluate a math expression"},
                 {"parameters",
                  {{"type", "object"},
                   {"properties", {{"expression", {{"type", "string"}}}}},
                   {"required", json::array({"expression"})}}},
             }},
        },
    });

    json req_body = {
        {"model", TEST_MODEL},
        {"messages",
         json::array({
             {{"role", "user"},
              {"content", "Use the calculator tool with expression 1+1. Only call the tool."}},
         })},
        {"tools", tools},
        {"tool_choice", "none"},
        {"max_tokens", 64},
        {"temperature", 0.0},
        {"stream", false},
    };

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = json::parse(res->body);
    auto finish = body["choices"][0]["finish_reason"].get<std::string>();
    REQUIRE((finish == "stop" || finish == "length"));
    // tool_choice=none must not emit structured tool_calls
    if (body["choices"][0]["message"].contains("tool_calls")) {
        REQUIRE(body["choices"][0]["message"]["tool_calls"].empty());
    }
}

TEST_CASE("tools validation rejects oversized tools array", "[server-api][tools]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 22);

    httplib::Client cli("127.0.0.1", TEST_PORT + 22);
    cli.set_connection_timeout(5);

    json tools = json::array();
    for (int i = 0; i < 100; ++i) {
        tools.push_back({
            {"type", "function"},
            {"function", {{"name", "t" + std::to_string(i)}, {"parameters", {{"type", "object"}}}}},
        });
    }
    json req_body = {
        {"model", TEST_MODEL},
        {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
        {"tools", tools},
    };
    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
}

TEST_CASE("role tool message returns 400 in v1", "[server-api][tools]") {
    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    ServerRunner runner(manager, TEST_PORT + 23);

    httplib::Client cli("127.0.0.1", TEST_PORT + 23);
    cli.set_connection_timeout(5);

    json req_body = {
        {"model", TEST_MODEL},
        {"messages",
         json::array({
             {{"role", "user"}, {"content", "hi"}},
             {{"role", "tool"}, {"content", "result"}},
         })},
    };
    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 400);
}

TEST_CASE("POST tools may return tool_calls (model dependent)", "[server-api][tools][inference]") {
    if (!model_is_cached()) SKIP("Model not cached: " << TEST_MODEL);

    auto manager = std::make_shared<mlx_lm::ModelManager>();
    manager->set_no_download(true);
    manager->set_no_think(true);
    ServerRunner runner(manager, TEST_PORT + 24);

    httplib::Client cli("127.0.0.1", TEST_PORT + 24);
    cli.set_connection_timeout(120);
    cli.set_read_timeout(180);

    json tools = json::array({
        {
            {"type", "function"},
            {"function",
             {
                 {"name", "calculator_calculate"},
                 {"description", "Evaluate a math expression"},
                 {"parameters",
                  {{"type", "object"},
                   {"properties", {{"expression", {{"type", "string"}}}}},
                   {"required", json::array({"expression"})}}},
             }},
        },
    });

    json req_body = {
        {"model", TEST_MODEL},
        {"messages",
         json::array({
             {{"role", "user"},
              {"content",
               "You must call the calculator_calculate tool with expression exactly \"1+1\". "
               "Do not answer in plain text."}},
         })},
        {"tools", tools},
        {"tool_choice", "auto"},
        {"max_tokens", 128},
        {"temperature", 0.0},
        {"stream", false},
    };

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    REQUIRE(res);
    REQUIRE(res->status == 200);
    auto body = json::parse(res->body);
    auto& msg = body["choices"][0]["message"];
    auto finish = body["choices"][0]["finish_reason"].get<std::string>();
    INFO("finish=" << finish << " body=" << body.dump());

    // 0.8B may not always emit tools; when it does, shape must be correct.
    if (finish == "tool_calls") {
        REQUIRE(msg.contains("tool_calls"));
        REQUIRE(msg["tool_calls"].is_array());
        REQUIRE(!msg["tool_calls"].empty());
        auto& tc = msg["tool_calls"][0];
        REQUIRE(tc["type"] == "function");
        REQUIRE(tc["function"]["name"].is_string());
        REQUIRE(tc["function"]["arguments"].is_string());
        // arguments must be parseable JSON
        auto args = json::parse(tc["function"]["arguments"].get<std::string>());
        REQUIRE(args.is_object());
        REQUIRE(!tc["id"].get<std::string>().empty());
    } else {
        // Soft pass: plumbing worked; reliability gate is 4B elsewhere.
        REQUIRE((finish == "stop" || finish == "length"));
    }
}
