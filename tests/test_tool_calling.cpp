// Tests for tool calling: parsers, processors, format inference, schema generation
// Ported from swift/Tests/MLXLMTests/ToolTests.swift

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/tool_calling.h>
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using json = nlohmann::json;

// ===========================================================================
// Tool Schema Generation
// ===========================================================================

TEST_CASE("Weather Tool Schema Generation", "[tool_calling]") {
    mlx_lm::Tool tool(
        "get_current_weather",
        "Get the current weather in a given location",
        {
            mlx_lm::ToolParameter::required(
                "location",
                mlx_lm::ToolParameterType::string_type(),
                "The city, e.g. Istanbul"),
            mlx_lm::ToolParameter::optional(
                "unit",
                mlx_lm::ToolParameterType::string_type(),
                "The unit of temperature",
                json{{"enum", json::array({"celsius", "fahrenheit"})}}),
        },
        [](const json& input) -> json {
            return json{{"temperature", 14.0}, {"conditions", "Sunny"}};
        });

    json expected = {
        {"type", "function"},
        {"function", {
            {"name", "get_current_weather"},
            {"description", "Get the current weather in a given location"},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"location", {
                        {"type", "string"},
                        {"description", "The city, e.g. Istanbul"},
                    }},
                    {"unit", {
                        {"type", "string"},
                        {"description", "The unit of temperature"},
                        {"enum", json::array({"celsius", "fahrenheit"})},
                    }},
                }},
                {"required", json::array({"location"})},
            }},
        }},
    };

    REQUIRE(tool.schema == expected);
}

// ===========================================================================
// JSON Format Tests
// ===========================================================================

TEST_CASE("JSON Tool Call Parser - Default Tags", "[tool_calling][json]") {
    mlx_lm::JSONToolCallParser parser("<tool_call>", "</tool_call>");
    std::string content =
        R"(<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>)";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "Paris");
}

TEST_CASE("JSON Tool Call Parser - LFM2 Tags", "[tool_calling][json]") {
    mlx_lm::JSONToolCallParser parser(
        "<|tool_call_start|>", "<|tool_call_end|>");
    std::string content =
        R"(<|tool_call_start|>{"name": "search", "arguments": {"query": "swift programming"}}<|tool_call_end|>)";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "search");
    CHECK(tool_call->function.arguments["query"] == "swift programming");
}

TEST_CASE("LFM2 Format via ToolCallProcessor", "[tool_calling][json]") {
    mlx_lm::ToolCallProcessor processor(mlx_lm::ToolCallFormat::lfm2);
    std::string content =
        R"(<|tool_call_start|>{"name": "calculator", "arguments": {"expression": "2+2"}}<|tool_call_end|>)";

    processor.process_chunk(content);

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();
    CHECK(tool_call.function.name == "calculator");
    CHECK(tool_call.function.arguments["expression"] == "2+2");
}

// ===========================================================================
// Streaming Tool Call Detection (Default JSON Format)
// ===========================================================================

TEST_CASE("Tool Call Detection - Streaming Chunks - Default JSON", "[tool_calling][streaming]") {
    mlx_lm::ToolCallProcessor processor;
    std::vector<std::string> chunks = {
        "<tool", "_", "call>", "{", "\"", "name", "\"", ":", " ", "\"", "get", "_", "current",
        "_", "weather", "\"", ",", " ", "\"", "arguments", "\"", ":", " ", "{", "\"",
        "location", "\"", ":", " ", "\"", "San", " Francisco", "\"", ",", " ", "\"", "unit",
        "\"", ":", " ", "\"", "celsius", "\"", "}", "}", "</tool", "_", "call>",
    };

    for (const auto& chunk : chunks) {
        auto result = processor.process_chunk(chunk);
        // During streaming, chunks are either buffered (nullopt) or partial text
        // The important thing is that after processing all chunks, we get a tool call
    }

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();

    CHECK(tool_call.function.name == "get_current_weather");
    CHECK(tool_call.function.arguments["location"] == "San Francisco");
    CHECK(tool_call.function.arguments["unit"] == "celsius");
}

// ===========================================================================
// XML Function Format Tests (Qwen3 Coder)
// ===========================================================================

TEST_CASE("XML Function Parser - Qwen3 Coder Format", "[tool_calling][xml]") {
    mlx_lm::XMLFunctionParser parser;
    std::string content =
        "<function=get_weather><parameter=location>Tokyo</parameter>"
        "<parameter=unit>celsius</parameter></function>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "Tokyo");
    CHECK(tool_call->function.arguments["unit"] == "celsius");
}

TEST_CASE("XML Function Parser - With Type Conversion", "[tool_calling][xml]") {
    mlx_lm::XMLFunctionParser parser;
    json tools = json::array({
        {{"function", {
            {"name", "set_temperature"},
            {"parameters", {
                {"properties", {
                    {"value", {{"type", "integer"}}},
                    {"enabled", {{"type", "boolean"}}},
                }},
            }},
        }}},
    });

    std::string content =
        "<function=set_temperature><parameter=value>25</parameter>"
        "<parameter=enabled>true</parameter></function>";

    auto tool_call = parser.parse(content, tools);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "set_temperature");
    CHECK(tool_call->function.arguments["value"] == 25);
    CHECK(tool_call->function.arguments["enabled"] == true);
}

// ===========================================================================
// GLM4 Format Tests
// ===========================================================================

TEST_CASE("GLM4 Tool Call Parser", "[tool_calling][glm4]") {
    mlx_lm::GLM4ToolCallParser parser;
    std::string content =
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Berlin</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value></tool_call>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "Berlin");
    CHECK(tool_call->function.arguments["unit"] == "celsius");
}

TEST_CASE("GLM4 Format via ToolCallProcessor", "[tool_calling][glm4]") {
    mlx_lm::ToolCallProcessor processor(mlx_lm::ToolCallFormat::glm4);
    std::string content =
        "<tool_call>search<arg_key>query</arg_key><arg_value>machine learning</arg_value></tool_call>";

    processor.process_chunk(content);

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();
    CHECK(tool_call.function.name == "search");
    CHECK(tool_call.function.arguments["query"] == "machine learning");
}

// ===========================================================================
// Gemma Format Tests
// ===========================================================================

TEST_CASE("Gemma Function Parser", "[tool_calling][gemma]") {
    mlx_lm::GemmaFunctionParser parser;
    std::string content =
        "<start_function_call>call:get_weather{location:Paris,unit:celsius}<end_function_call>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "Paris");
    CHECK(tool_call->function.arguments["unit"] == "celsius");
}

TEST_CASE("Gemma Function Parser - Escaped Strings", "[tool_calling][gemma]") {
    mlx_lm::GemmaFunctionParser parser;
    // Note: Gemma uses <escape> for both start and end markers (not </escape>)
    std::string content =
        "<start_function_call>call:search{query:<escape>hello, world!<escape>}<end_function_call>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "search");
    CHECK(tool_call->function.arguments["query"] == "hello, world!");
}

TEST_CASE("Gemma Format via ToolCallProcessor", "[tool_calling][gemma]") {
    mlx_lm::ToolCallProcessor processor(mlx_lm::ToolCallFormat::gemma);
    std::string content =
        "<start_function_call>call:calculator{expression:2+2}<end_function_call>";

    processor.process_chunk(content);

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();
    CHECK(tool_call.function.name == "calculator");
    CHECK(tool_call.function.arguments["expression"] == "2+2");
}

// ===========================================================================
// Kimi K2 Format Tests
// ===========================================================================

TEST_CASE("Kimi K2 Tool Call Parser", "[tool_calling][kimi_k2]") {
    mlx_lm::KimiK2ToolCallParser parser;
    std::string content =
        "<|tool_calls_section_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>{\"location\": \"London\"}"
        "<|tool_calls_section_end|>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "London");
}

TEST_CASE("Kimi K2 Format via ToolCallProcessor", "[tool_calling][kimi_k2]") {
    mlx_lm::ToolCallProcessor processor(mlx_lm::ToolCallFormat::kimi_k2);
    std::string content =
        "<|tool_calls_section_begin|>functions.search:0"
        "<|tool_call_argument_begin|>{\"query\": \"swift\"}"
        "<|tool_calls_section_end|>";

    processor.process_chunk(content);

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();
    CHECK(tool_call.function.name == "search");
    CHECK(tool_call.function.arguments["query"] == "swift");
}

// ===========================================================================
// MiniMax M2 Format Tests
// ===========================================================================

TEST_CASE("MiniMax M2 Tool Call Parser", "[tool_calling][minimax_m2]") {
    mlx_lm::MiniMaxM2ToolCallParser parser;
    std::string content =
        "<minimax:tool_call><invoke name=\"get_weather\">"
        "<parameter name=\"location\">Sydney</parameter>"
        "</invoke></minimax:tool_call>";

    auto tool_call = parser.parse(content, std::nullopt);

    REQUIRE(tool_call.has_value());
    CHECK(tool_call->function.name == "get_weather");
    CHECK(tool_call->function.arguments["location"] == "Sydney");
}

TEST_CASE("MiniMax M2 Format via ToolCallProcessor", "[tool_calling][minimax_m2]") {
    mlx_lm::ToolCallProcessor processor(mlx_lm::ToolCallFormat::minimax_m2);
    std::string content =
        "<minimax:tool_call><invoke name=\"search\">"
        "<parameter name=\"query\">AI news</parameter>"
        "</invoke></minimax:tool_call>";

    processor.process_chunk(content);

    REQUIRE(processor.tool_calls().size() == 1);
    auto& tool_call = processor.tool_calls().front();
    CHECK(tool_call.function.name == "search");
    CHECK(tool_call.function.arguments["query"] == "AI news");
}

// ===========================================================================
// ToolCallFormat Serialization Tests
// ===========================================================================

TEST_CASE("ToolCallFormat string round-trip", "[tool_calling][format]") {
    SECTION("json") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::json) == "json");
    }
    SECTION("lfm2") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::lfm2) == "lfm2");
    }
    SECTION("xml_function") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::xml_function) == "xml_function");
    }
    SECTION("glm4") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::glm4) == "glm4");
    }
    SECTION("gemma") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::gemma) == "gemma");
    }
    SECTION("kimi_k2") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::kimi_k2) == "kimi_k2");
    }
    SECTION("minimax_m2") {
        CHECK(mlx_lm::to_string(mlx_lm::ToolCallFormat::minimax_m2) == "minimax_m2");
    }

    SECTION("round-trip all formats via from_string") {
        std::vector<mlx_lm::ToolCallFormat> all_formats = {
            mlx_lm::ToolCallFormat::json,
            mlx_lm::ToolCallFormat::lfm2,
            mlx_lm::ToolCallFormat::xml_function,
            mlx_lm::ToolCallFormat::glm4,
            mlx_lm::ToolCallFormat::gemma,
            mlx_lm::ToolCallFormat::kimi_k2,
            mlx_lm::ToolCallFormat::minimax_m2,
        };
        for (auto fmt : all_formats) {
            auto s = mlx_lm::to_string(fmt);
            auto parsed = mlx_lm::tool_call_format_from_string(s);
            REQUIRE(parsed.has_value());
            CHECK(parsed.value() == fmt);
        }
    }
}

// ===========================================================================
// Format Inference Tests
// ===========================================================================

TEST_CASE("ToolCallFormat Inference from model type", "[tool_calling][format]") {
    SECTION("LFM2 models") {
        CHECK(mlx_lm::infer_tool_call_format("lfm2") == mlx_lm::ToolCallFormat::lfm2);
        CHECK(mlx_lm::infer_tool_call_format("LFM2") == mlx_lm::ToolCallFormat::lfm2);
        CHECK(mlx_lm::infer_tool_call_format("lfm2_moe") == mlx_lm::ToolCallFormat::lfm2);
    }

    SECTION("GLM4 models") {
        CHECK(mlx_lm::infer_tool_call_format("glm4") == mlx_lm::ToolCallFormat::glm4);
        CHECK(mlx_lm::infer_tool_call_format("glm4_moe") == mlx_lm::ToolCallFormat::glm4);
        CHECK(mlx_lm::infer_tool_call_format("glm4_moe_lite") == mlx_lm::ToolCallFormat::glm4);
    }

    SECTION("Gemma models") {
        CHECK(mlx_lm::infer_tool_call_format("gemma") == mlx_lm::ToolCallFormat::gemma);
        CHECK(mlx_lm::infer_tool_call_format("GEMMA") == mlx_lm::ToolCallFormat::gemma);
    }

    SECTION("Unknown models return nullopt") {
        CHECK(!mlx_lm::infer_tool_call_format("llama").has_value());
        CHECK(!mlx_lm::infer_tool_call_format("mistral").has_value());
        // Qwen family → XML function format (not JSON <tool_call> tags)
        CHECK(mlx_lm::infer_tool_call_format("qwen2") ==
              mlx_lm::ToolCallFormat::xml_function);
        CHECK(mlx_lm::infer_tool_call_format("qwen3") ==
              mlx_lm::ToolCallFormat::xml_function);
        CHECK(mlx_lm::infer_tool_call_format("qwen3_5") ==
              mlx_lm::ToolCallFormat::xml_function);
    }
}

// ===========================================================================
// Tool Execution Tests
// ===========================================================================

TEST_CASE("Tool execution with matching name", "[tool_calling]") {
    mlx_lm::Tool tool(
        "add",
        "Add two numbers",
        {
            mlx_lm::ToolParameter::required(
                "a", mlx_lm::ToolParameterType::int_type(), "First number"),
            mlx_lm::ToolParameter::required(
                "b", mlx_lm::ToolParameterType::int_type(), "Second number"),
        },
        [](const json& input) -> json {
            return json{{"result", input["a"].get<int>() + input["b"].get<int>()}};
        });

    mlx_lm::ToolCall call(mlx_lm::ToolCall::Function("add", json{{"a", 3}, {"b", 4}}));
    auto result = tool.execute(call);
    CHECK(result["result"] == 7);
}

TEST_CASE("Tool execution with mismatched name throws", "[tool_calling]") {
    mlx_lm::Tool tool(
        "add",
        "Add two numbers",
        {},
        [](const json&) -> json { return nullptr; });

    mlx_lm::ToolCall call(mlx_lm::ToolCall::Function("subtract", json::object()));
    REQUIRE_THROWS(tool.execute(call));
}

// ===========================================================================
// Parser Factory Tests
// ===========================================================================

TEST_CASE("create_parser returns correct parser types", "[tool_calling][format]") {
    SECTION("json format has default tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::json);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<tool_call>");
        CHECK(parser->end_tag() == "</tool_call>");
    }

    SECTION("lfm2 format has pipe-delimited tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::lfm2);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<|tool_call_start|>");
        CHECK(parser->end_tag() == "<|tool_call_end|>");
    }

    SECTION("xml_function format is inline (no wrapper tags)") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::xml_function);
        REQUIRE(parser != nullptr);
        CHECK(!parser->start_tag().has_value());
        CHECK(!parser->end_tag().has_value());
    }

    SECTION("glm4 format has tool_call tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::glm4);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<tool_call>");
        CHECK(parser->end_tag() == "</tool_call>");
    }

    SECTION("gemma format has function call tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::gemma);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<start_function_call>");
        CHECK(parser->end_tag() == "<end_function_call>");
    }

    SECTION("kimi_k2 format has section tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::kimi_k2);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<|tool_calls_section_begin|>");
        CHECK(parser->end_tag() == "<|tool_calls_section_end|>");
    }

    SECTION("minimax_m2 format has minimax tags") {
        auto parser = mlx_lm::create_parser(mlx_lm::ToolCallFormat::minimax_m2);
        REQUIRE(parser != nullptr);
        CHECK(parser->start_tag() == "<minimax:tool_call>");
        CHECK(parser->end_tag() == "</minimax:tool_call>");
    }
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST_CASE("JSON parser returns nullopt for invalid JSON", "[tool_calling][json]") {
    mlx_lm::JSONToolCallParser parser("<tool_call>", "</tool_call>");
    auto result = parser.parse("<tool_call>not json</tool_call>", std::nullopt);
    CHECK(!result.has_value());
}

TEST_CASE("JSON parser returns nullopt for JSON without name field", "[tool_calling][json]") {
    mlx_lm::JSONToolCallParser parser("<tool_call>", "</tool_call>");
    auto result = parser.parse(
        R"(<tool_call>{"arguments": {"x": 1}}</tool_call>)", std::nullopt);
    CHECK(!result.has_value());
}

TEST_CASE("tool_call_format_from_string returns nullopt for unknown format", "[tool_calling][format]") {
    CHECK(!mlx_lm::tool_call_format_from_string("unknown").has_value());
    CHECK(!mlx_lm::tool_call_format_from_string("").has_value());
}
