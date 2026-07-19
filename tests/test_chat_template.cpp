// Tests for ChatTemplate: Jinja2 rendering of HuggingFace chat templates.

#include <catch2/catch_test_macros.hpp>
#include <mlx-lm/common/chat_template.h>

#include <string>
#include <unordered_map>
#include <vector>

using namespace mlx_lm;

// Qwen3 chat template (from tokenizer_config.json).
static const char* QWEN3_TEMPLATE = R"({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {{- '<|im_start|>' + message.role + '\n' + content }}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %})";

// Minimal tokenizer_config with eos_token
static nlohmann::json make_qwen3_config() {
    return {
        {"eos_token", "<|im_end|>"},
        {"bos_token", nullptr},
    };
}

// ===========================================================================
// Template Construction Tests
// ===========================================================================

TEST_CASE("ChatTemplate construction from template string", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    CHECK(tmpl.eos_token() == "<|im_end|>");
    CHECK(tmpl.bos_token().empty());
    CHECK(!tmpl.template_string().empty());
}

TEST_CASE("ChatTemplate with null eos_token", "[chat_template]") {
    nlohmann::json config = {{"eos_token", nullptr}, {"bos_token", nullptr}};
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    CHECK(tmpl.eos_token().empty());
    CHECK(tmpl.bos_token().empty());
}

// ===========================================================================
// Template Rendering Tests
// ===========================================================================

TEST_CASE("ChatTemplate renders simple user message", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "user"}, {"content", "Hello!"}},
    };

    auto result = tmpl.apply(messages, true);

    // Should contain the user message with im_start/im_end markers
    CHECK(result.find("<|im_start|>user\nHello!<|im_end|>") != std::string::npos);

    // Should end with assistant prompt since add_generation_prompt=true
    CHECK(result.find("<|im_start|>assistant\n") != std::string::npos);
}

TEST_CASE("ChatTemplate renders system + user messages", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "system"}, {"content", "You are a helpful assistant."}},
        {{"role", "user"}, {"content", "What is 2+2?"}},
    };

    auto result = tmpl.apply(messages, true);

    // System message should be present
    CHECK(result.find("<|im_start|>system\nYou are a helpful assistant.<|im_end|>") != std::string::npos);

    // User message should follow
    CHECK(result.find("<|im_start|>user\nWhat is 2+2?<|im_end|>") != std::string::npos);

    // Should end with assistant prompt
    CHECK(result.find("<|im_start|>assistant\n") != std::string::npos);
}

TEST_CASE("ChatTemplate renders multi-turn conversation", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "user"}, {"content", "Hi"}},
        {{"role", "assistant"}, {"content", "Hello!"}},
        {{"role", "user"}, {"content", "How are you?"}},
    };

    auto result = tmpl.apply(messages, true);

    // All messages present in order
    CHECK(result.find("<|im_start|>user\nHi<|im_end|>") != std::string::npos);
    CHECK(result.find("<|im_start|>assistant\nHello!<|im_end|>") != std::string::npos);
    CHECK(result.find("<|im_start|>user\nHow are you?<|im_end|>") != std::string::npos);
}

TEST_CASE("ChatTemplate add_generation_prompt=false omits assistant", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "user"}, {"content", "Hi"}},
    };

    auto result = tmpl.apply(messages, false);

    // Should NOT end with assistant prompt
    auto last_pos = result.rfind("<|im_start|>assistant");
    CHECK(last_pos == std::string::npos);
}

TEST_CASE("ChatTemplate enable_thinking=false suppresses thinking", "[chat_template]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "user"}, {"content", "Hi"}},
    };

    nlohmann::json extra = {{"enable_thinking", false}};
    auto result = tmpl.apply(messages, true, extra);

    // Should contain the empty think block
    CHECK(result.find("<think>\n\n</think>") != std::string::npos);
}

TEST_CASE("ChatTemplate injects tools into prompt when tools provided", "[chat_template][tools]") {
    auto config = make_qwen3_config();
    ChatTemplate tmpl(QWEN3_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "user"}, {"content", "What is the weather?"}},
    };

    nlohmann::json tools = nlohmann::json::array({
        {
            {"type", "function"},
            {"function",
             {
                 {"name", "get_weather"},
                 {"description", "Get weather for a city"},
                 {"parameters",
                  {{"type", "object"},
                   {"properties",
                    {{"city", {{"type", "string"}, {"description", "City name"}}}}},
                   {"required", nlohmann::json::array({"city"})}}},
             }},
        },
    });

    auto without = tmpl.apply(messages, true, {}, nullptr);
    auto with = tmpl.apply(messages, true, {}, &tools);

    // Tools must increase prompt content (schemas present).
    CHECK(with.size() > without.size());
    CHECK(with.find("get_weather") != std::string::npos);
    // Qwen tools branch typically uses tools markup or tool schema text.
    CHECK((with.find("tool") != std::string::npos ||
           with.find("Tool") != std::string::npos ||
           with.find("get_weather") != std::string::npos));
}

// ===========================================================================
// load_chat_template Tests
// ===========================================================================

TEST_CASE("load_chat_template from nonexistent directory returns nullopt", "[chat_template]") {
    auto tmpl = load_chat_template("/nonexistent/path/to/model");
    CHECK(!tmpl.has_value());
}

// ===========================================================================
// Simple ChatML Template Test
// ===========================================================================

static const char* SIMPLE_CHATML_TEMPLATE =
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

TEST_CASE("Simple ChatML template renders correctly", "[chat_template]") {
    nlohmann::json config = {
        {"eos_token", "<|im_end|>"},
        {"bos_token", "<|im_start|>"},
    };
    ChatTemplate tmpl(SIMPLE_CHATML_TEMPLATE, config);

    std::vector<Message> messages = {
        {{"role", "system"}, {"content", "You are helpful."}},
        {{"role", "user"}, {"content", "Hello"}},
    };

    auto result = tmpl.apply(messages, true);

    CHECK(result.find("<|im_start|>system\nYou are helpful.<|im_end|>") != std::string::npos);
    CHECK(result.find("<|im_start|>user\nHello<|im_end|>") != std::string::npos);
    CHECK(result.find("<|im_start|>assistant\n") != std::string::npos);
}
