// Copyright (c) 2024-2025 Apple Inc. -- Ported to C++

#include "mlx-lm/common/tool_calling.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mlx_lm {

// ===========================================================================
// ToolParameterType
// ===========================================================================

nlohmann::json ToolParameterType::to_schema() const {
    switch (tag) {
    case kString:
        return {{"type", "string"}};
    case kBool:
        return {{"type", "boolean"}};
    case kInt:
        return {{"type", "integer"}};
    case kDouble:
        return {{"type", "number"}};
    case kData:
        return {{"type", "string"}, {"contentEncoding", "base64"}};
    case kArray: {
        nlohmann::json schema = {{"type", "array"}};
        if (element_type) {
            schema["items"] = element_type->to_schema();
        }
        return schema;
    }
    case kObject: {
        nlohmann::json schema = {{"type", "object"}};
        if (properties) {
            nlohmann::json props = nlohmann::json::object();
            std::vector<std::string> required_list;
            for (const auto& param : *properties) {
                props[param.name] = param.to_schema();
                if (param.is_required) {
                    required_list.push_back(param.name);
                }
            }
            schema["properties"] = std::move(props);
            schema["required"] = std::move(required_list);
        }
        return schema;
    }
    }
    return {{"type", "string"}};  // fallback
}

// ===========================================================================
// ToolParameter
// ===========================================================================

nlohmann::json ToolParameter::to_schema() const {
    nlohmann::json schema = type.to_schema();
    schema["description"] = description;
    // Merge extra properties
    if (extra_properties.is_object()) {
        for (auto& [key, val] : extra_properties.items()) {
            schema[key] = val;
        }
    }
    return schema;
}

ToolParameter ToolParameter::required(
    const std::string& name,
    ToolParameterType type,
    const std::string& description,
    nlohmann::json extra)
{
    return {name, std::move(type), description, true, std::move(extra)};
}

ToolParameter ToolParameter::optional(
    const std::string& name,
    ToolParameterType type,
    const std::string& description,
    nlohmann::json extra)
{
    return {name, std::move(type), description, false, std::move(extra)};
}

// ===========================================================================
// Tool
// ===========================================================================

std::string Tool::name() const {
    if (schema.contains("function") && schema["function"].contains("name")) {
        return schema["function"]["name"].get<std::string>();
    }
    return "";
}

Tool::Tool(
    const std::string& name,
    const std::string& description,
    const std::vector<ToolParameter>& parameters,
    std::function<nlohmann::json(const nlohmann::json&)> handler)
    : handler(std::move(handler))
{
    nlohmann::json props = nlohmann::json::object();
    std::vector<std::string> required_params;

    for (const auto& param : parameters) {
        props[param.name] = param.to_schema();
        if (param.is_required) {
            required_params.push_back(param.name);
        }
    }

    schema = {
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", description},
            {"parameters", {
                {"type", "object"},
                {"properties", std::move(props)},
                {"required", std::move(required_params)},
            }},
        }},
    };
}

Tool::Tool(
    ToolSpec schema,
    std::function<nlohmann::json(const nlohmann::json&)> handler)
    : schema(std::move(schema)), handler(std::move(handler))
{}

nlohmann::json Tool::execute(const ToolCall& call) const {
    if (name() != call.function.name) {
        throw std::runtime_error(
            "Tool name mismatch: expected '" + name() +
            "' but got '" + call.function.name + "'");
    }
    return handler(call.function.arguments);
}

// ===========================================================================
// detail:: Parser Utilities
// ===========================================================================

namespace detail {

// Helper: lowercase a string
static std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Helper: trim whitespace from both ends
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(start, end - start + 1);
}

nlohmann::json deserialize(const std::string& value) {
    try {
        return nlohmann::json::parse(value);
    } catch (...) {
        return value;
    }
}

bool is_string_type(
    const std::string& func_name,
    const std::string& arg_name,
    const std::optional<nlohmann::json>& tools)
{
    auto type_str = get_parameter_type(func_name, arg_name, tools);
    return type_str.has_value() && type_str.value() == "string";
}

std::optional<std::string> get_parameter_type(
    const std::string& func_name,
    const std::string& param_name,
    const std::optional<nlohmann::json>& tools)
{
    if (!tools.has_value() || !tools->is_array()) return std::nullopt;

    for (const auto& tool : *tools) {
        if (!tool.contains("function")) continue;
        const auto& function = tool["function"];
        if (!function.contains("name") ||
            function["name"].get<std::string>() != func_name) continue;
        if (!function.contains("parameters")) continue;
        const auto& parameters = function["parameters"];
        if (!parameters.contains("properties")) continue;
        const auto& properties = parameters["properties"];
        if (!properties.contains(param_name)) continue;
        const auto& param = properties[param_name];
        if (param.contains("type")) {
            return param["type"].get<std::string>();
        }
    }
    return std::nullopt;
}

nlohmann::json get_parameter_config(
    const std::string& func_name,
    const std::optional<nlohmann::json>& tools)
{
    if (!tools.has_value() || !tools->is_array()) return nlohmann::json::object();

    for (const auto& tool : *tools) {
        if (!tool.contains("function")) continue;
        const auto& function = tool["function"];
        if (!function.contains("name") ||
            function["name"].get<std::string>() != func_name) continue;
        if (!function.contains("parameters")) continue;
        const auto& parameters = function["parameters"];
        if (parameters.contains("properties")) {
            return parameters["properties"];
        }
    }
    return nlohmann::json::object();
}

std::vector<std::string> extract_types_from_schema(const nlohmann::json& schema) {
    if (schema.is_null()) return {"string"};

    std::set<std::string> types;

    // Handle direct "type" field
    if (schema.contains("type")) {
        const auto& type_val = schema["type"];
        if (type_val.is_string()) {
            types.insert(type_val.get<std::string>());
        } else if (type_val.is_array()) {
            for (const auto& t : type_val) {
                if (t.is_string()) {
                    types.insert(t.get<std::string>());
                }
            }
        }
    }

    // Handle enum -- infer types from enum values
    if (schema.contains("enum") && schema["enum"].is_array()) {
        for (const auto& val : schema["enum"]) {
            if (val.is_null())           types.insert("null");
            else if (val.is_boolean())   types.insert("boolean");
            else if (val.is_number_integer()) types.insert("integer");
            else if (val.is_number_float())   types.insert("number");
            else if (val.is_string())    types.insert("string");
            else if (val.is_array())     types.insert("array");
            else if (val.is_object())    types.insert("object");
        }
    }

    // Handle anyOf, oneOf, allOf -- recursively extract types
    for (const auto& choice_field : {"anyOf", "oneOf", "allOf"}) {
        if (schema.contains(choice_field) && schema[choice_field].is_array()) {
            for (const auto& choice : schema[choice_field]) {
                auto sub_types = extract_types_from_schema(choice);
                types.insert(sub_types.begin(), sub_types.end());
            }
        }
    }

    if (types.empty()) return {"string"};
    return {types.begin(), types.end()};
}

nlohmann::json convert_value_with_types(
    const std::string& value, const std::vector<std::string>& types)
{
    std::string lower_value = to_lower(value);

    // Handle null values
    if (lower_value == "null" || lower_value == "none" || lower_value == "nil") {
        return nullptr;
    }

    std::set<std::string> normalized;
    for (const auto& t : types) {
        normalized.insert(to_lower(t));
    }

    // Priority order: integer > number > boolean > object > array > string
    static const std::vector<std::string> type_priority = {
        "integer", "int", "number", "float", "boolean", "bool",
        "object", "array", "string", "str", "text",
    };

    for (const auto& param_type : type_priority) {
        if (normalized.find(param_type) == normalized.end()) continue;

        if (param_type == "string" || param_type == "str" || param_type == "text") {
            return value;
        }

        if (param_type == "integer" || param_type == "int") {
            try {
                size_t pos = 0;
                long long int_val = std::stoll(value, &pos);
                if (pos == value.size()) {
                    return int_val;
                }
            } catch (...) {}
        }

        if (param_type == "number" || param_type == "float") {
            try {
                size_t pos = 0;
                double float_val = std::stod(value, &pos);
                if (pos == value.size()) {
                    long long int_val = static_cast<long long>(float_val);
                    if (float_val != static_cast<double>(int_val)) {
                        return float_val;
                    }
                    return int_val;
                }
            } catch (...) {}
        }

        if (param_type == "boolean" || param_type == "bool") {
            std::string trimmed = trim(lower_value);
            if (trimmed == "true" || trimmed == "1" ||
                trimmed == "yes" || trimmed == "on") {
                return true;
            }
            if (trimmed == "false" || trimmed == "0" ||
                trimmed == "no" || trimmed == "off") {
                return false;
            }
        }

        if (param_type == "object" || param_type == "array") {
            try {
                return nlohmann::json::parse(value);
            } catch (...) {}
        }
    }

    // Fallback: try JSON parse, then return as string
    try {
        return nlohmann::json::parse(value);
    } catch (...) {
        return value;
    }
}

nlohmann::json convert_parameter_value(
    const std::string& value,
    const std::string& param_name,
    const std::string& func_name,
    const std::optional<nlohmann::json>& tools)
{
    auto param_type_opt = get_parameter_type(func_name, param_name, tools);
    if (!param_type_opt.has_value()) {
        return value;
    }

    std::string type = to_lower(param_type_opt.value());

    // String types -- return as-is
    static const std::set<std::string> string_types = {
        "string", "str", "text", "varchar", "char", "enum"
    };
    if (string_types.count(type)) {
        return value;
    }

    // Integer types
    if (type.substr(0, 3) == "int" || type.substr(0, 4) == "uint" ||
        type.substr(0, 4) == "long" || type.substr(0, 5) == "short" ||
        type.substr(0, 8) == "unsigned")
    {
        try {
            size_t pos = 0;
            long long int_val = std::stoll(value, &pos);
            if (pos == value.size()) return int_val;
        } catch (...) {}
        return value;
    }

    // Float types
    if (type.substr(0, 3) == "num" || type.substr(0, 5) == "float") {
        try {
            size_t pos = 0;
            double float_val = std::stod(value, &pos);
            if (pos == value.size()) {
                long long int_val = static_cast<long long>(float_val);
                if (float_val != static_cast<double>(int_val)) {
                    return float_val;
                }
                return int_val;
            }
        } catch (...) {}
        return value;
    }

    // Boolean types
    if (type == "boolean" || type == "bool" || type == "binary") {
        return to_lower(value) == "true";
    }

    // Object/Array types -- JSON decode
    if (type == "object" || type == "array" ||
        type.substr(0, 4) == "dict" || type.substr(0, 4) == "list")
    {
        try {
            return nlohmann::json::parse(value);
        } catch (...) {}
    }

    return value;
}

std::string extract_name(const std::string& name_str) {
    std::string trimmed = trim(name_str);
    if (trimmed.size() >= 2) {
        if ((trimmed.front() == '"' && trimmed.back() == '"') ||
            (trimmed.front() == '\'' && trimmed.back() == '\''))
        {
            return trimmed.substr(1, trimmed.size() - 2);
        }
    }
    return trimmed;
}

}  // namespace detail

// ===========================================================================
// Helper: find substring position (returns npos if not found)
// ===========================================================================

static std::string::size_type find_str(
    const std::string& haystack, const std::string& needle,
    std::string::size_type pos = 0)
{
    return haystack.find(needle, pos);
}

// Helper: replace all occurrences of a substring
static std::string replace_all(
    std::string s, const std::string& from, const std::string& to)
{
    if (from.empty()) return s;
    std::string::size_type pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// ===========================================================================
// JSONToolCallParser
// ===========================================================================

JSONToolCallParser::JSONToolCallParser(std::string start, std::string end)
    : start_tag_(std::move(start)), end_tag_(std::move(end))
{}

std::optional<std::string> JSONToolCallParser::start_tag() const {
    return start_tag_;
}

std::optional<std::string> JSONToolCallParser::end_tag() const {
    return end_tag_;
}

std::optional<ToolCall> JSONToolCallParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& /*tools*/) const
{
    std::string text = content;

    // Strip tags if present
    auto start_pos = find_str(text, start_tag_);
    if (start_pos != std::string::npos) {
        text = text.substr(start_pos + start_tag_.size());
    }
    auto end_pos = find_str(text, end_tag_);
    if (end_pos != std::string::npos) {
        text = text.substr(0, end_pos);
    }

    text = detail::trim(text);
    if (text.empty()) return std::nullopt;

    try {
        auto j = nlohmann::json::parse(text);
        if (!j.contains("name") || !j["name"].is_string()) return std::nullopt;

        std::string name = j["name"].get<std::string>();
        nlohmann::json arguments = nlohmann::json::object();
        if (j.contains("arguments") && j["arguments"].is_object()) {
            arguments = j["arguments"];
        }

        return ToolCall(ToolCall::Function(std::move(name), std::move(arguments)));
    } catch (...) {
        return std::nullopt;
    }
}

// ===========================================================================
// GLM4ToolCallParser
// ===========================================================================

std::optional<std::string> GLM4ToolCallParser::start_tag() const {
    return "<tool_call>";
}

std::optional<std::string> GLM4ToolCallParser::end_tag() const {
    return "</tool_call>";
}

std::optional<ToolCall> GLM4ToolCallParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& tools) const
{
    std::string text = content;
    text = replace_all(text, "<tool_call>", "");
    text = replace_all(text, "</tool_call>", "");
    text = detail::trim(text);

    // Extract function name (everything before first <arg_key>)
    auto arg_key_start = find_str(text, "<arg_key>");
    if (arg_key_start == std::string::npos) return std::nullopt;

    std::string func_name = detail::trim(text.substr(0, arg_key_start));
    if (func_name.empty()) return std::nullopt;

    nlohmann::json arguments = nlohmann::json::object();

    // Find all arg_key/arg_value pairs
    std::string::size_type search_pos = 0;
    while (true) {
        auto ks = find_str(text, "<arg_key>", search_pos);
        if (ks == std::string::npos) break;

        auto ke = find_str(text, "</arg_key>", ks + 9);
        if (ke == std::string::npos) break;

        std::string key = detail::trim(text.substr(ks + 9, ke - ks - 9));

        auto vs = find_str(text, "<arg_value>", ke + 10);
        if (vs == std::string::npos) break;

        auto ve = find_str(text, "</arg_value>", vs + 11);
        if (ve == std::string::npos) break;

        std::string value = detail::trim(text.substr(vs + 11, ve - vs - 11));

        // GLM4: deserialize if NOT a string type in schema
        if (!detail::is_string_type(func_name, key, tools)) {
            arguments[key] = detail::deserialize(value);
        } else {
            arguments[key] = value;
        }

        search_pos = ve + 12;  // length of "</arg_value>"
    }

    return ToolCall(ToolCall::Function(std::move(func_name), std::move(arguments)));
}

// ===========================================================================
// GemmaFunctionParser
// ===========================================================================

std::optional<std::string> GemmaFunctionParser::start_tag() const {
    return "<start_function_call>";
}

std::optional<std::string> GemmaFunctionParser::end_tag() const {
    return "<end_function_call>";
}

std::optional<ToolCall> GemmaFunctionParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& /*tools*/) const
{
    std::string text = content;
    text = replace_all(text, "<start_function_call>", "");
    text = replace_all(text, "<end_function_call>", "");

    // Find "call:" followed by function name and arguments
    auto call_pos = find_str(text, "call:");
    if (call_pos == std::string::npos) return std::nullopt;

    std::string remaining = text.substr(call_pos + 5);  // length of "call:"

    // Extract function name (word characters until {)
    auto brace_start = remaining.find('{');
    if (brace_start == std::string::npos) return std::nullopt;

    std::string func_name = remaining.substr(0, brace_start);
    if (func_name.empty()) return std::nullopt;

    // Extract arguments string (everything between { and last })
    auto brace_end = remaining.rfind('}');
    if (brace_end == std::string::npos || brace_end <= brace_start) return std::nullopt;

    std::string args_str = remaining.substr(brace_start + 1, brace_end - brace_start - 1);

    nlohmann::json arguments = nlohmann::json::object();

    const std::string escape_marker = "<escape>";

    // Parse key:value pairs
    while (!args_str.empty()) {
        // Find the key (everything before :)
        auto colon_idx = args_str.find(':');
        if (colon_idx == std::string::npos) break;

        std::string key = args_str.substr(0, colon_idx);
        args_str = args_str.substr(colon_idx + 1);

        // Handle escaped strings
        if (args_str.substr(0, escape_marker.size()) == escape_marker) {
            args_str = args_str.substr(escape_marker.size());
            auto end_escape = find_str(args_str, escape_marker);
            if (end_escape == std::string::npos) break;

            std::string value = args_str.substr(0, end_escape);
            arguments[key] = value;
            args_str = args_str.substr(end_escape + escape_marker.size());
            // Skip comma if present
            if (!args_str.empty() && args_str[0] == ',') {
                args_str = args_str.substr(1);
            }
            continue;
        }

        // Handle regular values (until comma or end)
        auto comma_idx = args_str.find(',');
        std::string value;
        if (comma_idx != std::string::npos) {
            value = args_str.substr(0, comma_idx);
            args_str = args_str.substr(comma_idx + 1);
        } else {
            value = args_str;
            args_str.clear();
        }

        // Try JSON decode, fallback to string
        try {
            arguments[key] = nlohmann::json::parse(value);
        } catch (...) {
            arguments[key] = value;
        }
    }

    return ToolCall(ToolCall::Function(std::move(func_name), std::move(arguments)));
}

// ===========================================================================
// XMLFunctionParser
// ===========================================================================

std::optional<std::string> XMLFunctionParser::start_tag() const {
    return std::nullopt;  // inline format
}

std::optional<std::string> XMLFunctionParser::end_tag() const {
    return std::nullopt;  // inline format
}

std::optional<ToolCall> XMLFunctionParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& tools) const
{
    // Pattern: <function=...>...</function> (allow newlines inside body).
    // ECMAScript '.' does not match '\n', so use [\s\S] for multi-line tool calls
    // as produced by Qwen3.5 chat templates.
    std::regex func_regex("<function=([\\s\\S]*?)</function>", std::regex::ECMAScript);
    std::smatch match;
    if (!std::regex_search(content, match, func_regex)) {
        return std::nullopt;
    }

    std::string func_content = match[0].str();

    // Extract function name (between <function= and first >)
    auto name_start_pos = find_str(func_content, "<function=");
    if (name_start_pos == std::string::npos) return std::nullopt;

    auto name_end_pos = func_content.find('>', name_start_pos + 10);
    if (name_end_pos == std::string::npos) return std::nullopt;

    std::string func_name = func_content.substr(name_start_pos + 10,
                                                 name_end_pos - name_start_pos - 10);

    std::string param_section = func_content.substr(name_end_pos + 1);

    nlohmann::json arguments = nlohmann::json::object();

    // Find all parameter tags
    std::string::size_type search_pos = 0;
    const std::string param_open = "<parameter=";
    const std::string param_close = "</parameter>";

    while (true) {
        auto ps = find_str(param_section, param_open, search_pos);
        if (ps == std::string::npos) break;

        auto ne = param_section.find('>', ps + param_open.size());
        if (ne == std::string::npos) break;

        std::string param_name = param_section.substr(
            ps + param_open.size(), ne - ps - param_open.size());

        auto pe = find_str(param_section, param_close, ne + 1);
        if (pe == std::string::npos) break;

        std::string param_value = param_section.substr(ne + 1, pe - ne - 1);

        // Trim leading/trailing newlines (matching Python behavior)
        if (!param_value.empty() && param_value.front() == '\n') {
            param_value = param_value.substr(1);
        }
        if (!param_value.empty() && param_value.back() == '\n') {
            param_value.pop_back();
        }

        // Convert value based on schema type
        arguments[param_name] = detail::convert_parameter_value(
            param_value, param_name, func_name, tools);

        search_pos = pe + param_close.size();
    }

    return ToolCall(ToolCall::Function(std::move(func_name), std::move(arguments)));
}

// ===========================================================================
// KimiK2ToolCallParser
// ===========================================================================

std::optional<std::string> KimiK2ToolCallParser::start_tag() const {
    return "<|tool_calls_section_begin|>";
}

std::optional<std::string> KimiK2ToolCallParser::end_tag() const {
    return "<|tool_calls_section_end|>";
}

std::optional<ToolCall> KimiK2ToolCallParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& /*tools*/) const
{
    std::string text = content;

    // Strip outer tags
    text = replace_all(text, "<|tool_calls_section_begin|>", "");
    text = replace_all(text, "<|tool_calls_section_end|>", "");

    // Strip inner tags
    text = replace_all(text, "<|tool_call_begin|>", "");
    text = replace_all(text, "<|tool_call_end|>", "");
    text = detail::trim(text);

    // Find <|tool_call_argument_begin|>
    const std::string arg_begin_tag = "<|tool_call_argument_begin|>";
    auto arg_begin_pos = find_str(text, arg_begin_tag);
    if (arg_begin_pos == std::string::npos) return std::nullopt;

    std::string before_arg = detail::trim(text.substr(0, arg_begin_pos));

    // Find the last colon followed by a number
    auto last_colon = before_arg.rfind(':');
    if (last_colon == std::string::npos) return std::nullopt;

    std::string func_name = detail::trim(before_arg.substr(0, last_colon));

    // Strip "functions." prefix if present
    const std::string func_prefix = "functions.";
    if (func_name.substr(0, func_prefix.size()) == func_prefix) {
        func_name = func_name.substr(func_prefix.size());
    } else {
        auto dot_pos = func_name.find('.');
        if (dot_pos != std::string::npos) {
            func_name = func_name.substr(dot_pos + 1);
        }
    }

    if (func_name.empty()) return std::nullopt;

    // Extract arguments JSON (everything after the tag)
    std::string args_str = detail::trim(text.substr(arg_begin_pos + arg_begin_tag.size()));

    auto deserialized = detail::deserialize(args_str);
    if (!deserialized.is_object()) return std::nullopt;

    return ToolCall(ToolCall::Function(std::move(func_name), std::move(deserialized)));
}

// ===========================================================================
// MiniMaxM2ToolCallParser
// ===========================================================================

std::optional<std::string> MiniMaxM2ToolCallParser::start_tag() const {
    return "<minimax:tool_call>";
}

std::optional<std::string> MiniMaxM2ToolCallParser::end_tag() const {
    return "</minimax:tool_call>";
}

std::optional<ToolCall> MiniMaxM2ToolCallParser::parse(
    const std::string& content,
    const std::optional<nlohmann::json>& tools) const
{
    std::string text = content;
    text = replace_all(text, "<minimax:tool_call>", "");
    text = replace_all(text, "</minimax:tool_call>", "");
    text = detail::trim(text);

    // Find <invoke name=...>...</invoke>
    auto invoke_start = find_str(text, "<invoke name=");
    if (invoke_start == std::string::npos) return std::nullopt;

    auto invoke_end = find_str(text, "</invoke>");
    if (invoke_end == std::string::npos) return std::nullopt;

    std::string invoke_content = text.substr(
        invoke_start + 13, invoke_end - invoke_start - 13);  // 13 = length of "<invoke name="

    // Extract function name (between name= and first >)
    auto name_end = invoke_content.find('>');
    if (name_end == std::string::npos) return std::nullopt;

    std::string func_name = detail::extract_name(invoke_content.substr(0, name_end));
    if (func_name.empty()) return std::nullopt;

    // Get parameter config from tools schema
    nlohmann::json param_config = detail::get_parameter_config(func_name, tools);

    nlohmann::json arguments = nlohmann::json::object();
    std::string param_section = invoke_content.substr(name_end + 1);

    // Find all <parameter name=...>...</parameter> tags
    const std::string param_open = "<parameter name=";
    const std::string param_close = "</parameter>";

    std::string::size_type search_pos = 0;
    while (true) {
        auto ps = find_str(param_section, param_open, search_pos);
        if (ps == std::string::npos) break;

        auto ne = param_section.find('>', ps + param_open.size());
        if (ne == std::string::npos) break;

        std::string param_name = detail::extract_name(
            param_section.substr(ps + param_open.size(), ne - ps - param_open.size()));

        auto pe = find_str(param_section, param_close, ne + 1);
        if (pe == std::string::npos) break;

        std::string param_value = param_section.substr(ne + 1, pe - ne - 1);

        // Trim leading/trailing whitespace and newlines (matching Python behavior)
        // Trim whitespace first
        auto ws_start = param_value.find_first_not_of(" \t");
        auto ws_end = param_value.find_last_not_of(" \t");
        if (ws_start != std::string::npos) {
            param_value = param_value.substr(ws_start, ws_end - ws_start + 1);
        } else {
            param_value.clear();
        }
        if (!param_value.empty() && param_value.front() == '\n') {
            param_value = param_value.substr(1);
        }
        if (!param_value.empty() && param_value.back() == '\n') {
            param_value.pop_back();
        }

        // Get types from schema for this parameter
        nlohmann::json param_schema;
        if (param_config.contains(param_name)) {
            param_schema = param_config[param_name];
        }
        auto param_types = detail::extract_types_from_schema(param_schema);
        arguments[param_name] = detail::convert_value_with_types(param_value, param_types);

        search_pos = pe + param_close.size();
    }

    return ToolCall(ToolCall::Function(std::move(func_name), std::move(arguments)));
}

// ===========================================================================
// ToolCallFormat helpers
// ===========================================================================

std::string to_string(ToolCallFormat fmt) {
    switch (fmt) {
    case ToolCallFormat::json:         return "json";
    case ToolCallFormat::lfm2:         return "lfm2";
    case ToolCallFormat::xml_function: return "xml_function";
    case ToolCallFormat::glm4:         return "glm4";
    case ToolCallFormat::gemma:        return "gemma";
    case ToolCallFormat::kimi_k2:      return "kimi_k2";
    case ToolCallFormat::minimax_m2:   return "minimax_m2";
    }
    return "json";
}

std::optional<ToolCallFormat> tool_call_format_from_string(const std::string& s) {
    if (s == "json")           return ToolCallFormat::json;
    if (s == "lfm2")           return ToolCallFormat::lfm2;
    if (s == "xml_function")   return ToolCallFormat::xml_function;
    if (s == "glm4")           return ToolCallFormat::glm4;
    if (s == "gemma")          return ToolCallFormat::gemma;
    if (s == "kimi_k2")        return ToolCallFormat::kimi_k2;
    if (s == "minimax_m2")     return ToolCallFormat::minimax_m2;
    return std::nullopt;
}

std::unique_ptr<ToolCallParser> create_parser(ToolCallFormat fmt) {
    switch (fmt) {
    case ToolCallFormat::json:
        return std::make_unique<JSONToolCallParser>("<tool_call>", "</tool_call>");
    case ToolCallFormat::lfm2:
        return std::make_unique<JSONToolCallParser>("<|tool_call_start|>", "<|tool_call_end|>");
    case ToolCallFormat::xml_function:
        return std::make_unique<XMLFunctionParser>();
    case ToolCallFormat::glm4:
        return std::make_unique<GLM4ToolCallParser>();
    case ToolCallFormat::gemma:
        return std::make_unique<GemmaFunctionParser>();
    case ToolCallFormat::kimi_k2:
        return std::make_unique<KimiK2ToolCallParser>();
    case ToolCallFormat::minimax_m2:
        return std::make_unique<MiniMaxM2ToolCallParser>();
    }
    return std::make_unique<JSONToolCallParser>("<tool_call>", "</tool_call>");
}

std::optional<ToolCallFormat> infer_tool_call_format(const std::string& model_type) {
    std::string lower = detail::to_lower(model_type);
    if (lower == "lfm2" || lower == "lfm2_moe")   return ToolCallFormat::lfm2;
    if (lower == "glm4" || lower == "glm4_moe" || lower == "glm4_moe_lite")
        return ToolCallFormat::glm4;
    if (lower == "gemma" || lower.rfind("gemma", 0) == 0) return ToolCallFormat::gemma;
    // Qwen3 / Qwen3.5 chat templates commonly emit <function=name>…</function>
    // (XML function style), not JSON inside <tool_call> tags.
    if (lower.find("qwen3") != std::string::npos ||
        lower.find("qwen2") != std::string::npos ||
        lower == "qwen") {
        return ToolCallFormat::xml_function;
    }
    return std::nullopt;
}

// ===========================================================================
// ToolCallProcessor
// ===========================================================================

ToolCallProcessor::ToolCallProcessor(
    ToolCallFormat format,
    std::optional<nlohmann::json> tools)
    : parser_(create_parser(format))
    , tools_(std::move(tools))
{}

bool ToolCallProcessor::is_inline_format() const {
    return !parser_->start_tag().has_value() || !parser_->end_tag().has_value();
}

std::optional<char> ToolCallProcessor::start_tag_first_char() const {
    auto tag = parser_->start_tag();
    if (tag.has_value() && !tag->empty()) {
        return tag->front();
    }
    return std::nullopt;
}

std::optional<std::string> ToolCallProcessor::process_chunk(const std::string& chunk) {
    if (is_inline_format()) {
        return process_inline_chunk(chunk);
    }
    return process_tagged_chunk(chunk);
}

std::optional<std::string> ToolCallProcessor::process_inline_chunk(const std::string& chunk) {
    buffer_ += chunk;

    auto tool_call = parser_->parse(buffer_, tools_);
    if (tool_call.has_value()) {
        tool_calls_.push_back(std::move(tool_call.value()));
        buffer_.clear();
        return std::nullopt;
    }

    // Return chunk as-is; caller handles incomplete inline tool calls
    return chunk;
}

std::optional<std::string> ToolCallProcessor::process_tagged_chunk(const std::string& chunk) {
    auto start_tag = parser_->start_tag();
    auto start_char = start_tag_first_char();

    if (!start_tag.has_value() || !start_char.has_value()) {
        return chunk;
    }

    // In normal state, if the chunk doesn't contain the start character, pass through
    if (state_ == State::normal &&
        chunk.find(start_char.value()) == std::string::npos)
    {
        return chunk;
    }

    buffer_ += chunk;
    std::optional<std::string> leading_token;

    switch (state_) {
    case State::normal: {
        // Change state to potential tool call
        state_ = State::potential_tool_call;

        leading_token = separate_token(
            buffer_, std::string(1, start_char.value()), true);

        // fallthrough to potential_tool_call
    }
    [[fallthrough]];
    case State::potential_tool_call: {
        if (partial_match(buffer_, start_tag.value())) {
            if (buffer_.substr(0, start_tag->size()) == start_tag.value()) {
                state_ = State::collecting_tool_call;
                goto collecting;
            } else {
                return std::nullopt;
            }
        } else {
            // Not a match -- return collected text and reset state
            state_ = State::normal;
            std::string result = buffer_;
            buffer_.clear();
            if (leading_token.has_value()) {
                return leading_token.value() + result;
            }
            return result;
        }
    }
    collecting:
    case State::collecting_tool_call: {
        auto end_tag = parser_->end_tag();
        if (!end_tag.has_value()) {
            return std::nullopt;
        }

        auto end_pos = find_str(buffer_, end_tag.value());
        if (end_pos != std::string::npos) {
            // Separate the trailing token
            auto trailing_token = separate_token(buffer_, end_tag.value(), false);

            // Parse the tool call using the parser
            auto tool_call = parser_->parse(buffer_, tools_);
            if (tool_call.has_value()) {
                tool_calls_.push_back(std::move(tool_call.value()));
            }

            state_ = State::normal;
            buffer_.clear();

            // If the trailing token contains the start character, there may be more tool calls
            if (trailing_token.has_value() && start_char.has_value() &&
                trailing_token->find(start_char.value()) != std::string::npos)
            {
                return process_chunk(trailing_token.value());
            } else {
                // Otherwise, return the trailing token or nil if empty
                if (!trailing_token.has_value() || trailing_token->empty()) {
                    return std::nullopt;
                }
                return trailing_token;
            }
        } else {
            return std::nullopt;
        }
    }
    }

    return chunk;  // unreachable, but satisfies compiler
}

std::optional<std::string> ToolCallProcessor::separate_token(
    std::string& buffer, const std::string& separator, bool return_leading)
{
    auto pos = find_str(buffer, separator);
    if (pos == std::string::npos) return std::nullopt;

    std::string token;
    if (return_leading) {
        token = buffer.substr(0, pos);
        buffer = buffer.substr(pos);
    } else {
        token = buffer.substr(pos + separator.size());
        buffer = buffer.substr(0, pos + separator.size());
    }

    return token;
}

bool ToolCallProcessor::partial_match(
    const std::string& buffer, const std::string& tag) const
{
    size_t len = std::min(buffer.size(), tag.size());
    for (size_t i = 0; i < len; ++i) {
        if (buffer[i] != tag[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace mlx_lm
