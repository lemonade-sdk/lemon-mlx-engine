// Copyright © 2025 — Ported to C++

#include <mlx-lm/common/chat_template.h>
#include <minja/chat-template.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlx_lm {

// --- Helper: extract a special token string from tokenizer_config.json ---
// HF configs store tokens as either a plain string or an object with a "content" field.
static std::string extract_token(const nlohmann::json& config, const std::string& key) {
    if (!config.contains(key) || config[key].is_null()) return "";
    const auto& val = config[key];
    if (val.is_string()) return val.get<std::string>();
    if (val.is_object() && val.contains("content")) return val["content"].get<std::string>();
    return "";
}

// --- Impl (pImpl idiom to hide minja headers from public API) ---

struct ChatTemplate::Impl {
    minja::chat_template tmpl;
    std::string template_str;
    std::string eos_token;
    std::string bos_token;

    Impl(const std::string& source, const std::string& bos, const std::string& eos)
        : tmpl(source, bos, eos), template_str(source), eos_token(eos), bos_token(bos) {}
};

// --- ChatTemplate ---

ChatTemplate::ChatTemplate(const std::string& template_str,
                           const nlohmann::json& tokenizer_config) {
    auto bos = extract_token(tokenizer_config, "bos_token");
    auto eos = extract_token(tokenizer_config, "eos_token");
    impl_ = std::make_unique<Impl>(template_str, bos, eos);
}

ChatTemplate::~ChatTemplate() = default;
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;

std::string ChatTemplate::apply(
    const std::vector<Message>& messages,
    bool add_generation_prompt,
    const nlohmann::json& extra_context) const
{
    // Convert Message maps to nlohmann::ordered_json array.
    nlohmann::ordered_json json_messages = nlohmann::ordered_json::array();
    for (const auto& msg : messages) {
        nlohmann::ordered_json j;
        for (const auto& [key, value] : msg) {
            j[key] = value;
        }
        json_messages.push_back(std::move(j));
    }

    // Build inputs for minja.
    minja::chat_template_inputs inputs;
    inputs.messages = std::move(json_messages);
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.now = std::chrono::system_clock::now();

    if (!extra_context.is_null() && !extra_context.empty()) {
        inputs.extra_context = extra_context;
    }

    minja::chat_template_options opts;
    // Enable all polyfills by default for broad model compatibility.
    return impl_->tmpl.apply(inputs, opts);
}

const std::string& ChatTemplate::template_string() const { return impl_->template_str; }
const std::string& ChatTemplate::eos_token() const { return impl_->eos_token; }
const std::string& ChatTemplate::bos_token() const { return impl_->bos_token; }

// --- load_chat_template ---

std::optional<ChatTemplate> load_chat_template(const std::string& model_directory) {
    auto config_path = fs::path(model_directory) / "tokenizer_config.json";

    // Try tokenizer_config.json first.
    if (fs::exists(config_path)) {
        std::ifstream f(config_path);
        if (!f) return std::nullopt;

        nlohmann::json config;
        f >> config;

        if (config.contains("chat_template")) {
            const auto& ct = config["chat_template"];
            std::string template_str;

            if (ct.is_string()) {
                template_str = ct.get<std::string>();
            } else if (ct.is_array()) {
                // Array of {name, template} objects — pick "default" or first.
                for (const auto& entry : ct) {
                    if (entry.contains("name") && entry["name"] == "default") {
                        template_str = entry["template"].get<std::string>();
                        break;
                    }
                }
                if (template_str.empty() && !ct.empty()) {
                    template_str = ct[0]["template"].get<std::string>();
                }
            }

            if (!template_str.empty()) {
                // Patch Jinja2 pipe filters for minja compatibility.
                // Remove unsupported filters like | capitalize, | trim (cosmetic only).
                for (auto& filter : {"| capitalize", "| trim", "| upper", "| lower", "| title"}) {
                    std::string f(filter);
                    for (auto pos = template_str.find(f); pos != std::string::npos; pos = template_str.find(f, pos)) {
                        template_str.erase(pos, f.size());
                    }
                }
                return ChatTemplate(template_str, config);
            }
        }
    }

    // Fallback: try chat_template.jinja file.
    auto jinja_path = fs::path(model_directory) / "chat_template.jinja";
    if (fs::exists(jinja_path)) {
        std::ifstream f(jinja_path);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            auto template_str = ss.str();
            if (!template_str.empty()) {
                // Patch Jinja2 pipe filters for minja compatibility.
                for (auto& filter : {"| capitalize", "| trim", "| upper", "| lower", "| title"}) {
                    std::string f(filter);
                    for (auto pos = template_str.find(f); pos != std::string::npos; pos = template_str.find(f, pos))
                        template_str.erase(pos, f.size());
                }
                // Load tokenizer_config for special tokens even without chat_template field.
                nlohmann::json config;
                if (fs::exists(config_path)) {
                    std::ifstream cf(config_path);
                    if (cf) cf >> config;
                }
                return ChatTemplate(template_str, config);
            }
        }
    }

    return std::nullopt;
}

} // namespace mlx_lm
