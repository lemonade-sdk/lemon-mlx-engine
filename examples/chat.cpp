// Example: Interactive chat CLI using mlx-cpp-lm
// Usage: ./chat <model_id_or_path> [options]

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/chat_session.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#include <iostream>
#include <set>
#include <string>

namespace mx = mlx::core;

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024)
        return std::to_string(bytes / (1024ULL * 1024 * 1024)) + "." +
               std::to_string((bytes / (1024ULL * 1024 * 100)) % 10) + " GB";
    if (bytes >= 1024ULL * 1024)
        return std::to_string(bytes / (1024ULL * 1024)) + " MB";
    return std::to_string(bytes) + " B";
}

// Simple CLI argument parser.
struct CliArgs {
    std::string model_path;
    std::string system_prompt;
    int max_tokens = 2048;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 0.0f;
    size_t memory_limit_mb = 0;
    bool no_think = false;
    bool raw_mode = false;  // Skip chat template, use raw encoding
    int kv_bits = 0;        // KV cache quantization bits (0=off, 4 or 8)
    int kv_group_size = 64; // KV cache quantization group size
    int ctx_size = 0;       // Context size for KV cache pre-allocation (0=auto)
};

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_id_or_directory> [options]\n"
                  << "  --system-prompt \"...\"   System instructions\n"
                  << "  --max-tokens N          Max tokens to generate (default: 2048)\n"
                  << "  --temperature T         Sampling temperature (default: 0.7)\n"
                  << "  --top-p P               Nucleus sampling (default: 0.9)\n"
                  << "  --repetition-penalty F  Repetition penalty (default: off)\n"
                  << "  --memory-limit MB       GPU wired memory limit\n"
                  << "  --no-think              Disable thinking/reasoning (Qwen3)\n"
                  << "  --raw                   Skip chat template, raw encoding\n"
                  << "  --kv-bits N             KV cache quantization (0=off, 4 or 8)\n"
                  << "  --kv-group-size N       KV cache quant group size (default: 64)\n"
                  << "  --ctx-size N            Pre-allocate KV cache for N tokens (0=auto)\n";
        std::exit(1);
    }
    args.model_path = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--system-prompt" && i + 1 < argc) {
            args.system_prompt = argv[++i];
        } else if (flag == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (flag == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        } else if (flag == "--top-p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (flag == "--repetition-penalty" && i + 1 < argc) {
            args.repetition_penalty = std::stof(argv[++i]);
        } else if (flag == "--memory-limit" && i + 1 < argc) {
            args.memory_limit_mb = std::stoul(argv[++i]);
        } else if (flag == "--no-think") {
            args.no_think = true;
        } else if (flag == "--raw") {
            args.raw_mode = true;
        } else if (flag == "--kv-bits" && i + 1 < argc) {
            args.kv_bits = std::stoi(argv[++i]);
        } else if (flag == "--kv-group-size" && i + 1 < argc) {
            args.kv_group_size = std::stoi(argv[++i]);
        } else if (flag == "--ctx-size" && i + 1 < argc) {
            args.ctx_size = std::stoi(argv[++i]);
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    if (args.memory_limit_mb > 0) {
        mx::set_wired_limit(args.memory_limit_mb * 1024ULL * 1024);
        std::cerr << "GPU wired memory limit: " << args.memory_limit_mb << " MB" << std::endl;
    }

    try {
        std::cout << "Loading model: " << args.model_path << std::endl;

        auto ctx = mlx_lm::load_llm(args.model_path);

        // Warmup: run a dummy forward pass to prime the GPU allocator cache.
        // Without this, the first real prompt pays ~2s of hipExtMallocWithFlags
        // cold-start overhead. After warmup, allocations hit the buffer cache.
        {
            mlx_lm::GenerateParameters warmup_params;
            warmup_params.max_tokens = 1;
            warmup_params.temperature = 0.0f;
            auto warmup_cache = ctx.new_cache_fn(warmup_params);
            mx::array dummy_tokens = mx::reshape(mx::array({1}), {1, 1});
            mlx_lm::LMInput::Text warmup_text(dummy_tokens);
            auto warmup_out = ctx.call_fn(warmup_text, &warmup_cache, nullptr);
            mx::eval(warmup_out.logits);
        }

        std::cerr << "Model loaded. Memory: active="
                  << format_bytes(mx::get_active_memory())
                  << ", peak=" << format_bytes(mx::get_peak_memory())
                  << ", cache=" << format_bytes(mx::get_cache_memory()) << std::endl;

        bool has_chat_template = static_cast<bool>(ctx.apply_chat_template_fn);

        mlx_lm::GenerateParameters params;
        params.temperature = args.temperature;
        params.top_p = args.top_p;
        params.max_tokens = args.max_tokens;
        if (args.repetition_penalty > 0.0f) {
            params.repetition_penalty = args.repetition_penalty;
        }
        if (args.ctx_size > 0) {
            params.ctx_size = args.ctx_size;
        }
        if (args.kv_bits > 0) {
            params.kv_bits = args.kv_bits;
            params.kv_group_size = args.kv_group_size;
        }

        // Use ChatSession if chat template is available and not in raw mode.
        if (has_chat_template && !args.raw_mode) {
            // Qwen3 (and similar reasoning models) need enable_thinking set
            // explicitly in the template context. Default to true unless --no-think.
            if (ctx.template_extra_context) {
                (*ctx.template_extra_context)["enable_thinking"] = !args.no_think;
            }

            auto container = std::make_shared<mlx_lm::ModelContainer>(std::move(ctx));

            std::optional<std::string> instructions;
            if (!args.system_prompt.empty()) {
                instructions = args.system_prompt;
            }

            mlx_lm::ChatSession session(container, instructions, params);

            std::cout << "Type your message (or 'quit' to exit):" << std::endl;

            while (true) {
                std::cout << "\n> ";
                std::string input;
                std::getline(std::cin, input);

                if (input == "quit" || input == "exit" || std::cin.eof()) break;
                if (input.empty()) continue;

                std::cout << "\nAssistant: ";
                session.stream_response(
                    input,
                    [](const std::string& chunk) -> bool {
                        std::cout << chunk << std::flush;
                        return true;
                    },
                    [](const mlx_lm::GenerateInfo& info) {
                        std::cerr << "\nPrompt:     " << info.prompt_tokens << " tokens, "
                                  << info.prompt_tokens_per_second() << " tokens/s, "
                                  << info.prompt_time_s << "s\n"
                                  << "Generation: " << info.generated_tokens << " tokens, "
                                  << info.tokens_per_second() << " tokens/s, "
                                  << info.generation_time_s << "s" << std::endl;
                    });
                std::cout << std::endl;
            }
        } else {
            // Fallback: raw encoding without chat template.
            if (!has_chat_template) {
                std::cerr << "Warning: No chat template found. Using raw encoding." << std::endl;
            }
            auto container = mlx_lm::ModelContainer(std::move(ctx));

            std::cout << "Type your message (or 'quit' to exit):" << std::endl;

            while (true) {
                std::cout << "\n> ";
                std::string input;
                std::getline(std::cin, input);

                if (input == "quit" || input == "exit" || std::cin.eof()) break;
                if (input.empty()) continue;

                container.perform([&](mlx_lm::ModelContext& ctx) {
                    auto tokens = ctx.encode_fn(input);
                    auto token_array = mx::array(
                        tokens.data(),
                        {static_cast<int>(tokens.size())},
                        mx::int32);
                    mlx_lm::LMInput lm_input(token_array);

                    std::set<int> eos_set;
                    if (ctx.eos_token_ids.has_value()) {
                        for (int id : ctx.eos_token_ids.value()) {
                            eos_set.insert(id);
                        }
                    }

                    std::cout << "\nAssistant: ";
                    auto info = mlx_lm::generate_text(
                        ctx, lm_input, params, eos_set,
                        [](const std::string& text, int /*token*/) {
                            std::cout << text << std::flush;
                            return mlx_lm::GenerateDisposition::more;
                        });
                    std::cout << std::endl;
                    std::cerr << info.summary() << std::endl;
                });
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
