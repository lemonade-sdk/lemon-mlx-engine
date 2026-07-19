// OpenAI-compatible inference server with auto-load support.
// Usage:
//   ./server [model_id_or_path] [options]
//   ./server --auto [options]           # Start without loading, auto-load on request

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/kv_cache.h>
#include <mlx-lm/common/model_container.h>
#include <mlx-lm/common/model_manager.h>
#include <mlx-lm/common/server.h>
#include <mlx/mlx.h>
#include <csignal>
#include <iostream>
#include <string>

namespace mx = mlx::core;

static mlx_lm::Server* g_server = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_server) g_server->stop();
}

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024ULL * 1024 * 1024)
        return std::to_string(bytes / (1024ULL * 1024 * 1024)) + "." +
               std::to_string((bytes / (1024ULL * 1024 * 100)) % 10) + " GB";
    if (bytes >= 1024ULL * 1024)
        return std::to_string(bytes / (1024ULL * 1024)) + " MB";
    return std::to_string(bytes) + " B";
}

struct CliArgs {
    std::string model_path;         // empty = auto-load mode
    std::string host = "127.0.0.1";
    int port = 8080;
    // 4096: thinking-on CoT often exhausts 2048 before a final answer.
    int max_tokens = 4096;
    float temperature = 0.6f;
    float top_p = 1.0f;
    float repetition_penalty = 0.0f;
    size_t memory_limit_mb = 0;
    bool no_think = false;
    int kv_bits = 0;
    int kv_group_size = 64;
    int ctx_size = 0;
    bool no_download = false;
    int max_loaded = 1;
    bool use_mtp = false;
    int n_draft_tokens = 3;
};

static CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;

    for (int i = 1; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (flag == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
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
        } else if (flag == "--no-download") {
            args.no_download = true;
        } else if (flag == "--max-loaded" && i + 1 < argc) {
            args.max_loaded = std::stoi(argv[++i]);
        } else if (flag == "--kv-bits" && i + 1 < argc) {
            args.kv_bits = std::stoi(argv[++i]);
        } else if (flag == "--kv-group-size" && i + 1 < argc) {
            args.kv_group_size = std::stoi(argv[++i]);
        } else if (flag == "--ctx-size" && i + 1 < argc) {
            args.ctx_size = std::stoi(argv[++i]);
        } else if (flag == "--use-mtp") {
            args.use_mtp = true;
        } else if (flag == "--n-draft-tokens" && i + 1 < argc) {
            args.n_draft_tokens = std::stoi(argv[++i]);
        } else if (flag == "-h" || flag == "--help") {
            std::cerr << "Usage: " << argv[0] << " [model_id_or_directory] [options]\n"
                      << "\n"
                      << "If no model is specified, starts in auto-load mode.\n"
                      << "Models are loaded on demand when API requests arrive.\n"
                      << "\n"
                      << "Options:\n"
                      << "  --host HOST             Bind address (default: 127.0.0.1)\n"
                      << "  --port PORT             Listen port (default: 8080)\n"
                      << "  --max-tokens N          Default max tokens (default: 4096)\n"
                      << "  --temperature T         Default temperature (default: 0.6)\n"
                      << "  --top-p P               Default top-p (default: 1.0)\n"
                      << "  --repetition-penalty F  Default repetition penalty (off)\n"
                      << "  --memory-limit MB       GPU wired memory limit\n"
                      << "  --no-think              Disable thinking/reasoning\n"
                      << "  --no-download           Don't auto-download models from HF Hub\n"
                      << "  --max-loaded N          Max models in memory (default: 1, LRU eviction)\n"
                      << "  --kv-bits N             KV cache quantization (0=off, 4 or 8)\n"
                      << "  --kv-group-size N       KV cache quant group size (default: 64)\n"
                      << "  --ctx-size N            Pre-allocate KV cache (0=auto)\n"
                      << "  --use-mtp               Enable MTP speculative decoding (model must have mtp.* weights)\n"
                      << "  --n-draft-tokens N      MTP draft tokens per step (default: 3)\n"
                      << "\n"
                      << "Endpoints:\n"
                      << "  GET  /health              Health check\n"
                      << "  GET  /v1/models           List available MLX models\n"
                      << "  POST /v1/chat/completions Chat completion (auto-loads model)\n"
                      << "  POST /v1/completions      Text completion (auto-loads model)\n"
                      << "  POST /load                Explicitly load a model\n"
                      << "  POST /unload              Unload a model\n"
                      << "\n"
                      << "Environment variables:\n"
                      << "  HF_HUB_CACHE   HuggingFace cache directory (highest priority)\n"
                      << "  HF_HOME        HuggingFace home ($HF_HOME/hub used as cache)\n"
                      << "  HF_TOKEN       HuggingFace API token for private models\n";
            std::exit(0);
        } else if (flag[0] != '-' && args.model_path.empty()) {
            args.model_path = flag;
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    // Explicitly select the device to avoid ROCm fallback on non-GPU systems.
    // MLX_BUILD_ROCM is only defined when the build targets ROCm (ubuntu-rocm).
#if defined(MLX_BUILD_ROCM) && MLX_BUILD_ROCM
    // ROCm backend — use default (GPU).
    std::cerr << "Device: ROCm GPU\n";
#else
    // No ROCm backend — force CPU to prevent hip_kernel errors.
    std::cerr << "Device: CPU (ROCm disabled)\n";
    mx::set_default_device(mx::Device::cpu);
#endif

    if (args.memory_limit_mb > 0) {
        mx::set_wired_limit(args.memory_limit_mb * 1024ULL * 1024);
        std::cerr << "GPU wired memory limit: " << args.memory_limit_mb << " MB\n";
    }

    try {
        // Create model manager.
        auto manager = std::make_shared<mlx_lm::ModelManager>();
        manager->set_no_download(args.no_download);
        manager->set_no_think(args.no_think);
        manager->set_max_loaded(args.max_loaded);

        // Build default params.
        mlx_lm::GenerateParameters default_params;
        default_params.temperature = args.temperature;
        default_params.top_p = args.top_p;
        default_params.max_tokens = args.max_tokens;
        if (args.repetition_penalty > 0.0f) {
            default_params.repetition_penalty = args.repetition_penalty;
        }
        if (args.kv_bits > 0) {
            default_params.kv_bits = args.kv_bits;
            default_params.kv_group_size = args.kv_group_size;
        }
        if (args.ctx_size > 0) {
            default_params.ctx_size = args.ctx_size;
        }
        default_params.use_mtp = args.use_mtp;
        default_params.n_draft_tokens = args.n_draft_tokens;
        manager->set_default_params(default_params);

        // If a model was specified, pre-load it.
        if (!args.model_path.empty()) {
            std::cerr << "Loading model: " << args.model_path << "\n";

            auto ctx = mlx_lm::load_llm(args.model_path);

            // Warmup: prime GPU allocator cache.
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
                      << ", cache=" << format_bytes(mx::get_cache_memory()) << "\n";

            // Always set explicitly (same as chat.cpp / ModelManager).
            if (ctx.template_extra_context) {
                (*ctx.template_extra_context)["enable_thinking"] = !args.no_think;
            }

            auto container = std::make_shared<mlx_lm::ModelContainer>(std::move(ctx));
            manager->add_loaded(args.model_path, std::move(container));
        } else {
            std::cerr << "Starting in auto-load mode (no model pre-loaded).\n";
            std::cerr << "Models will be loaded on demand from API requests.\n";

            // Show available cached models.
            auto available = manager->list_available();
            if (!available.empty()) {
                std::cerr << "\nAvailable MLX models in HF cache:\n";
                for (const auto& m : available) {
                    std::cerr << "  " << m.model_id;
                    if (!m.model_type.empty()) std::cerr << " (" << m.model_type << ")";
                    std::cerr << "\n";
                }
                std::cerr << "\n";
            }
        }

        // Build server config.
        mlx_lm::ServerConfig config;
        config.host = args.host;
        config.port = args.port;
        config.default_params = default_params;

        mlx_lm::Server server(manager, config);
        g_server = &server;

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        std::cerr << "Endpoints:\n"
                  << "  GET  /health\n"
                  << "  GET  /v1/models\n"
                  << "  POST /v1/chat/completions\n"
                  << "  POST /v1/completions\n"
                  << "  POST /load\n"
                  << "  POST /unload\n";

        server.start();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
