// Example: Interactive chat CLI using mlx-cpp-lm
// Usage: ./chat <model_id_or_path> [options]

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/chat_session.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#if defined(MLX_BUILD_ROCM)
#include <hip/hip_runtime.h>
#endif
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

namespace mx = mlx::core;

// GPU selection / enumeration. Selecting a device sets HIP_VISIBLE_DEVICES
// before any HIP/MLX call so the chosen GPU becomes device 0 (which the MLX
// ROCm backend uses); the backend's is_integrated() then auto-detects whether
// it's the integrated APU (unified memory) or a discrete GPU (VRAM + host
// staging) and routes the allocator accordingly. Works on any system with one
// or more GPUs. Listing shells out to rocm-smi to avoid a HIP header dependency
// in this host-only example.
static void select_or_list_gpu(int argc, char* argv[]) {
    bool list = false;
    int device = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list-devices")
            list = true;
        else if (a == "--device" && i + 1 < argc)
            device = std::atoi(argv[i + 1]);
    }
    if (list) {
        std::cout << "Available GPUs (HIP device index = order shown by "
                     "rocm-smi):\n";
        int rc = std::system(
            "rocm-smi --showproductname 2>/dev/null | grep -iE 'GPU\\[|Card "
            "Series|GFX Version' || rocminfo 2>/dev/null | grep -iE 'Marketing "
            "Name|gfx1|gfx9'");
        (void)rc;
        std::cout << "\nSelect with:  --device N   (N = HIP device index; "
                     "default is 0)\n";
        std::exit(0);
    }
    if (device >= 0) {
        // Bind the chosen GPU via the MLX device index (hipSetDevice under the
        // hood) instead of masking with HIP_VISIBLE_DEVICES — the env approach
        // is timing-fragile (MLX can grab device 0 before the mask applies).
        mx::set_default_device(mx::Device(mx::Device::gpu, device));
        std::cerr << "Selecting GPU device index " << device << "\n";
    }
}

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
    size_t cache_limit_mb = 0;
    bool no_think = false;
    bool raw_mode = false;  // Skip chat template, use raw encoding
    int kv_bits = 0;        // KV cache quantization bits (0=off, 4 or 8)
    int kv_group_size = 64; // KV cache quantization group size
    int ctx_size = 0;       // Context size for KV cache pre-allocation (0=auto)
    bool use_mtp = false;
    int n_draft_tokens = 1;
    int device = -1;          // GPU index to use (-1 = auto / default device 0)
    bool list_devices = false;
    bool ignore_eos = false;  // Benchmark: keep generating to --max-tokens (ignore EOS)
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
                  << "  --ctx-size N            Pre-allocate KV cache for N tokens (0=auto)\n"
                  << "  --use-mtp               Enable MTP speculative decode (scaffolding)\n"
                  << "  --n-draft N             MTP draft tokens per step (default: 1)\n"
                  << "  --device N              GPU index to run on (default: auto)\n"
                  << "  --list-devices          List available GPUs and exit\n";
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
        } else if (flag == "--cache-limit" && i + 1 < argc) {
            args.cache_limit_mb = std::stoul(argv[++i]);
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
        } else if (flag == "--use-mtp") {
            args.use_mtp = true;
        } else if (flag == "--n-draft" && i + 1 < argc) {
            args.n_draft_tokens = std::stoi(argv[++i]);
        } else if (flag == "--device" && i + 1 < argc) {
            args.device = std::stoi(argv[++i]);
        } else if (flag == "--list-devices") {
            args.list_devices = true;
        } else if (flag == "--ignore-eos") {
            args.ignore_eos = true;
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    // Unbuffered stdout so generated tokens appear live (not flushed in a block
    // when piped to a file/pipe).
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Handle --list-devices / --device before anything touches HIP/MLX.
    select_or_list_gpu(argc, argv);

    auto args = parse_args(argc, argv);

    if (args.memory_limit_mb > 0) {
        mx::set_wired_limit(args.memory_limit_mb * 1024ULL * 1024);
        std::cerr << "GPU wired memory limit: " << args.memory_limit_mb << " MB" << std::endl;
    }
    if (args.cache_limit_mb > 0) {
        // Cap the buffer-reuse pool (max_pool_size_). Keeps the cache from
        // ballooning during load on a large dedicated GPU.
        mx::set_cache_limit(args.cache_limit_mb * 1024ULL * 1024);
        std::cerr << "GPU cache limit: " << args.cache_limit_mb << " MB" << std::endl;
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

        // Bound the buffer cache so it can't balloon and fill VRAM, while KEEPING
        // a pool for fast reuse (clearing it outright would force cold allocations
        // during generation). Auto-fit: keep the resident model plus a working
        // reserve for KV/activation growth, and cap the cache at the remainder so
        // it always leaves headroom. The HIP-graph decode arena is separately a
        // fixed size and is unaffected.
        {
            // The model loads once and the KV cache aligns statically once; almost
            // nothing is allocated per token after that. So the buffer-REUSE pool
            // (the "cache" — NOT the KV cache) only needs to cover one forward's
            // transient scratch, not gigabytes. A large pool just crowds VRAM and
            // pushes peak usage to the physical ceiling, where the driver's TTM
            // eviction fires mid-forward and wedges the queue on a discrete GPU.
            // Keep the pool small and leave the rest of VRAM free for KV cache +
            // prefill activations + driver headroom.
            size_t budget = mx::get_memory_limit();
            size_t active = mx::get_active_memory();
            size_t free_after_model = (budget > active) ? (budget - active) : 0;
            // Unified-memory APUs (system-RAM pool) want a large pool: a small cap
            // forces constant eviction, and each eviction is a blocking hipFree that
            // can deadlock under async load. Discrete GPUs want a small cap so the
            // pool can't balloon past VRAM and spill across the link. The MLX memory
            // budget is the APU singleton's value even on the dGPU, so probe the
            // actual running device.
            size_t cache_cap = free_after_model / 4;
#if defined(MLX_BUILD_ROCM)
            int curdev = 0;
            hipGetDevice(&curdev);
            hipDeviceProp_t prop{};
            hipGetDeviceProperties(&prop, curdev);
            if (!prop.integrated) {
                size_t fb = 0, tb = 0;
                hipMemGetInfo(&fb, &tb);
                cache_cap = std::min<size_t>(static_cast<size_t>(2) << 30, fb / 4);
            }
#endif
            mx::set_cache_limit(cache_cap);
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
        params.use_mtp = args.use_mtp;
        params.n_draft_tokens = args.n_draft_tokens;
        if (const char* e = std::getenv("MLX_PREFILL_STEP"))
            params.prefill_step_size = std::atoi(e);
        if (args.use_mtp) {
            std::cerr << "MTP enabled (scaffolding): n_draft="
                      << args.n_draft_tokens << "\n";
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
                    if (ctx.eos_token_ids.has_value() && !args.ignore_eos) {
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
