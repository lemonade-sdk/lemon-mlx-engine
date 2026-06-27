// Example: Interactive chat CLI using mlx-cpp-lm
// Usage: ./chat <model_id_or_path> [options]

#include <mlx-lm/llm/llm_factory.h>
#include <mlx-lm/common/chat_session.h>
#include <mlx-lm/common/registry.h>
#include <mlx-lm/common/generate.h>
#include <mlx-lm/common/model_container.h>
#include <mlx/mlx.h>
#if defined(MLX_BUILD_ROCM)
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <vector>
#endif
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

namespace mx = mlx::core;


#if defined(MLX_BUILD_ROCM)
namespace {
namespace fs = std::filesystem;

static bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

static void add_unique_candidate(std::vector<fs::path>& candidates,
                                 const fs::path& candidate) {
    if (candidate.empty()) {
        return;
    }
    for (const auto& existing : candidates) {
        if (existing == candidate) {
            return;
        }
    }
    candidates.push_back(candidate);
}

static bool has_tensile_library_files(const fs::path& directory) {
    std::error_code ec;
    if (!fs::is_directory(directory, ec)) {
        return false;
    }

    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (ec) {
            return false;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (starts_with(filename, "TensileLibrary_lazy_") &&
            entry.path().extension() == ".dat") {
            return true;
        }
    }
    return false;
}

static fs::path loaded_library_directory(const char* library_name,
                                         const char* symbol_name) {
    void* symbol = nullptr;
#ifdef RTLD_NOLOAD
    void* handle = dlopen(library_name, RTLD_LAZY | RTLD_NOLOAD);
    if (handle != nullptr) {
        symbol = dlsym(handle, symbol_name);
        dlclose(handle);
    }
#endif
    if (symbol == nullptr) {
        symbol = dlsym(RTLD_DEFAULT, symbol_name);
    }

    Dl_info info{};
    if (symbol != nullptr && dladdr(symbol, &info) != 0 &&
        info.dli_fname != nullptr) {
        return fs::path(info.dli_fname).parent_path();
    }
    return {};
}

static void add_rocm_opt_candidates(std::vector<fs::path>& candidates,
                                    const std::string& component) {
    std::error_code ec;
    const fs::path opt_dir("/opt");
    if (!fs::is_directory(opt_dir, ec)) {
        return;
    }

    for (const auto& entry : fs::directory_iterator(opt_dir, ec)) {
        if (ec) {
            return;
        }
        if (!entry.is_directory(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (starts_with(name, "rocm")) {
            add_unique_candidate(candidates,
                                 entry.path() / "lib" / component / "library");
        }
    }
}

static void add_therock_venv_candidates(std::vector<fs::path>& candidates,
                                        const std::string& component) {
    std::error_code ec;
    const fs::path lib_dir("/tmp/rocm_venv/lib");
    if (!fs::is_directory(lib_dir, ec)) {
        return;
    }

    for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
        if (ec) {
            return;
        }
        if (!entry.is_directory(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (starts_with(name, "python")) {
            add_unique_candidate(
                candidates,
                entry.path() / "site-packages" / "_rocm_sdk_libraries" / "lib" /
                    component / "library");
        }
    }
}

static std::string path_with_trailing_slash(const fs::path& path) {
    std::string value = path.string();
    if (!value.empty() && value.back() != '/') {
        value.push_back('/');
    }
    return value;
}

static void auto_configure_tensile_path(const char* env_var,
                                        const char* library_name,
                                        const char* symbol_name,
                                        const std::string& component) {
    if (std::getenv(env_var) != nullptr) {
        return;
    }

    std::vector<fs::path> candidates;
    const fs::path loaded_dir = loaded_library_directory(library_name, symbol_name);
    if (!loaded_dir.empty()) {
        add_unique_candidate(candidates, loaded_dir / component / "library");
    }
    add_unique_candidate(candidates, fs::path("/opt/rocm/lib") / component / "library");
    add_unique_candidate(candidates,
                         fs::path("/opt/rocm-7.2.4/lib") / component / "library");
    if (const char* rocm_dir = std::getenv("ROCm_DIR")) {
        if (rocm_dir[0] != '\0') {
            add_unique_candidate(candidates,
                                 fs::path(rocm_dir) / "lib" / component / "library");
        }
    }
    add_rocm_opt_candidates(candidates, component);
    add_therock_venv_candidates(candidates, component);

    for (const auto& candidate : candidates) {
        if (has_tensile_library_files(candidate)) {
            const std::string path = path_with_trailing_slash(candidate);
            if (setenv(env_var, path.c_str(), 0) == 0) {
                std::cerr << "[rocm-tensile] Set " << env_var << "=" << path
                          << std::endl;
            }
            return;
        }
    }
}
} // namespace

static void auto_configure_rocm_tensile_paths() {
    auto_configure_tensile_path("ROCBLAS_TENSILE_LIBPATH", "librocblas.so",
                                "rocblas_create_handle", "rocblas");
    auto_configure_tensile_path("HIPBLASLT_TENSILE_LIBPATH", "libhipblaslt.so",
                                "hipblasLtCreate", "hipblaslt");
}
#else
static void auto_configure_rocm_tensile_paths() {}
#endif

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
    std::string register_arch;  // Path to architecture registration JSON file
    bool ignore_eos = false;  // Benchmark: keep generating to --max-tokens (ignore EOS)
    bool auto_quantize = false;  // Auto-quantize unquantized bf16/fp16 models to 4-bit
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
                  << "  --register-arch FILE   Register custom architecture from JSON file\n"
                  << "  --auto-quantize        Auto-quantize unquantized bf16/fp16 models to 4-bit at load time\n"
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
        } else if (flag == "--auto-quantize") {
            args.auto_quantize = true;
        } else if (flag == "--list-devices") {
            args.list_devices = true;
        } else if (flag == "--ignore-eos") {
            args.ignore_eos = true;
        } else if (flag == "--register-arch" && i + 1 < argc) {
            args.register_arch = argv[++i];
        }
    }
    return args;
}

int main(int argc, char* argv[]) {
    // Unbuffered stdout so generated tokens appear live (not flushed in a block
    // when piped to a file/pipe).
    setvbuf(stdout, nullptr, _IONBF, 0);

    // Configure ROCm Tensile paths before anything touches HIP/MLX.
    auto_configure_rocm_tensile_paths();

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
        // Load custom architecture registrations if specified
        if (!args.register_arch.empty()) {
            std::cerr << "Loading architecture registrations: " << args.register_arch << std::endl;
            mlx_lm::ArchitectureRegistry::instance().load_from_file(args.register_arch);
        }

        std::cout << "Loading model: " << args.model_path << std::endl;

        auto ctx = mlx_lm::load_llm(args.model_path, "", args.auto_quantize);

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
