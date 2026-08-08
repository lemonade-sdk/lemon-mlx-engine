// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2: dlopen Redline C-API init smoke (no forward replacement).

#include <mlx-lm/common/redline_decode_session.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

#if defined(MLX_BUILD_ROCM)
#include <dlfcn.h>
#endif

namespace mlx_lm {

namespace {

bool env_exact_one(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1' && v[1] == '\0';
}

std::mutex g_mu;
bool g_inited = false;
RedlineSessionState g_state = RedlineSessionState::Disabled;
std::string g_err;
#if defined(MLX_BUILD_ROCM)
void* g_lib = nullptr;
#endif

#if defined(MLX_BUILD_ROCM)
// Minimal C-API surface (from redline_dispatch.h / redline-capi).
using rl_abi_version_fn = uint32_t (*)();
using rl_gpu_new_fn = void* (*)(int32_t);
using rl_gpu_free_fn = void (*)(void*);

const char* default_lib_candidates[] = {
    "libredline_dispatch.so",
    // Common research layout (E0):
    "/tmp/redline-warpfront-target/release/libredline_dispatch.so",
    nullptr,
};

void* try_dlopen() {
    if (const char* custom = std::getenv("MLX_REDLINE_LIB")) {
        if (custom[0] != '\0') {
            void* h = ::dlopen(custom, RTLD_NOW | RTLD_LOCAL);
            if (h) {
                return h;
            }
            g_err = std::string("dlopen MLX_REDLINE_LIB failed: ") +
                    (dlerror() ? dlerror() : "unknown");
            return nullptr;
        }
    }
    for (const char** p = default_lib_candidates; *p; ++p) {
        // Clear stale errors.
        (void)dlerror();
        void* h = ::dlopen(*p, RTLD_NOW | RTLD_LOCAL);
        if (h) {
            return h;
        }
    }
    const char* e = dlerror();
    g_err = std::string("dlopen libredline_dispatch.so failed: ") +
            (e ? e : "not found (set MLX_REDLINE_LIB)");
    return nullptr;
}

bool init_smoke(void* lib) {
    auto* abi = reinterpret_cast<rl_abi_version_fn>(::dlsym(lib, "rl_abi_version"));
    if (!abi) {
        g_err = "dlsym rl_abi_version failed";
        return false;
    }
    uint32_t ver = abi();
    auto* gpu_new = reinterpret_cast<rl_gpu_new_fn>(::dlsym(lib, "rl_gpu_new"));
    auto* gpu_free = reinterpret_cast<rl_gpu_free_fn>(::dlsym(lib, "rl_gpu_free"));
    if (!gpu_new || !gpu_free) {
        // abi alone is enough for P2 symbol smoke; gpu binds optional.
        g_err = "abi=" + std::to_string(ver) + " gpu_syms=missing";
        return true;
    }
    void* gpu = gpu_new(0);
    if (!gpu) {
        // Observed: rl_gpu_new may return null if HIP/MLX already owns the
        // device (post-load). Standalone C smoke still succeeds pre-MLX.
        // P2 gate = dlopen + abi; gpu bind is best-effort for later P3.
        g_err = "abi=" + std::to_string(ver) +
                " gpu_new=null (try early init before MLX load; fallback ok)";
        return true;
    }
    gpu_free(gpu);
    g_err = "abi=" + std::to_string(ver) + " gpu_new=ok";
    return true;
}
#endif // MLX_BUILD_ROCM

} // namespace

RedlineSessionState redline_session_ensure_init() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_inited) {
        return g_state;
    }
    g_inited = true;

#if !defined(MLX_BUILD_ROCM)
    g_state = RedlineSessionState::Disabled;
    return g_state;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE")) {
        g_state = RedlineSessionState::Disabled;
        return g_state;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        g_state = RedlineSessionState::XorEager;
        g_err = "XOR with MLX_DECODE_GRAPH_PURE=1";
        return g_state;
    }

    g_lib = try_dlopen();
    if (!g_lib) {
        g_state = RedlineSessionState::Failed;
        return g_state;
    }
    if (!init_smoke(g_lib)) {
        ::dlclose(g_lib);
        g_lib = nullptr;
        g_state = RedlineSessionState::Failed;
        return g_state;
    }
    g_state = RedlineSessionState::Ready;
    return g_state;
#endif
}

void maybe_log_redline_session_status() {
#if !defined(MLX_BUILD_ROCM)
    return;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE")) {
        return;
    }
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    auto st = redline_session_ensure_init();
    switch (st) {
        case RedlineSessionState::XorEager:
            std::cerr
                << "[redline] MLX_REDLINE_DECODE=1 and MLX_DECODE_GRAPH_PURE=1: "
                   "fail-closed to eager (XOR until measured; no Redline session)\n";
            break;
        case RedlineSessionState::Ready:
            std::cerr
                << "[redline] session READY (P2 init-only; forward still product; "
                << redline_session_last_error() << ")\n";
            break;
        case RedlineSessionState::Failed:
            std::cerr
                << "[redline] session FAILED → fallback product HIP"
                << (redline_session_last_error().empty()
                        ? ""
                        : (std::string(" (") + redline_session_last_error() + ")"))
                << "\n";
            break;
        case RedlineSessionState::Disabled:
        default:
            // Should not log if env off; if we got here, treat as no-op.
            break;
    }
#endif
}

const std::string& redline_session_last_error() {
    return g_err;
}

} // namespace mlx_lm
