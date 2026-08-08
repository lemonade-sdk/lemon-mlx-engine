// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2: dlopen Redline C-API init smoke (no forward replacement).
// P5/P6: optional in-process PM4 micro-op when MLX_REDLINE_HSACO is set;
//     P6 bakes graph_decode_pos device ptr as accumulator. NOT gen t/s.
// P7: optional L=1 sidecar (MLX_REDLINE_SIDECAR=1) — retained IB patch+replay
//     alongside product call_fn; does not replace forward; NOT gen t/s.

#include <mlx-lm/common/graph_decode.h>
#include <mlx-lm/common/redline_decode_session.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#if defined(MLX_BUILD_ROCM)
#include <dlfcn.h>
// HIP host headers require an explicit platform macro (see server CMake).
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
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
using rl_gpu_load_module_fn =
    int32_t (*)(const void* gpu, const uint8_t* code, size_t len, void** out);
using rl_module_free_fn = void (*)(void* module);
using rl_pm4_builder_new_fn = void* (*)(const void* gpu);
using rl_pm4_builder_free_fn = void (*)(void* builder);
using rl_pm4_dispatch_fn = int32_t (*)(
    void* builder,
    const void* module,
    const char* symbol,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint32_t block_x,
    uint32_t block_y,
    uint32_t block_z,
    uint32_t dynamic_group_bytes,
    const uint8_t* kernarg,
    size_t kernarg_len);
using rl_pm4_finalize_fn =
    int32_t (*)(const void* gpu, void* builder, void** out_ib);
using rl_pm4_replay_fn = int32_t (*)(void* ib);
using rl_pm4_ib_set_kernargs_fn = int32_t (*)(
    void* ib,
    size_t dispatch_index,
    size_t byte_offset,
    const uint8_t* kernarg,
    size_t len);
using rl_pm4_ib_free_fn = void (*)(void* ib);

// P7 retained sidecar (process lifetime when MLX_REDLINE_SIDECAR=1 + micro PASS).
void* g_gpu = nullptr;
void* g_mod = nullptr;
void* g_ib = nullptr;
unsigned int* g_side_acc = nullptr;
rl_pm4_replay_fn g_fn_replay = nullptr;
rl_pm4_ib_set_kernargs_fn g_fn_set_k = nullptr;
rl_gpu_free_fn g_fn_gpu_free = nullptr;
rl_module_free_fn g_fn_mod_free = nullptr;
rl_pm4_ib_free_fn g_fn_ib_free = nullptr;
bool g_sidecar_ready = false;
uint32_t g_sidecar_n = 0;
uint64_t g_sidecar_expected = 0;
bool g_sidecar_first_logged = false;

constexpr int32_t kRlOk = 0;

const char* default_lib_candidates[] = {
    "libredline_dispatch.so",
    // Common research layout (E0):
    "/tmp/redline-warpfront-target/release/libredline_dispatch.so",
    nullptr,
};

// RTLD_GLOBAL: HSA may already be mapped by MLX/HIP; LOCAL can break
// redline-rocr's libloading. Optional DEEPBIND via MLX_REDLINE_DEEPBIND=1.
int dlopen_flags() {
    int flags = RTLD_NOW | RTLD_GLOBAL;
#ifdef RTLD_DEEPBIND
    if (env_exact_one("MLX_REDLINE_DEEPBIND")) {
        flags |= RTLD_DEEPBIND;
    }
#endif
    return flags;
}

void* try_dlopen() {
    const int flags = dlopen_flags();
    if (const char* custom = std::getenv("MLX_REDLINE_LIB")) {
        if (custom[0] != '\0') {
            void* h = ::dlopen(custom, flags);
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
        void* h = ::dlopen(*p, flags);
        if (h) {
            return h;
        }
    }
    const char* e = dlerror();
    g_err = std::string("dlopen libredline_dispatch.so failed: ") +
            (e ? e : "not found (set MLX_REDLINE_LIB)");
    return nullptr;
}

// Optional P5/P6: engine-side C-API micro-op (decode_kernargs pattern).
// Gated by MLX_REDLINE_HSACO path; never product forward; NOT gen t/s.
// P6: accumulator = graph_decode_pos device buffer (stable product ptr bake).
// Appends " gd_bind=... micro=..." to g_err.
void try_micro_op(void* lib, void* gpu) {
    const char* hsaco = std::getenv("MLX_REDLINE_HSACO");
    if (!hsaco || hsaco[0] == '\0') {
        g_err += " micro=skip";
        return;
    }
    if (!gpu) {
        g_err += " micro=skip_gpu_null";
        return;
    }

    auto* load_mod =
        reinterpret_cast<rl_gpu_load_module_fn>(::dlsym(lib, "rl_gpu_load_module"));
    auto* mod_free =
        reinterpret_cast<rl_module_free_fn>(::dlsym(lib, "rl_module_free"));
    auto* b_new =
        reinterpret_cast<rl_pm4_builder_new_fn>(::dlsym(lib, "rl_pm4_builder_new"));
    auto* b_free = reinterpret_cast<rl_pm4_builder_free_fn>(
        ::dlsym(lib, "rl_pm4_builder_free"));
    auto* dispatch =
        reinterpret_cast<rl_pm4_dispatch_fn>(::dlsym(lib, "rl_pm4_dispatch"));
    auto* finalize =
        reinterpret_cast<rl_pm4_finalize_fn>(::dlsym(lib, "rl_pm4_finalize"));
    auto* set_k = reinterpret_cast<rl_pm4_ib_set_kernargs_fn>(
        ::dlsym(lib, "rl_pm4_ib_set_kernargs"));
    auto* replay =
        reinterpret_cast<rl_pm4_replay_fn>(::dlsym(lib, "rl_pm4_replay"));
    auto* ib_free =
        reinterpret_cast<rl_pm4_ib_free_fn>(::dlsym(lib, "rl_pm4_ib_free"));

    if (!load_mod || !mod_free || !b_new || !dispatch || !finalize || !set_k ||
        !replay || !ib_free) {
        g_err += " micro=FAIL_syms";
        return;
    }

    std::ifstream ifs(hsaco, std::ios::binary | std::ios::ate);
    if (!ifs) {
        g_err += " micro=FAIL_open_hsaco";
        return;
    }
    const auto sz = static_cast<std::streamoff>(ifs.tellg());
    if (sz <= 0) {
        g_err += " micro=FAIL_empty_hsaco";
        return;
    }
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> code(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(code.data()), sz)) {
        g_err += " micro=FAIL_read_hsaco";
        return;
    }

    const char* symbol_env = std::getenv("MLX_REDLINE_SYMBOL");
    const char* symbol =
        (symbol_env && symbol_env[0] != '\0') ? symbol_env : "acc_k.kd";

    int tokens = 64;
    if (const char* t = std::getenv("MLX_REDLINE_MICRO_TOKENS")) {
        char* end = nullptr;
        long v = std::strtol(t, &end, 10);
        if (end != t && v >= 1 && v <= 4096) {
            tokens = static_cast<int>(v);
        }
    }

    // --- P6: product stable buffers (graph_decode_*) ---
    auto& pos_arr = graph_decode_pos();
    auto& input_arr = graph_decode_input();
    void* pos_ptr0 = graph_decode_device_data_ptr(pos_arr);
    void* input_ptr0 = graph_decode_device_data_ptr(input_arr);
    if (!pos_ptr0 || !input_ptr0) {
        g_err += " gd_bind=FAIL_null_ptr micro=skip";
        return;
    }
    // Mutate in place (no realloc) and re-resolve — addresses must stay fixed.
    set_graph_decode_pos(0);
    set_graph_decode_pos(7);
    advance_graph_decode_pos(1);
    void* pos_ptr1 = graph_decode_device_data_ptr(pos_arr);
    void* input_ptr1 = graph_decode_device_data_ptr(input_arr);
    const bool stable = (pos_ptr1 == pos_ptr0) && (input_ptr1 == input_ptr0);
    {
        std::ostringstream oss;
        oss << " gd_bind=" << (stable ? "PASS" : "FAIL")
            << " pos=0x" << std::hex
            << reinterpret_cast<uintptr_t>(pos_ptr0) << std::dec
            << " input=0x" << std::hex
            << reinterpret_cast<uintptr_t>(input_ptr0) << std::dec;
        g_err += oss.str();
    }
    if (!stable) {
        g_err += " micro=skip_gd_unstable";
        return;
    }

    // Accumulator = product graph_decode_pos device buffer (int32/u32 slot).
    auto* d_acc = static_cast<unsigned int*>(pos_ptr0);
    if (hipMemset(d_acc, 0, sizeof(unsigned int)) != hipSuccess) {
        g_err += " micro=FAIL_hipMemset";
        return;
    }
    (void)hipDeviceSynchronize();

    void* mod = nullptr;
    if (load_mod(gpu, code.data(), code.size(), &mod) != kRlOk || !mod) {
        g_err += " micro=FAIL_load_module";
        return;
    }

    // kernarg: [acc:u64 @0][val:u32 @8] — pad to 512 like decode_kernargs.c
    // Bake stable product pos ptr once (not hipMalloc).
    uint8_t karg[512];
    std::memset(karg, 0, sizeof(karg));
    const uint64_t acc_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_acc));
    std::memcpy(karg, &acc_ptr, sizeof(acc_ptr));

    void* builder = b_new(gpu);
    if (!builder) {
        mod_free(mod);
        g_err += " micro=FAIL_builder";
        return;
    }
    if (dispatch(
            builder,
            mod,
            symbol,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            karg,
            sizeof(karg)) != kRlOk) {
        if (b_free) {
            b_free(builder);
        }
        mod_free(mod);
        g_err += " micro=FAIL_dispatch";
        return;
    }

    void* ib = nullptr;
    if (finalize(gpu, builder, &ib) != kRlOk || !ib) {
        mod_free(mod);
        g_err += " micro=FAIL_finalize";
        return;
    }

    uint64_t expected = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned int t = 1; t <= static_cast<unsigned int>(tokens); ++t) {
        if (set_k(ib, 0, 8, reinterpret_cast<const uint8_t*>(&t), sizeof(t)) !=
            kRlOk) {
            ib_free(ib);
            mod_free(mod);
            g_err += " micro=FAIL_set_kernargs";
            return;
        }
        if (replay(ib) != kRlOk) {
            ib_free(ib);
            mod_free(mod);
            g_err += " micro=FAIL_replay";
            return;
        }
        expected += t;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double host_total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();

    unsigned int observed = 0;
    if (hipMemcpy(
            &observed, d_acc, sizeof(unsigned int), hipMemcpyDeviceToHost) !=
        hipSuccess) {
        ib_free(ib);
        mod_free(mod);
        g_err += " micro=FAIL_hipMemcpy";
        return;
    }

    // Post-check: device ptr still stable after retained replays.
    void* pos_ptr2 = graph_decode_device_data_ptr(pos_arr);
    if (pos_ptr2 != pos_ptr0) {
        g_err += " gd_post=FAIL_moved";
    } else {
        g_err += " gd_post=stable";
    }

    const bool pass = (observed == static_cast<unsigned int>(expected));
    g_err += " micro=";
    g_err += pass ? "PASS" : "FAIL";
    g_err += " observed=" + std::to_string(observed);
    g_err += " expected=" + std::to_string(expected);
    g_err += " tokens=" + std::to_string(tokens);
    g_err += " host_total_us=" + std::to_string(host_total_us);
    g_err += " (NOT gen t/s)";

    // Restore product pos (micro used it as acc).
    set_graph_decode_pos(0);

    // P7: arm retained sidecar — dedicated hipMalloc acc so product pos stays free.
    // Default OFF unless MLX_REDLINE_SIDECAR=1 and micro PASS.
    const bool want_sidecar = env_exact_one("MLX_REDLINE_SIDECAR") && pass;
    if (!want_sidecar) {
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=skip";
        return;
    }

    unsigned int* side_acc = nullptr;
    if (hipMalloc(&side_acc, sizeof(unsigned int)) != hipSuccess || !side_acc) {
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=FAIL_hipMalloc";
        return;
    }
    if (hipMemset(side_acc, 0, sizeof(unsigned int)) != hipSuccess) {
        (void)hipFree(side_acc);
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=FAIL_hipMemset";
        return;
    }
    const uint64_t side_ptr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(side_acc));
    if (set_k(ib, 0, 0, reinterpret_cast<const uint8_t*>(&side_ptr), sizeof(side_ptr)) !=
        kRlOk) {
        (void)hipFree(side_acc);
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=FAIL_rebind_acc";
        return;
    }
    // Clear stale micro val@8; prime new acc pointer; ensure GPU sees memset.
    {
        const unsigned int zero = 0;
        if (set_k(ib, 0, 8, reinterpret_cast<const uint8_t*>(&zero), sizeof(zero)) !=
            kRlOk) {
            (void)hipFree(side_acc);
            ib_free(ib);
            mod_free(mod);
            g_err += " sidecar=FAIL_prime_val";
            return;
        }
        if (replay(ib) != kRlOk) {
            (void)hipFree(side_acc);
            ib_free(ib);
            mod_free(mod);
            g_err += " sidecar=FAIL_prime_replay";
            return;
        }
        if (hipMemset(side_acc, 0, sizeof(unsigned int)) != hipSuccess) {
            (void)hipFree(side_acc);
            ib_free(ib);
            mod_free(mod);
            g_err += " sidecar=FAIL_rezero";
            return;
        }
        (void)hipDeviceSynchronize();
    }

    // Inline correctness smoke (does not need model load / L=1).
    int side_tokens = 16;
    if (const char* st = std::getenv("MLX_REDLINE_SIDECAR_TOKENS")) {
        char* end = nullptr;
        long v = std::strtol(st, &end, 10);
        if (end != st && v >= 1 && v <= 4096) {
            side_tokens = static_cast<int>(v);
        }
    }
    uint64_t side_expected = 0;
    const auto s0 = std::chrono::steady_clock::now();
    for (unsigned int t = 1; t <= static_cast<unsigned int>(side_tokens); ++t) {
        if (set_k(ib, 0, 8, reinterpret_cast<const uint8_t*>(&t), sizeof(t)) !=
            kRlOk) {
            (void)hipFree(side_acc);
            ib_free(ib);
            mod_free(mod);
            g_err += " sidecar=FAIL_set_kernargs";
            return;
        }
        if (replay(ib) != kRlOk) {
            (void)hipFree(side_acc);
            ib_free(ib);
            mod_free(mod);
            g_err += " sidecar=FAIL_replay";
            return;
        }
        side_expected += t;
    }
    const auto s1 = std::chrono::steady_clock::now();
    const double side_us =
        std::chrono::duration<double, std::micro>(s1 - s0).count();
    unsigned int side_obs = 0;
    if (hipMemcpy(
            &side_obs, side_acc, sizeof(unsigned int), hipMemcpyDeviceToHost) !=
        hipSuccess) {
        (void)hipFree(side_acc);
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=FAIL_hipMemcpy";
        return;
    }
    const bool side_pass = (side_obs == static_cast<unsigned int>(side_expected));
    g_err += " sidecar=";
    g_err += side_pass ? "PASS" : "FAIL";
    g_err += " side_obs=" + std::to_string(side_obs);
    g_err += " side_exp=" + std::to_string(side_expected);
    g_err += " side_tokens=" + std::to_string(side_tokens);
    g_err += " side_host_us=" + std::to_string(side_us);
    g_err += " (NOT gen t/s)";

    if (!side_pass) {
        (void)hipFree(side_acc);
        ib_free(ib);
        mod_free(mod);
        return;
    }

    // Keep retained resources for L=1 ticks (product forward still owns call_fn).
    if (hipMemset(side_acc, 0, sizeof(unsigned int)) != hipSuccess) {
        (void)hipFree(side_acc);
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=FAIL_reset";
        return;
    }
    g_gpu = gpu;
    g_mod = mod;
    g_ib = ib;
    g_side_acc = side_acc;
    g_fn_replay = replay;
    g_fn_set_k = set_k;
    g_fn_mod_free = mod_free;
    g_fn_ib_free = ib_free;
    g_sidecar_ready = true;
    g_sidecar_n = 0;
    g_sidecar_expected = 0;
    g_err += " sidecar_armed=1";
}

// Returns true if caller should gpu_free(gpu) (sidecar did not retain it).
bool try_micro_op_and_maybe_arm(void* lib, void* gpu, rl_gpu_free_fn gpu_free) {
    g_fn_gpu_free = gpu_free;
    try_micro_op(lib, gpu);
    if (g_sidecar_ready) {
        return false; // keep gpu
    }
    return true;
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
        // Typical residual: executable DT_RPATH puts conda/miniforge before
        // ROCm core → Redline load_symbols binds HSA without ROCm ≥7.14
        // symbols (hsa_amd_counted_queue_acquire). Fix: RUNPATH + core first
        // (CMake chat/server link options). See P2_INIT.md.
        g_err = "abi=" + std::to_string(ver) +
                " gpu_new=null (check RUNPATH: /opt/rocm/core/lib before conda; "
                "needs HSA with counted_queue_acquire)";
        return true;
    }
    g_err = "abi=" + std::to_string(ver) + " gpu_new=ok";
    // P5/P6 micro; optional P7 sidecar arm (keeps gpu when armed).
    if (try_micro_op_and_maybe_arm(lib, gpu, gpu_free)) {
        gpu_free(gpu);
    }
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
                << "[redline] session READY (P2/P5/P6/P7; forward still product; "
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

void maybe_probe_redline_graph_decode_bind() {
#if !defined(MLX_BUILD_ROCM)
    return;
#else
    // P6: stable graph_decode_* buffer addresses (E4 kernarg-patch hinge).
    // Only when master env is exact "1". Does not enable HIP graphs / pure-graph.
    // Does not replace call_fn. NOT gen t/s.
    if (!env_exact_one("MLX_REDLINE_DECODE")) {
        return;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return; // XOR path: stay silent (session fail-closed already logged)
    }
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;

    namespace mx = mlx::core;
    try {
        auto& in = graph_decode_input();
        auto& pos = graph_decode_pos();
        // Prefer VRAM addresses (RocmBuffer::data) for future HSA kernarg bake;
        // raw_ptr alone may be host shadow on discrete GPUs.
        void* in0 = graph_decode_device_data_ptr(in);
        void* pos0 = graph_decode_device_data_ptr(pos);

        // In-place mutations that must NOT reallocate (same as pure-graph loop).
        set_graph_decode_pos(0);
        mx::array tok = mx::array(static_cast<int32_t>(1), mx::int32);
        mx::eval(tok);
        set_graph_decode_input_from(tok);
        mx::synchronize();

        void* in1 = graph_decode_device_data_ptr(in);
        void* pos1 = graph_decode_device_data_ptr(pos);

        const bool nonnull = (in0 != nullptr && pos0 != nullptr);
        const bool stable = nonnull && (in0 == in1) && (pos0 == pos1);

        std::ostringstream oss;
        oss << "[redline] gd_bind " << (stable ? "PASS" : "FAIL")
            << " input=" << in0 << " pos=" << pos0
            << " stable=" << (stable ? 1 : 0)
            << " (P6 VRAM ptr; not gen t/s; forward still product)\n";
        std::cerr << oss.str();
        if (!stable) {
            g_err += " gd_bind=FAIL";
        } else {
            g_err += " gd_bind=PASS";
        }
    } catch (const std::exception& e) {
        std::cerr << "[redline] gd_bind FAIL exception: " << e.what()
                  << " (P6; not gen t/s)\n";
        g_err += " gd_bind=FAIL_exc";
    }
#endif
}

void maybe_redline_sidecar_l1() {
#if !defined(MLX_BUILD_ROCM)
    return;
#else
    // P7: L=1 retained patch+replay sidecar. Default OFF (MLX_REDLINE_SIDECAR=1).
    // Does not replace call_fn. NOT gen t/s.
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_SIDECAR")) {
        return;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return;
    }

    // Ensure session/micro/arm ran (early chat probe may have already).
    (void)redline_session_ensure_init();

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_sidecar_ready || !g_ib || !g_fn_set_k || !g_fn_replay) {
        return;
    }

    g_sidecar_n += 1;
    const unsigned int t = g_sidecar_n;
    if (g_fn_set_k(
            g_ib, 0, 8, reinterpret_cast<const uint8_t*>(&t), sizeof(t)) !=
        kRlOk) {
        if (!g_sidecar_first_logged) {
            g_sidecar_first_logged = true;
            std::cerr << "[redline] sidecar L1 FAIL set_kernargs n=" << t
                      << " (forward still product; NOT gen t/s)\n";
        }
        return;
    }
    if (g_fn_replay(g_ib) != kRlOk) {
        if (!g_sidecar_first_logged) {
            g_sidecar_first_logged = true;
            std::cerr << "[redline] sidecar L1 FAIL replay n=" << t
                      << " (forward still product; NOT gen t/s)\n";
        }
        return;
    }
    g_sidecar_expected += t;

    if (!g_sidecar_first_logged) {
        g_sidecar_first_logged = true;
        std::cerr
            << "[redline] sidecar L1 tick (retained PM4; call_fn still product; "
               "NOT gen t/s)\n";
    }
#endif
}

} // namespace mlx_lm
