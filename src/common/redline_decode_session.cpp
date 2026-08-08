// Copyright © 2024-2025 Apple Inc. — Ported to C++
// P2: dlopen Redline C-API init smoke (no forward replacement).
// P5/P6: optional in-process PM4 micro-op when MLX_REDLINE_HSACO is set;
//     P6 bakes graph_decode_pos device ptr as accumulator. NOT gen t/s.
// P7: optional L=1 sidecar (MLX_REDLINE_SIDECAR=1) — retained IB patch+replay
//     alongside product call_fn; does not replace forward; NOT gen t/s.
// P8: optional L=1 small-op (MLX_REDLINE_SMALL_OP=1) — engine-owned product
//     graph_decode_input VRAM consume + retained PM4; call_fn still product.
// P9: MLX_REDLINE_OWN_GLUE=1 — Redline PM4 owns product glue launches
//     (pos_set/pos_inc/scalar_copy) instead of mlx hipLaunchKernelGGL.
// P10: retained PM4 IBs for OWN_GLUE (set_kernargs+replay; not one-shot
//     builder/finalize per product glue call). Default still OFF.
// P12: MLX_REDLINE_OWN_RMSNORM=1 — Redline PM4 owns packed product RMSNorm
//     launches (multi-instance non-qmm family; mid-eval stream sync). Default OFF.
// P12b/P12d POST: MLX_REDLINE_POST_SYNC=auto|device|stream|off (default auto).
//     auto/off: no extra host fence after rl_pm4_replay (API already waits).
// P12c/P12d PRE: set_k-before-pre; MLX_REDLINE_PRE_SYNC=stream|force|device|off
//     (default stream = hipStreamQuery then Sync if needed). RMS_PROFILE=1 timers.
//     Same-queue HIP attach not in redline-capi (documented P12d limit).

#include <mlx-lm/common/graph_decode.h>
#include <mlx-lm/common/redline_decode_session.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mlx/mlx.h>

#if defined(MLX_BUILD_ROCM)
#include <dlfcn.h>
// HIP host headers require an explicit platform macro (see server CMake).
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime.h>
// Product HIP stream for OWN_GLUE Path B (same queue as gpu_kv_pos_*).
#include <mlx/backend/rocm/device.h>
#endif

namespace mlx_lm {

namespace {

bool env_exact_one(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1' && v[1] == '\0';
}

// P12b/P12d: post-replay host fence after Redline retained PM4.
// redline-capi: rl_pm4_replay = submit + wait for wave retirement. Default
// "auto" skips an extra host fence (was double-wait). "device"/"stream" force
// extra fences for paranoid cross-engine A/B; "off" same as auto for post.
enum class RedlinePostSync : uint8_t {
    Auto = 0,   // P12d default: trust rl_pm4_replay wait
    Device = 1,
    Stream = 2,
    Off = 3,
};

// P12c/P12d: pre-replay host fence so product HIP producers complete before
// Redline reads VRAM. Default "stream" = Query then Sync if not ready (P12d).
// "force" = always Synchronize (P12c historical). "device"/"off" as before.
enum class RedlinePreSync : uint8_t {
    Stream = 0, // P12d: query-then-sync
    Force = 1,  // always hipStreamSynchronize (legacy stream behavior)
    Device = 2,
    Off = 3,
};

RedlinePostSync post_sync_mode() {
    const char* v = std::getenv("MLX_REDLINE_POST_SYNC");
    if (!v || v[0] == '\0' ||
        (v[0] == 'a' && v[1] == 'u' && v[2] == 't' && v[3] == 'o' &&
         v[4] == '\0')) {
        return RedlinePostSync::Auto;
    }
    if (v[0] == 'o' && v[1] == 'f' && v[2] == 'f' && v[3] == '\0') {
        return RedlinePostSync::Off;
    }
    if (v[0] == 's' && v[1] == 't' && v[2] == 'r' && v[3] == 'e' &&
        v[4] == 'a' && v[5] == 'm' && v[6] == '\0') {
        return RedlinePostSync::Stream;
    }
    if (v[0] == 'd' && v[1] == 'e' && v[2] == 'v' && v[3] == 'i' &&
        v[4] == 'c' && v[5] == 'e' && v[6] == '\0') {
        return RedlinePostSync::Device;
    }
    return RedlinePostSync::Auto;
}

RedlinePreSync pre_sync_mode() {
    const char* v = std::getenv("MLX_REDLINE_PRE_SYNC");
    if (!v || v[0] == '\0') {
        return RedlinePreSync::Stream;
    }
    if (v[0] == 'o' && v[1] == 'f' && v[2] == 'f' && v[3] == '\0') {
        return RedlinePreSync::Off;
    }
    if (v[0] == 'd' && v[1] == 'e' && v[2] == 'v' && v[3] == 'i' &&
        v[4] == 'c' && v[5] == 'e' && v[6] == '\0') {
        return RedlinePreSync::Device;
    }
    // force = always Synchronize (pre-P12d stream behavior)
    if (v[0] == 'f' && v[1] == 'o' && v[2] == 'r' && v[3] == 'c' &&
        v[4] == 'e' && v[5] == '\0') {
        return RedlinePreSync::Force;
    }
    if (v[0] == 's' && v[1] == 't' && v[2] == 'r' && v[3] == 'e' &&
        v[4] == 'a' && v[5] == 'm' && v[6] == '\0') {
        return RedlinePreSync::Stream;
    }
    return RedlinePreSync::Stream;
}

const char* pre_sync_label() {
    switch (pre_sync_mode()) {
        case RedlinePreSync::Off:
            return "off";
        case RedlinePreSync::Device:
            return "device";
        case RedlinePreSync::Force:
            return "force";
        case RedlinePreSync::Stream:
        default:
            return "stream"; // query-then-sync
    }
}

const char* post_sync_label() {
    switch (post_sync_mode()) {
        case RedlinePostSync::Off:
            return "off";
        case RedlinePostSync::Stream:
            return "stream";
        case RedlinePostSync::Device:
            return "device";
        case RedlinePostSync::Auto:
        default:
            return "auto";
    }
}

// P12d profile: how often pre-sync skipped via hipStreamQuery.
uint64_t g_pre_query_skip = 0;
uint64_t g_pre_sync_wait = 0;

#if defined(MLX_BUILD_ROCM)
void redline_post_sync(void* hip_stream) {
    switch (post_sync_mode()) {
        case RedlinePostSync::Auto:
        case RedlinePostSync::Off:
            // P12d: rl_pm4_replay already waited for Redline waves.
            return;
        case RedlinePostSync::Stream:
            if (hip_stream) {
                (void)hipStreamSynchronize(static_cast<hipStream_t>(hip_stream));
            } else {
                (void)hipDeviceSynchronize();
            }
            return;
        case RedlinePostSync::Device:
            (void)hipDeviceSynchronize();
            return;
    }
}

// Drain product HIP producers immediately before Redline replay reads VRAM.
// P12d stream mode: hipStreamQuery — Synchronize only if work still pending.
void redline_pre_sync(void* hip_stream) {
    switch (pre_sync_mode()) {
        case RedlinePreSync::Off:
            return;
        case RedlinePreSync::Device:
            (void)hipDeviceSynchronize();
            ++g_pre_sync_wait;
            return;
        case RedlinePreSync::Force:
            if (hip_stream) {
                (void)hipStreamSynchronize(static_cast<hipStream_t>(hip_stream));
            } else {
                (void)hipDeviceSynchronize();
            }
            ++g_pre_sync_wait;
            return;
        case RedlinePreSync::Stream:
        default:
            if (hip_stream) {
                auto* st = static_cast<hipStream_t>(hip_stream);
                const hipError_t q = hipStreamQuery(st);
                if (q == hipSuccess) {
                    // All prior work on this stream already complete.
                    ++g_pre_query_skip;
                    return;
                }
                // hipErrorNotReady or other → full drain (correctness).
                (void)hipGetLastError(); // clear sticky NotReady if needed
                (void)hipStreamSynchronize(st);
                ++g_pre_sync_wait;
            } else {
                (void)hipDeviceSynchronize();
                ++g_pre_sync_wait;
            }
            return;
    }
}
#endif

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
// P13 PR-A: wait HIP stream + replay (phase1 host join; phase2 WAIT_REG_MEM).
// Path B async: submit_after (no wait_signal) + consumer WaitValue on product stream.
using rl_pm4_replay_after_hip_stream_fn = int32_t (*)(void* ib, void* hip_stream);
using rl_pm4_submit_after_hip_stream_fn = int32_t (*)(void* ib, void* hip_stream);
using rl_gpu_consumer_wait_hip_stream_fn = int32_t (*)(void* hip_stream);
using rl_pm4_wait_fn = int32_t (*)(void* ib);
using rl_pm4_submit_fn = int32_t (*)(void* ib);
using rl_feature_bits_fn = uint32_t (*)();
using rl_pm4_ib_set_kernargs_fn = int32_t (*)(
    void* ib,
    size_t dispatch_index,
    size_t byte_offset,
    const uint8_t* kernarg,
    size_t len);
using rl_pm4_ib_free_fn = void (*)(void* ib);

constexpr uint32_t kRlFeatureHipStreamWait = 1u;

// P7/P8 retained IB (process lifetime when SIDECAR=1 or SMALL_OP=1 + micro PASS).
void* g_gpu = nullptr;
void* g_mod = nullptr;
void* g_ib = nullptr;
unsigned int* g_side_acc = nullptr;
rl_pm4_replay_fn g_fn_replay = nullptr;
rl_pm4_replay_after_hip_stream_fn g_fn_replay_after_hip = nullptr;
rl_pm4_replay_after_hip_stream_fn g_fn_replay_after_hip_p2 = nullptr;
rl_pm4_submit_after_hip_stream_fn g_fn_submit_after_p2 = nullptr;
rl_gpu_consumer_wait_hip_stream_fn g_fn_consumer_wait = nullptr;
rl_pm4_wait_fn g_fn_pm4_wait = nullptr;
rl_pm4_submit_fn g_fn_pm4_submit = nullptr;
bool g_hip_stream_bridge = false;
bool g_hip_stream_phase2 = false;
bool g_hip_stream_async = false; // symbols for Path B present
// IBs that had async submit and still need rl_pm4_wait before set_k reuse.
std::unordered_map<void*, bool> g_async_ib_needs_wait;
rl_pm4_ib_set_kernargs_fn g_fn_set_k = nullptr;
rl_gpu_free_fn g_fn_gpu_free = nullptr;
rl_module_free_fn g_fn_mod_free = nullptr;
rl_pm4_ib_free_fn g_fn_ib_free = nullptr;
bool g_sidecar_ready = false;
bool g_small_op_mode = false; // true when armed under MLX_REDLINE_SMALL_OP=1
void* g_small_op_input_ptr0 = nullptr; // product graph_decode_input VRAM at arm
uint32_t g_sidecar_n = 0;
uint64_t g_sidecar_expected = 0;
bool g_sidecar_first_logged = false;
uint32_t g_small_op_pos_book = 0; // host bookkeeping; does not enable external_pos

// P9/P10 glue ownership (process lifetime when OWN_GLUE=1).
void* g_glue_mod = nullptr;
void* g_glue_ib_set = nullptr;  // retained glue_pos_set.kd
void* g_glue_ib_inc = nullptr;  // retained glue_pos_inc.kd
void* g_glue_ib_copy = nullptr; // retained glue_scalar_copy_i32.kd
bool g_glue_armed = false;
bool g_glue_logged = false;
rl_pm4_builder_new_fn g_fn_b_new = nullptr;
rl_pm4_builder_free_fn g_fn_b_free = nullptr;
rl_pm4_dispatch_fn g_fn_dispatch = nullptr;
rl_pm4_finalize_fn g_fn_finalize = nullptr;

// P12 OWN_RMSNORM (process lifetime when OWN_RMSNORM=1).
void* g_rms_mod = nullptr;
void* g_rms_gpu = nullptr; // keep session gpu for lazy IB build
bool g_rms_armed = false;
bool g_rms_logged = false;
bool g_rms_profile_logged = false;
uint64_t g_rms_own_count = 0;
uint64_t g_rms_fallback_count = 0;
// P12c optional host-phase accumulators (ns); enabled by MLX_REDLINE_RMS_PROFILE=1.
// These are SUMS over g_rms_own_count calls — profile print shows total + mean.
// g_rms_ns_ordered is NOT pure RL doorbell: ordered join (PRE drain / bridge)
// + retained submit + host completion wait (or async submit+WaitValue enqueue).
uint64_t g_rms_ns_setk = 0;
uint64_t g_rms_ns_pre = 0;
uint64_t g_rms_ns_ordered = 0; // was mislabeled "replay"
// Path B split of ordered_join (async only; sum ≈ ordered when both recorded).
uint64_t g_rms_ns_submit = 0; // idle Query + submit_after / rl_pm4_submit
uint64_t g_rms_ns_cwait = 0;  // hipStreamWaitValue32 enqueue (consumer)
uint64_t g_rms_ns_post = 0;
uint64_t g_rms_ns_ib_wait = 0; // host rl_pm4_wait for IB reuse (Path B)
uint64_t g_rms_ib_wait_count = 0;
uint64_t g_rms_ib_dbl_skip_wait = 0; // dual-buffer avoided host wait
// Hot-path flag cache (filled under g_mu on first OWN_RMSNORM). Env is stable
// for a bench process; avoids repeated getenv on ~31 owns/token.
struct RmsHotFlags {
    bool inited = false;
    bool decode = false;
    bool own_rms = false;
    bool graph_pure = false;
    bool phase2 = false;
    bool phase2_async = false;
    bool async_hostwait = false;
    bool profile = false;
};
RmsHotFlags g_rms_hot{};
// key = (dtype_code << 24) | n_rows → dual retained IBs for Path B overlap
struct RmsIbPair {
    void* slot[2] = {nullptr, nullptr};
    uint8_t next = 0; // which slot to try first
};
std::unordered_map<uint32_t, RmsIbPair> g_rms_ib_by_key;
constexpr int kRmsBlock = 256;
constexpr size_t kRmsKargLive = 40; // 3×u64 + f32 + u32 + i64

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

    // P7/P8: arm retained IB — dedicated hipMalloc acc so product pos stays free.
    // Default OFF unless SIDECAR=1 or SMALL_OP=1 and micro PASS.
    const bool want_small_op = env_exact_one("MLX_REDLINE_SMALL_OP") && pass;
    const bool want_sidecar = env_exact_one("MLX_REDLINE_SIDECAR") && pass;
    if (!want_sidecar && !want_small_op) {
        ib_free(ib);
        mod_free(mod);
        g_err += " sidecar=skip";
        return;
    }
    if (want_small_op) {
        g_err += " small_op=want";
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
    g_small_op_mode = want_small_op;
    g_small_op_input_ptr0 = input_ptr0;
    g_sidecar_n = 0;
    g_sidecar_expected = 0;
    g_small_op_pos_book = 0;
    g_err += " sidecar_armed=1";
    if (want_small_op) {
        g_err += " small_op_armed=1";
    }
}

// P9/P10: load glue CO + arm retained IBs (product-equivalent pos/token).
// Keeps gpu on success. Product path uses set_kernargs+replay (not one-shot).
// Forward: defined with HIP stream bridge helpers.
void resolve_hip_stream_bridge(void* lib);

bool try_arm_glue(void* lib, void* gpu) {
    if (!env_exact_one("MLX_REDLINE_OWN_GLUE")) {
        g_err += " glue=skip";
        return false;
    }
    // Path B symbols (same .so as RMS) — required even when OWN_RMSNORM is off.
    resolve_hip_stream_bridge(lib);
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
    auto* gpu_free_fn =
        reinterpret_cast<rl_gpu_free_fn>(::dlsym(lib, "rl_gpu_free"));
    // P10: set_k required for retained path.
    if (!load_mod || !mod_free || !b_new || !dispatch || !finalize || !set_k ||
        !replay || !ib_free || !gpu_free_fn) {
        g_err += " glue=FAIL_syms";
        return false;
    }

    const char* path = std::getenv("MLX_REDLINE_GLUE_HSACO");
    const char* candidates[] = {
        path && path[0] ? path : nullptr,
        "docs/experiments/redline-kernel-launch/logs/glue_kernels-gfx1150.co",
        "/home/antmi/lemon-mlx-engine/docs/experiments/redline-kernel-launch/logs/"
        "glue_kernels-gfx1150.co",
        nullptr,
    };
    std::vector<uint8_t> code;
    const char* used = nullptr;
    for (const char** c = candidates; *c; ++c) {
        if (!*c) {
            continue;
        }
        std::ifstream ifs(*c, std::ios::binary | std::ios::ate);
        if (!ifs) {
            continue;
        }
        auto sz = static_cast<std::streamoff>(ifs.tellg());
        if (sz <= 0) {
            continue;
        }
        ifs.seekg(0);
        code.resize(static_cast<size_t>(sz));
        if (!ifs.read(reinterpret_cast<char*>(code.data()), sz)) {
            continue;
        }
        used = *c;
        break;
    }
    if (!used || code.empty()) {
        g_err += " glue=FAIL_open_hsaco";
        return false;
    }

    void* mod = nullptr;
    if (load_mod(gpu, code.data(), code.size(), &mod) != kRlOk || !mod) {
        g_err += " glue=FAIL_load_module";
        return false;
    }

    auto& pos = graph_decode_pos();
    void* p = graph_decode_device_data_ptr(pos);
    if (!p) {
        mod_free(mod);
        g_err += " glue=FAIL_null_pos";
        return false;
    }

    uint8_t karg[512];
    auto pack_pos_val = [&](int32_t v) {
        std::memset(karg, 0, sizeof(karg));
        const uint64_t pp = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
        std::memcpy(karg, &pp, sizeof(pp));
        std::memcpy(karg + 8, &v, sizeof(v));
    };

    // Build a retained single-dispatch IB for a glue kernel (P10).
    auto build_retained = [&](const char* symbol, void** out_ib) -> bool {
        void* builder = b_new(gpu);
        if (!builder) {
            return false;
        }
        if (dispatch(
                builder, mod, symbol, 1, 1, 1, 1, 1, 1, 0, karg, sizeof(karg)) !=
            kRlOk) {
            if (b_free) {
                b_free(builder);
            }
            return false;
        }
        void* ib = nullptr;
        if (finalize(gpu, builder, &ib) != kRlOk || !ib) {
            return false;
        }
        *out_ib = ib;
        return true;
    };

    auto fail_cleanup = [&](void*& ib_a, void*& ib_b, void*& ib_c, int32_t* tmp) {
        if (ib_a) {
            ib_free(ib_a);
            ib_a = nullptr;
        }
        if (ib_b) {
            ib_free(ib_b);
            ib_b = nullptr;
        }
        if (ib_c) {
            ib_free(ib_c);
            ib_c = nullptr;
        }
        if (tmp) {
            (void)hipFree(tmp);
        }
        mod_free(mod);
    };

    // Placeholder kernargs for IB bake (patched before every product replay).
    pack_pos_val(0);
    void* ib_set = nullptr;
    void* ib_inc = nullptr;
    void* ib_copy = nullptr;
    if (!build_retained("glue_pos_set.kd", &ib_set)) {
        fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
        g_err += " glue=FAIL_build_pos_set";
        return false;
    }
    if (!build_retained("glue_pos_inc.kd", &ib_inc)) {
        fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
        g_err += " glue=FAIL_build_pos_inc";
        return false;
    }

    auto& in = graph_decode_input();
    void* in_ptr = graph_decode_device_data_ptr(in);
    if (!in_ptr) {
        fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
        g_err += " glue=FAIL_null_input";
        return false;
    }
    int32_t* tmp = nullptr;
    if (hipMalloc(&tmp, sizeof(int32_t)) != hipSuccess || !tmp) {
        fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
        g_err += " glue=FAIL_tmp_malloc";
        return false;
    }
    {
        std::memset(karg, 0, sizeof(karg));
        const uint64_t dst_u =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(in_ptr));
        const uint64_t src_u =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp));
        std::memcpy(karg, &dst_u, sizeof(dst_u));
        std::memcpy(karg + 8, &src_u, sizeof(src_u));
    }
    if (!build_retained("glue_scalar_copy_i32.kd", &ib_copy)) {
        fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
        g_err += " glue=FAIL_build_copy";
        return false;
    }

    // Only patch the live kernarg prefix (kernel segment is << 512; full-buffer
    // set_k returns RL_ERR_RECORD when len exceeds allocated segment).
    // Prefer two small patches (u64@0 + i32/u64@8) matching micro-op style.
    auto replay_karg = [&](void* ib) -> int32_t {
        int32_t rc = set_k(ib, 0, 0, karg, 8);
        if (rc != kRlOk) {
            return rc;
        }
        rc = set_k(ib, 0, 8, karg + 8, 8);
        if (rc != kRlOk) {
            // i32-only kernels may have segment size 12 (u64+i32); try 4B @8.
            rc = set_k(ib, 0, 8, karg + 8, 4);
            if (rc != kRlOk) {
                return rc;
            }
        }
        return replay(ib);
    };

    // Correctness smoke via retained path: set → 7, inc +3 → 10, copy → 42.
    pack_pos_val(7);
    {
        const int32_t rc = replay_karg(ib_set);
        if (rc != kRlOk) {
            fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
            g_err += " glue=FAIL_pos_set_smoke rc=" + std::to_string(rc);
            return false;
        }
    }
    (void)hipDeviceSynchronize();
    int32_t host_v = 0;
    if (hipMemcpy(&host_v, p, sizeof(host_v), hipMemcpyDeviceToHost) != hipSuccess ||
        host_v != 7) {
        fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
        g_err += " glue=FAIL_pos_set_verify obs=" + std::to_string(host_v);
        return false;
    }

    pack_pos_val(3);
    {
        const int32_t rc = replay_karg(ib_inc);
        if (rc != kRlOk) {
            fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
            g_err += " glue=FAIL_pos_inc_smoke rc=" + std::to_string(rc);
            return false;
        }
    }
    (void)hipDeviceSynchronize();
    if (hipMemcpy(&host_v, p, sizeof(host_v), hipMemcpyDeviceToHost) != hipSuccess ||
        host_v != 10) {
        fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
        g_err += " glue=FAIL_pos_inc_verify obs=" + std::to_string(host_v);
        return false;
    }

    const int32_t v42 = 42;
    if (hipMemcpy(tmp, &v42, sizeof(v42), hipMemcpyHostToDevice) != hipSuccess) {
        fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
        g_err += " glue=FAIL_tmp_h2d";
        return false;
    }
    {
        std::memset(karg, 0, sizeof(karg));
        const uint64_t dst_u =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(in_ptr));
        const uint64_t src_u =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp));
        std::memcpy(karg, &dst_u, sizeof(dst_u));
        std::memcpy(karg + 8, &src_u, sizeof(src_u));
    }
    {
        const int32_t rc = replay_karg(ib_copy);
        if (rc != kRlOk) {
            fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
            g_err += " glue=FAIL_copy_smoke rc=" + std::to_string(rc);
            return false;
        }
    }
    (void)hipDeviceSynchronize();
    int32_t host_in = 0;
    if (hipMemcpy(&host_in, in_ptr, sizeof(host_in), hipMemcpyDeviceToHost) !=
            hipSuccess ||
        host_in != 42) {
        fail_cleanup(ib_set, ib_inc, ib_copy, tmp);
        g_err += " glue=FAIL_copy_verify obs=" + std::to_string(host_in);
        return false;
    }
    (void)hipFree(tmp);
    tmp = nullptr;

    // Host-wall microbench: N× one-shot builder vs N× retained set_k+replay
    // on glue_pos_set only. NOT gen t/s.
    constexpr int kGlueBenchN = 64;
    auto one_shot_pos_set = [&]() -> bool {
        void* builder = b_new(gpu);
        if (!builder) {
            return false;
        }
        if (dispatch(
                builder,
                mod,
                "glue_pos_set.kd",
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
            return false;
        }
        void* ib = nullptr;
        if (finalize(gpu, builder, &ib) != kRlOk || !ib) {
            return false;
        }
        const bool ok = (replay(ib) == kRlOk);
        ib_free(ib);
        return ok;
    };

    pack_pos_val(1);
    // Warmup
    for (int i = 0; i < 4; ++i) {
        pack_pos_val(i);
        (void)one_shot_pos_set();
        (void)replay_karg(ib_set);
    }
    (void)hipDeviceSynchronize();

    const auto t_os0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kGlueBenchN; ++i) {
        pack_pos_val(i);
        if (!one_shot_pos_set()) {
            fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
            g_err += " glue=FAIL_bench_oneshot";
            return false;
        }
    }
    const auto t_os1 = std::chrono::steady_clock::now();
    const double oneshot_us =
        std::chrono::duration<double, std::micro>(t_os1 - t_os0).count();

    const auto t_rt0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kGlueBenchN; ++i) {
        pack_pos_val(i);
        if (replay_karg(ib_set) != kRlOk) {
            fail_cleanup(ib_set, ib_inc, ib_copy, nullptr);
            g_err += " glue=FAIL_bench_retained";
            return false;
        }
    }
    const auto t_rt1 = std::chrono::steady_clock::now();
    const double retained_us =
        std::chrono::duration<double, std::micro>(t_rt1 - t_rt0).count();
    const double oneshot_med = oneshot_us / static_cast<double>(kGlueBenchN);
    const double retained_med = retained_us / static_cast<double>(kGlueBenchN);
    const double speedup =
        (retained_med > 0.0) ? (oneshot_med / retained_med) : 0.0;

    // Restore product buffers via raw H2D (do not re-enter OWN_GLUE).
    host_v = 0;
    (void)hipMemcpy(p, &host_v, sizeof(host_v), hipMemcpyHostToDevice);
    host_in = 0;
    (void)hipMemcpy(in_ptr, &host_in, sizeof(host_in), hipMemcpyHostToDevice);

    if (!g_gpu) {
        g_gpu = gpu;
        g_fn_gpu_free = gpu_free_fn;
    }
    if (!g_fn_mod_free) {
        g_fn_mod_free = mod_free;
    }
    g_glue_mod = mod;
    g_glue_ib_set = ib_set;
    g_glue_ib_inc = ib_inc;
    g_glue_ib_copy = ib_copy;
    g_fn_b_new = b_new;
    g_fn_b_free = b_free;
    g_fn_dispatch = dispatch;
    g_fn_finalize = finalize;
    g_fn_set_k = set_k;
    g_fn_replay = replay;
    g_fn_ib_free = ib_free;
    g_glue_armed = true;
    {
        std::ostringstream oss;
        oss << " glue=PASS glue_armed=1 retained=1 set=7 inc=10 copy=42"
            << " oneshot_us_per=" << oneshot_med
            << " retained_us_per=" << retained_med
            << " speedup=" << speedup << "x N=" << kGlueBenchN
            << " (NOT gen t/s)";
        g_err += oss.str();
    }
    return true;
}

// Pack packed-RMSNorm kernarg (matches harness/rms_norm_kernels.hip).
void pack_rms_karg(
    uint8_t* karg,
    size_t karg_cap,
    const void* x,
    const void* w,
    void* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride) {
    std::memset(karg, 0, karg_cap);
    const uint64_t xu = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x));
    const uint64_t wu = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(w));
    const uint64_t ou = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out));
    std::memcpy(karg + 0, &xu, 8);
    std::memcpy(karg + 8, &wu, 8);
    std::memcpy(karg + 16, &ou, 8);
    std::memcpy(karg + 24, &eps, 4);
    std::memcpy(karg + 28, &axis_size, 4);
    std::memcpy(karg + 32, &w_stride, 8);
}

const char* rms_symbol_for_dtype(int dtype_code) {
    switch (dtype_code) {
        case 0:
            return "rms_norm_f32.kd";
        case 1:
            return "rms_norm_f16.kd";
        case 2:
            return "rms_norm_bf16.kd";
        default:
            return nullptr;
    }
}

// Forward: defined with HIP stream bridge helpers below.
bool ensure_ib_ready_for_setk(void* ib);

// Build one retained IB for (dtype, n_rows). Caller holds g_mu.
// Redline rl_pm4_dispatch grid is **total workitems** (HIP gridDim×blockDim),
// not HIP gridDim; block is workgroup size. See redline-capi rl_pm4_dispatch.
void* build_one_rms_ib(int dtype_code, uint32_t n_rows) {
    if (!g_rms_armed || !g_rms_mod || !g_rms_gpu || !g_fn_b_new || !g_fn_dispatch ||
        !g_fn_finalize || n_rows == 0) {
        return nullptr;
    }
    const char* sym = rms_symbol_for_dtype(dtype_code);
    if (!sym) {
        return nullptr;
    }
    const uint32_t block = static_cast<uint32_t>(kRmsBlock);
    // Total workitems = n_rows * block (one HIP block per row).
    const uint64_t work_u = static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(block);
    if (work_u > 0xFFFFFFFFull) {
        return nullptr;
    }
    const uint32_t work_x = static_cast<uint32_t>(work_u);
    uint8_t karg[512];
    pack_rms_karg(karg, sizeof(karg), nullptr, nullptr, nullptr, 1e-6f, 1, 1);
    void* builder = g_fn_b_new(g_rms_gpu);
    if (!builder) {
        return nullptr;
    }
    if (g_fn_dispatch(
            builder,
            g_rms_mod,
            sym,
            work_x,
            1,
            1,
            block,
            1,
            1,
            0,
            karg,
            sizeof(karg)) != kRlOk) {
        if (g_fn_b_free) {
            g_fn_b_free(builder);
        }
        return nullptr;
    }
    void* ib = nullptr;
    if (g_fn_finalize(g_rms_gpu, builder, &ib) != kRlOk || !ib) {
        return nullptr;
    }
    return ib;
}

void free_all_rms_ibs() {
    for (auto& kv : g_rms_ib_by_key) {
        for (int s = 0; s < 2; ++s) {
            if (kv.second.slot[s] && g_fn_ib_free) {
                g_fn_ib_free(kv.second.slot[s]);
            }
        }
    }
    g_rms_ib_by_key.clear();
    g_async_ib_needs_wait.clear();
}

// Path B: dual retained IBs per shape so set_k/submit can use a free slot while
// the other is still in flight (product stream WaitValue orders consumers).
// Non-async: always slot0 (single IB). Caller holds g_mu.
void* acquire_rms_ib(int dtype_code, uint32_t n_rows, bool dual_for_async) {
    if (!g_rms_armed || n_rows == 0) {
        return nullptr;
    }
    const uint32_t key =
        (static_cast<uint32_t>(dtype_code) << 24) | (n_rows & 0x00FFFFFFu);
    auto it = g_rms_ib_by_key.find(key);
    if (it == g_rms_ib_by_key.end()) {
        // Cap cache — dual slots double memory; keep key budget modest.
        if (g_rms_ib_by_key.size() >= 32) {
            return nullptr;
        }
        RmsIbPair pair;
        pair.slot[0] = build_one_rms_ib(dtype_code, n_rows);
        if (!pair.slot[0]) {
            return nullptr;
        }
        if (dual_for_async) {
            pair.slot[1] = build_one_rms_ib(dtype_code, n_rows);
            // If second fails, continue single-slot (still correct).
        }
        it = g_rms_ib_by_key.emplace(key, pair).first;
    }
    RmsIbPair& pair = it->second;
    // Lazy second slot if async became available after first build.
    if (dual_for_async && !pair.slot[1]) {
        pair.slot[1] = build_one_rms_ib(dtype_code, n_rows);
    }

    auto slot_busy = [](void* ib) -> bool {
        if (!ib) {
            return true;
        }
        auto wit = g_async_ib_needs_wait.find(ib);
        return wit != g_async_ib_needs_wait.end() && wit->second;
    };

    if (!dual_for_async || !pair.slot[1]) {
        // Single-IB: host-wait prior async if any, then reuse.
        if (!ensure_ib_ready_for_setk(pair.slot[0])) {
            return nullptr;
        }
        return pair.slot[0];
    }

    // Dual: prefer a slot that does not need host wait.
    for (int attempt = 0; attempt < 2; ++attempt) {
        const uint8_t idx = static_cast<uint8_t>((pair.next + attempt) & 1u);
        void* ib = pair.slot[idx];
        if (!ib) {
            continue;
        }
        if (!slot_busy(ib)) {
            pair.next = static_cast<uint8_t>((idx + 1u) & 1u);
            ++g_rms_ib_dbl_skip_wait;
            return ib;
        }
    }
    // Both in flight — host-join the next slot (oldest in round-robin).
    void* ib = pair.slot[pair.next & 1u];
    if (!ensure_ib_ready_for_setk(ib)) {
        return nullptr;
    }
    pair.next = static_cast<uint8_t>((pair.next + 1u) & 1u);
    return ib;
}

// Smoke / simple path: one IB for (dtype, n_rows).
void* get_or_build_rms_ib(int dtype_code, uint32_t n_rows) {
    return acquire_rms_ib(dtype_code, n_rows, /*dual_for_async=*/false);
}

// P13: optional PR-A symbols (hip stream wait + ordered replay + Path B async).
void resolve_hip_stream_bridge(void* lib) {
    g_fn_replay_after_hip = nullptr;
    g_fn_replay_after_hip_p2 = nullptr;
    g_fn_submit_after_p2 = nullptr;
    g_fn_consumer_wait = nullptr;
    g_fn_pm4_wait = nullptr;
    g_fn_pm4_submit = nullptr;
    g_hip_stream_bridge = false;
    g_hip_stream_phase2 = false;
    g_hip_stream_async = false;
    if (!lib) {
        return;
    }
    auto* feat = reinterpret_cast<rl_feature_bits_fn>(::dlsym(lib, "rl_feature_bits"));
    auto* after = reinterpret_cast<rl_pm4_replay_after_hip_stream_fn>(
        ::dlsym(lib, "rl_pm4_replay_after_hip_stream"));
    auto* after2 = reinterpret_cast<rl_pm4_replay_after_hip_stream_fn>(
        ::dlsym(lib, "rl_pm4_replay_after_hip_stream_phase2"));
    auto* submit2 = reinterpret_cast<rl_pm4_submit_after_hip_stream_fn>(
        ::dlsym(lib, "rl_pm4_submit_after_hip_stream_phase2"));
    auto* cwait = reinterpret_cast<rl_gpu_consumer_wait_hip_stream_fn>(
        ::dlsym(lib, "rl_gpu_consumer_wait_hip_stream"));
    auto* pwait = reinterpret_cast<rl_pm4_wait_fn>(::dlsym(lib, "rl_pm4_wait"));
    auto* psubmit = reinterpret_cast<rl_pm4_submit_fn>(::dlsym(lib, "rl_pm4_submit"));
    (void)feat;
    if (after) {
        g_fn_replay_after_hip = after;
        g_hip_stream_bridge = true;
    }
    if (after2) {
        g_fn_replay_after_hip_p2 = after2;
        g_hip_stream_phase2 = true;
    }
    if (submit2 && cwait && pwait) {
        g_fn_submit_after_p2 = submit2;
        g_fn_consumer_wait = cwait;
        g_fn_pm4_wait = pwait;
        g_fn_pm4_submit = psubmit; // optional idle-stream fast path
        g_hip_stream_async = true;
    }
    (void)kRlFeatureHipStreamWait;
}

#if defined(MLX_BUILD_ROCM)
// Product default-stream HIP handle (encoder for default_stream). Used by OWN_GLUE
// Path B so WaitValue / WriteValue order with gpu_kv_pos_* / matmul on that stream.
void* product_default_hip_stream() {
    try {
        namespace mx = mlx::core;
        auto& enc =
            mx::rocm::get_command_encoder(mx::default_stream(mx::default_device()));
        return static_cast<void*>(static_cast<hipStream_t>(enc.stream()));
    } catch (...) {
        return nullptr;
    }
}
#else
void* product_default_hip_stream() {
    return nullptr;
}
#endif

// Shared Path B launch for any retained owned IB (RMS, glue, …).
// Returns false on hard failure. Sets *mode_out to phase tag string.
// When used_async_out non-null, sets whether consumer WaitValue path was used.
bool launch_owned_ib(
    void* ib,
    void* hip_stream,
    const char** mode_out,
    bool* used_async_out) {
    if (mode_out) {
        *mode_out = "replay";
    }
    if (used_async_out) {
        *used_async_out = false;
    }
    if (!ib || !g_fn_replay) {
        return false;
    }
    const RedlinePreSync pre_mode = pre_sync_mode();
    const bool want_phase2 = env_exact_one("MLX_REDLINE_PHASE2");
    const bool want_async =
        want_phase2 && env_exact_one("MLX_REDLINE_PHASE2_ASYNC") && g_hip_stream_async &&
        hip_stream && pre_mode != RedlinePreSync::Off &&
        pre_mode != RedlinePreSync::Device;
    const bool async_hostwait = want_async && env_exact_one("MLX_REDLINE_ASYNC_HOSTWAIT");
    bool used_bridge = false;
    bool used_async = false;
    bool used_async_hostwait = false;

    if (want_async) {
        ++g_pre_sync_wait;
        bool submitted = false;
        if (g_fn_pm4_submit && hip_stream) {
            auto* st = static_cast<hipStream_t>(hip_stream);
            if (hipStreamQuery(st) == hipSuccess) {
                ++g_pre_query_skip;
                submitted = (g_fn_pm4_submit(ib) == kRlOk);
            } else {
                (void)hipGetLastError();
            }
        }
        if (!submitted && g_fn_submit_after_p2) {
            submitted = (g_fn_submit_after_p2(ib, hip_stream) == kRlOk);
        }
        if (!submitted) {
            if (g_fn_replay_after_hip_p2 &&
                g_fn_replay_after_hip_p2(ib, hip_stream) == kRlOk) {
                used_bridge = true;
            } else if (g_fn_replay_after_hip &&
                       g_fn_replay_after_hip(ib, hip_stream) == kRlOk) {
                used_bridge = true;
            } else {
                return false;
            }
        } else if (async_hostwait) {
            g_async_ib_needs_wait[ib] = true;
            if (!g_fn_pm4_wait || g_fn_pm4_wait(ib) != kRlOk) {
                return false;
            }
            g_async_ib_needs_wait[ib] = false;
            used_bridge = true;
            used_async_hostwait = true;
        } else {
            g_async_ib_needs_wait[ib] = true;
            if (!g_fn_consumer_wait || g_fn_consumer_wait(hip_stream) != kRlOk) {
                if (g_fn_pm4_wait) {
                    (void)g_fn_pm4_wait(ib);
                    g_async_ib_needs_wait[ib] = false;
                }
                return false;
            }
            used_bridge = true;
            used_async = true;
        }
    } else {
        auto* ordered = g_fn_replay_after_hip;
        if (want_phase2 && g_fn_replay_after_hip_p2) {
            ordered = g_fn_replay_after_hip_p2;
        }
        if (ordered && hip_stream && pre_mode != RedlinePreSync::Off &&
            pre_mode != RedlinePreSync::Device) {
            if (pre_mode == RedlinePreSync::Stream) {
                auto* st = static_cast<hipStream_t>(hip_stream);
                if (hipStreamQuery(st) == hipSuccess) {
                    ++g_pre_query_skip;
                    if (g_fn_replay(ib) != kRlOk) {
                        return false;
                    }
                } else {
                    (void)hipGetLastError();
                    ++g_pre_sync_wait;
                    if (ordered(ib, hip_stream) != kRlOk) {
                        return false;
                    }
                    used_bridge = true;
                }
            } else {
                ++g_pre_sync_wait;
                if (ordered(ib, hip_stream) != kRlOk) {
                    return false;
                }
                used_bridge = true;
            }
        } else {
            redline_pre_sync(hip_stream);
            if (g_fn_replay(ib) != kRlOk) {
                return false;
            }
        }
    }

    if (!used_async) {
        redline_post_sync(hip_stream);
    }
    if (used_async_out) {
        *used_async_out = used_async;
    }
    if (mode_out) {
        if (used_async) {
            *mode_out = "phase2-async-used";
        } else if (used_async_hostwait) {
            *mode_out = "phase2-async-hostwait";
        } else if (used_bridge && want_phase2 && g_fn_replay_after_hip_p2) {
            *mode_out = "phase2-used";
        } else if (used_bridge) {
            *mode_out = "phase1-used";
        } else {
            *mode_out = "replay";
        }
    }
    return true;
}

// Before set_k on a retained IB, finish any prior async submit on that IB.
bool ensure_ib_ready_for_setk(void* ib) {
    if (!ib) {
        return false;
    }
    auto it = g_async_ib_needs_wait.find(ib);
    if (it == g_async_ib_needs_wait.end() || !it->second) {
        return true;
    }
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    if (!g_fn_pm4_wait || g_fn_pm4_wait(ib) != kRlOk) {
        ++g_rms_fallback_count;
        return false;
    }
    g_rms_ns_ib_wait += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0)
            .count());
    ++g_rms_ib_wait_count;
    it->second = false;
    return true;
}

// P12: arm OWN_RMSNORM retained module + correctness smoke (f32 n_rows=1).
// Keeps gpu on success. Product path uses set_kernargs+replay.
bool try_arm_rmsnorm(void* lib, void* gpu) {
    if (!env_exact_one("MLX_REDLINE_OWN_RMSNORM")) {
        g_err += " rms=skip";
        return false;
    }
    resolve_hip_stream_bridge(lib);
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
        g_err += " rms=FAIL_syms";
        return false;
    }

    const char* path = std::getenv("MLX_REDLINE_RMS_HSACO");
    const char* candidates[] = {
        path && path[0] ? path : nullptr,
        "docs/experiments/redline-kernel-launch/logs/rms_norm_kernels-gfx1150.co",
        "/home/antmi/lemon-mlx-engine/docs/experiments/redline-kernel-launch/logs/"
        "rms_norm_kernels-gfx1150.co",
        nullptr,
    };
    std::vector<uint8_t> code;
    const char* used = nullptr;
    for (const char** c = candidates; *c; ++c) {
        if (!*c) {
            continue;
        }
        std::ifstream ifs(*c, std::ios::binary | std::ios::ate);
        if (!ifs) {
            continue;
        }
        auto sz = static_cast<std::streamoff>(ifs.tellg());
        if (sz <= 0) {
            continue;
        }
        ifs.seekg(0);
        code.resize(static_cast<size_t>(sz));
        if (!ifs.read(reinterpret_cast<char*>(code.data()), sz)) {
            continue;
        }
        used = *c;
        break;
    }
    if (!used || code.empty()) {
        g_err += " rms=FAIL_open_hsaco";
        return false;
    }

    void* mod = nullptr;
    if (load_mod(gpu, code.data(), code.size(), &mod) != kRlOk || !mod) {
        g_err += " rms=FAIL_load_module";
        return false;
    }

    // Correctness smoke: axis=4, n_rows=1, f32, w=1, x=[1,2,3,4]
    constexpr uint32_t kAxis = 4;
    constexpr float kEps = 1e-6f;
    float h_x[4] = {1.f, 2.f, 3.f, 4.f};
    float h_w[4] = {1.f, 1.f, 1.f, 1.f};
    float h_out[4] = {0, 0, 0, 0};
    float sumsq = 0.f;
    for (int i = 0; i < 4; ++i) {
        sumsq += h_x[i] * h_x[i];
    }
    const float inv = 1.f / std::sqrt(sumsq / 4.f + kEps);
    float expect[4];
    for (int i = 0; i < 4; ++i) {
        expect[i] = h_w[i] * (h_x[i] * inv);
    }

    float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    if (hipMalloc(&d_x, sizeof(h_x)) != hipSuccess ||
        hipMalloc(&d_w, sizeof(h_w)) != hipSuccess ||
        hipMalloc(&d_out, sizeof(h_out)) != hipSuccess) {
        if (d_x) {
            (void)hipFree(d_x);
        }
        if (d_w) {
            (void)hipFree(d_w);
        }
        if (d_out) {
            (void)hipFree(d_out);
        }
        mod_free(mod);
        g_err += " rms=FAIL_malloc";
        return false;
    }
    (void)hipMemcpy(d_x, h_x, sizeof(h_x), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_w, h_w, sizeof(h_w), hipMemcpyHostToDevice);
    (void)hipMemset(d_out, 0, sizeof(h_out));

    // Install globals needed by get_or_build_rms_ib.
    g_rms_mod = mod;
    g_rms_gpu = gpu;
    g_fn_b_new = b_new;
    g_fn_b_free = b_free;
    g_fn_dispatch = dispatch;
    g_fn_finalize = finalize;
    g_fn_set_k = set_k;
    g_fn_replay = replay;
    g_fn_ib_free = ib_free;
    g_rms_armed = true; // temporarily so get_or_build works

    void* ib = get_or_build_rms_ib(/*f32*/ 0, /*n_rows*/ 1);
    if (!ib) {
        g_rms_armed = false;
        g_rms_mod = nullptr;
        g_rms_gpu = nullptr;
        mod_free(mod);
        (void)hipFree(d_x);
        (void)hipFree(d_w);
        (void)hipFree(d_out);
        g_err += " rms=FAIL_build_ib";
        return false;
    }

    uint8_t karg[512];
    pack_rms_karg(karg, sizeof(karg), d_x, d_w, d_out, kEps, kAxis, /*w_stride*/ 1);
    // Patch live kernarg in chunks (avoid RL_ERR_RECORD on oversize).
    auto set_prefix = [&](void* ibv) -> int32_t {
        int32_t rc = set_k(ibv, 0, 0, karg, 24); // 3 pointers
        if (rc != kRlOk) {
            return rc;
        }
        rc = set_k(ibv, 0, 24, karg + 24, 8); // eps + axis_size
        if (rc != kRlOk) {
            return rc;
        }
        return set_k(ibv, 0, 32, karg + 32, 8); // w_stride
    };
    if (set_prefix(ib) != kRlOk || replay(ib) != kRlOk) {
        g_rms_armed = false;
        free_all_rms_ibs();
        g_rms_mod = nullptr;
        g_rms_gpu = nullptr;
        mod_free(mod);
        (void)hipFree(d_x);
        (void)hipFree(d_w);
        (void)hipFree(d_out);
        g_err += " rms=FAIL_smoke_replay";
        return false;
    }
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(h_out, d_out, sizeof(h_out), hipMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        const float d = std::fabs(h_out[i] - expect[i]);
        if (!(d < 1e-4f)) {
            ok = false;
        }
    }
    (void)hipFree(d_x);
    (void)hipFree(d_w);
    (void)hipFree(d_out);

    if (!ok) {
        g_rms_armed = false;
        free_all_rms_ibs();
        g_rms_mod = nullptr;
        g_rms_gpu = nullptr;
        mod_free(mod);
        std::ostringstream oss;
        oss << " rms=FAIL_smoke_val out=[" << h_out[0] << "," << h_out[1] << ","
            << h_out[2] << "," << h_out[3] << "] exp=[" << expect[0] << ","
            << expect[1] << "," << expect[2] << "," << expect[3] << "]";
        g_err += oss.str();
        return false;
    }

    // Multi-dispatch chain smoke: N=4 dispatches in one retained IB (structural
    // multi-launch path for future fused chains; product RMSNorm uses 1-dispatch).
    {
        void* builder = b_new(gpu);
        if (builder) {
            uint8_t mk[512];
            pack_rms_karg(mk, sizeof(mk), nullptr, nullptr, nullptr, kEps, kAxis, 1);
            const uint32_t block = static_cast<uint32_t>(kRmsBlock);
            const uint32_t work = block; // n_rows=1 → workitems=block
            bool multi_ok = true;
            for (int i = 0; i < 4; ++i) {
                if (dispatch(
                        builder,
                        mod,
                        "rms_norm_f32.kd",
                        work,
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        mk,
                        sizeof(mk)) != kRlOk) {
                    multi_ok = false;
                    break;
                }
            }
            void* mib = nullptr;
            if (multi_ok && finalize(gpu, builder, &mib) == kRlOk && mib) {
                ib_free(mib); // structural PASS; product uses per-call single IB
                g_err += " rms_multi=PASS_n4";
            } else {
                g_err += " rms_multi=FAIL";
                // Non-fatal for arm if single smoke passed.
            }
        }
    }

    (void)used;
    g_rms_armed = true;
    {
        std::ostringstream oss;
        oss << " rms=PASS rms_armed=1 smoke_f32=[1,2,3,4] retained=1"
            << " (OWN_RMSNORM packed product path; NOT gen t/s)";
        g_err += oss.str();
    }
    return true;
}

// Returns true if caller should gpu_free(gpu) (nothing retained it).
bool try_micro_op_and_maybe_arm(void* lib, void* gpu, rl_gpu_free_fn gpu_free) {
    g_fn_gpu_free = gpu_free;
    try_micro_op(lib, gpu);
    bool keep = g_sidecar_ready;
    if (try_arm_glue(lib, gpu)) {
        keep = true;
    }
    if (try_arm_rmsnorm(lib, gpu)) {
        keep = true;
    }
    if (keep) {
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
                << "[redline] session READY (P2–P12; forward still product; "
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
    // When SMALL_OP=1, product-buffer ticks own the IB — skip synthetic n.
    // Does not replace call_fn. NOT gen t/s.
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_SIDECAR")) {
        return;
    }
    if (env_exact_one("MLX_REDLINE_SMALL_OP")) {
        return; // P8 owns L=1 retained ticks
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

void maybe_redline_small_op_l1(mlx::core::array& previous_token) {
#if !defined(MLX_BUILD_ROCM)
    (void)previous_token;
    return;
#else
    // P8: engine-owned L=1 small op consuming live graph_decode_* VRAM.
    // Default OFF (MLX_REDLINE_SMALL_OP=1). Does not replace call_fn. NOT gen t/s.
    // Does not set graph_external_pos — product RoPE/KV stays host-offset path.
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_SMALL_OP")) {
        return;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return;
    }

    (void)redline_session_ensure_init();

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_sidecar_ready || !g_small_op_mode || !g_ib || !g_fn_set_k ||
        !g_fn_replay) {
        return;
    }

    try {
        // Product stable input buffer: engine writes previous sample token.
        namespace mx = mlx::core;
        mx::eval(previous_token);
        set_graph_decode_input_from(previous_token);

        // Bookkeep pos in the fixed product buffer without enabling external_pos
        // (models only consume graph_decode_pos when graph_external_pos() is true).
        set_graph_decode_pos(static_cast<int>(g_small_op_pos_book));
        g_small_op_pos_book += 1;

        auto& in = graph_decode_input();
        auto& pos = graph_decode_pos();
        void* in_ptr = graph_decode_device_data_ptr(in);
        void* pos_ptr = graph_decode_device_data_ptr(pos);
        if (!in_ptr || !pos_ptr) {
            if (!g_sidecar_first_logged) {
                g_sidecar_first_logged = true;
                std::cerr
                    << "[redline] small_op L1 FAIL null_gd_ptr "
                       "(call_fn still product; NOT gen t/s)\n";
            }
            return;
        }
        if (g_small_op_input_ptr0 && in_ptr != g_small_op_input_ptr0) {
            if (!g_sidecar_first_logged) {
                g_sidecar_first_logged = true;
                std::cerr
                    << "[redline] small_op L1 FAIL input_ptr_moved "
                       "(call_fn still product; NOT gen t/s)\n";
            }
            return;
        }

        // Consume product VRAM: D2H token id from graph_decode_input device ptr.
        int32_t tok_i32 = 0;
        if (hipMemcpy(
                &tok_i32, in_ptr, sizeof(int32_t), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            if (!g_sidecar_first_logged) {
                g_sidecar_first_logged = true;
                std::cerr
                    << "[redline] small_op L1 FAIL_hipMemcpy_input "
                       "(call_fn still product; NOT gen t/s)\n";
            }
            return;
        }
        const unsigned int val = static_cast<unsigned int>(tok_i32);

        if (g_fn_set_k(
                g_ib,
                0,
                8,
                reinterpret_cast<const uint8_t*>(&val),
                sizeof(val)) != kRlOk) {
            if (!g_sidecar_first_logged) {
                g_sidecar_first_logged = true;
                std::cerr
                    << "[redline] small_op L1 FAIL set_kernargs val=" << val
                    << " (call_fn still product; NOT gen t/s)\n";
            }
            return;
        }
        if (g_fn_replay(g_ib) != kRlOk) {
            if (!g_sidecar_first_logged) {
                g_sidecar_first_logged = true;
                std::cerr
                    << "[redline] small_op L1 FAIL replay val=" << val
                    << " (call_fn still product; NOT gen t/s)\n";
            }
            return;
        }

        g_sidecar_n += 1;
        g_sidecar_expected += static_cast<uint64_t>(val);

        if (!g_sidecar_first_logged) {
            g_sidecar_first_logged = true;
            std::cerr
                << "[redline] small_op L1 tick (product graph_decode_input VRAM "
                   "val="
                << val
                << "; retained PM4; call_fn still product; NOT gen t/s)\n";
        }
    } catch (const std::exception& e) {
        if (!g_sidecar_first_logged) {
            g_sidecar_first_logged = true;
            std::cerr << "[redline] small_op L1 FAIL exception: " << e.what()
                      << " (call_fn still product; NOT gen t/s)\n";
        }
    }
#endif
}

void maybe_redline_sidecar_verify() {
#if !defined(MLX_BUILD_ROCM)
    return;
#else
    // P7b/P8: full-gen L=1 correctness — D2H side_acc vs host expected.
    // SIDECAR-only: sum(1..n). SMALL_OP: sum of product token ids from
    // graph_decode_input VRAM. Does not replace call_fn. NOT gen t/s.
    if (!env_exact_one("MLX_REDLINE_DECODE")) {
        return;
    }
    const bool want_side = env_exact_one("MLX_REDLINE_SIDECAR");
    const bool want_small = env_exact_one("MLX_REDLINE_SMALL_OP");
    if (!want_side && !want_small) {
        return;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_sidecar_ready || !g_side_acc || g_sidecar_n == 0) {
        return;
    }

    (void)hipDeviceSynchronize();
    unsigned int side_obs = 0;
    if (hipMemcpy(
            &side_obs,
            g_side_acc,
            sizeof(unsigned int),
            hipMemcpyDeviceToHost) != hipSuccess) {
        std::cerr
            << "[redline] "
            << (g_small_op_mode ? "small_op" : "sidecar")
            << " L1 fullgen FAIL_hipMemcpy n=" << g_sidecar_n
            << " (call_fn still product; NOT gen t/s)\n";
        return;
    }

    const uint64_t side_exp = g_sidecar_expected;
    bool pass = (static_cast<uint64_t>(side_obs) == side_exp);
    if (!g_small_op_mode) {
        // Acc kernel is u32; expected fits for small max_tokens research runs.
        pass = pass &&
               (side_exp ==
                (static_cast<uint64_t>(g_sidecar_n) *
                 (static_cast<uint64_t>(g_sidecar_n) + 1ULL)) /
                    2ULL);
    }

    std::cerr << "[redline] "
              << (g_small_op_mode ? "small_op" : "sidecar") << " L1 fullgen "
              << (pass ? "PASS" : "FAIL") << " n=" << g_sidecar_n
              << " side_obs=" << side_obs << " side_exp=" << side_exp
              << (g_small_op_mode
                      ? " (product graph_decode_input token-sum; call_fn still product; NOT gen t/s)\n"
                      : " (retained PM4 L=1 ticks; call_fn still product; NOT gen t/s)\n");
#endif
}

bool redline_try_own_pos_set(mlx::core::array& pos, int v) {
#if !defined(MLX_BUILD_ROCM)
    (void)pos;
    (void)v;
    return false;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_OWN_GLUE")) {
        return false;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return false;
    }
    // Do NOT call redline_session_ensure_init here: it holds g_mu and init
    // already calls set_graph_decode_pos (would self-deadlock). Glue arms
    // only during session init; until armed, fall back to product HIP.
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false; // init or another glue launch holds g_mu → HIP fallback
    }
    // P10: retained IB path (set_kernargs + Path B launch when stream bridge ok).
    if (!g_glue_armed || !g_glue_ib_set || !g_fn_set_k || !g_fn_replay) {
        return false;
    }
    void* p = graph_decode_device_data_ptr(pos);
    if (!p) {
        return false;
    }
    void* ib = g_glue_ib_set;
    if (!ensure_ib_ready_for_setk(ib)) {
        return false;
    }
    uint8_t karg[512];
    std::memset(karg, 0, sizeof(karg));
    const uint64_t pp = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    std::memcpy(karg, &pp, sizeof(pp));
    const int32_t vv = static_cast<int32_t>(v);
    std::memcpy(karg + 8, &vv, sizeof(vv));

    if (g_fn_set_k(ib, 0, 0, karg, 8) != kRlOk) {
        return false;
    }
    if (g_fn_set_k(ib, 0, 8, karg + 8, 4) != kRlOk) {
        return false;
    }
    void* hip_stream = product_default_hip_stream();
    const char* mode = "replay";
    const bool ok = launch_owned_ib(ib, hip_stream, &mode, nullptr);
    if (ok && !g_glue_logged) {
        g_glue_logged = true;
        std::cerr
            << "[redline] OWN_GLUE pos_set handled by Redline retained PM4 "
               "(product HIP glue skipped; P13 bridge="
            << (g_hip_stream_bridge ? "yes" : "no") << " " << mode
            << "; POST_SYNC=" << post_sync_label() << "; NOT gen t/s)\n";
    }
    return ok;
#endif
}

bool redline_try_own_pos_inc(mlx::core::array& pos, int delta) {
#if !defined(MLX_BUILD_ROCM)
    (void)pos;
    (void)delta;
    return false;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_OWN_GLUE")) {
        return false;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return false;
    }
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    if (!g_glue_armed || !g_glue_ib_inc || !g_fn_set_k || !g_fn_replay) {
        return false;
    }
    void* p = graph_decode_device_data_ptr(pos);
    if (!p) {
        return false;
    }
    void* ib = g_glue_ib_inc;
    if (!ensure_ib_ready_for_setk(ib)) {
        return false;
    }
    uint8_t karg[512];
    std::memset(karg, 0, sizeof(karg));
    const uint64_t pp = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
    std::memcpy(karg, &pp, sizeof(pp));
    const int32_t dd = static_cast<int32_t>(delta);
    std::memcpy(karg + 8, &dd, sizeof(dd));

    if (g_fn_set_k(ib, 0, 0, karg, 8) != kRlOk) {
        return false;
    }
    if (g_fn_set_k(ib, 0, 8, karg + 8, 4) != kRlOk) {
        return false;
    }
    void* hip_stream = product_default_hip_stream();
    const char* mode = "replay";
    return launch_owned_ib(ib, hip_stream, &mode, nullptr);
#endif
}

bool redline_try_own_scalar_copy_i32(mlx::core::array& dst, mlx::core::array& src) {
#if !defined(MLX_BUILD_ROCM)
    (void)dst;
    (void)src;
    return false;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE") || !env_exact_one("MLX_REDLINE_OWN_GLUE")) {
        return false;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return false;
    }
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    if (!g_glue_armed || !g_glue_ib_copy || !g_fn_set_k || !g_fn_replay) {
        return false;
    }
    namespace mx = mlx::core;
    mx::eval(src);
    void* d = graph_decode_device_data_ptr(dst);
    // src may be transient; still need VRAM/device-visible address.
    void* s = graph_decode_device_data_ptr(src);
    if (!d || !s) {
        return false;
    }
    void* ib = g_glue_ib_copy;
    if (!ensure_ib_ready_for_setk(ib)) {
        return false;
    }
    uint8_t karg[512];
    std::memset(karg, 0, sizeof(karg));
    const uint64_t dd = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d));
    const uint64_t ss = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(s));
    std::memcpy(karg, &dd, sizeof(dd));
    std::memcpy(karg + 8, &ss, sizeof(ss));

    if (g_fn_set_k(ib, 0, 0, karg, 8) != kRlOk) {
        return false;
    }
    if (g_fn_set_k(ib, 0, 8, karg + 8, 8) != kRlOk) {
        return false;
    }
    void* hip_stream = product_default_hip_stream();
    const char* mode = "replay";
    const bool ok = launch_owned_ib(ib, hip_stream, &mode, nullptr);
    if (ok && !g_glue_logged) {
        g_glue_logged = true;
        std::cerr
            << "[redline] OWN_GLUE scalar_copy/pos handled by Redline retained PM4 "
               "(product HIP glue skipped; P13 bridge="
            << (g_hip_stream_bridge ? "yes" : "no") << " " << mode
            << "; NOT gen t/s)\n";
    }
    return ok;
#endif
}

// P12: product packed RMSNorm ownership (called from C ABI + MLX weak hook).
bool redline_try_own_rmsnorm_packed(
    const void* x,
    const void* w,
    void* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride,
    uint32_t n_rows,
    int dtype_code,
    void* hip_stream) {
#if !defined(MLX_BUILD_ROCM)
    (void)x;
    (void)w;
    (void)out;
    (void)eps;
    (void)axis_size;
    (void)w_stride;
    (void)n_rows;
    (void)dtype_code;
    (void)hip_stream;
    return false;
#else
    if (!env_exact_one("MLX_REDLINE_DECODE") ||
        !env_exact_one("MLX_REDLINE_OWN_RMSNORM")) {
        return false;
    }
    if (env_exact_one("MLX_DECODE_GRAPH_PURE")) {
        return false;
    }
    if (!x || !w || !out || n_rows == 0 || axis_size == 0) {
        return false;
    }
    if (dtype_code < 0 || dtype_code > 2) {
        return false;
    }

    // Do NOT call redline_session_ensure_init under eval (deadlock risk).
    std::unique_lock<std::mutex> lock(g_mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        ++g_rms_fallback_count;
        return false;
    }
    if (!g_rms_armed || !g_fn_set_k || !g_fn_replay) {
        ++g_rms_fallback_count;
        return false;
    }

    // Process-stable hot flags (bench env does not flip mid-gen).
    if (!g_rms_hot.inited) {
        g_rms_hot.decode = env_exact_one("MLX_REDLINE_DECODE");
        g_rms_hot.own_rms = env_exact_one("MLX_REDLINE_OWN_RMSNORM");
        g_rms_hot.graph_pure = env_exact_one("MLX_DECODE_GRAPH_PURE");
        g_rms_hot.phase2 = env_exact_one("MLX_REDLINE_PHASE2");
        g_rms_hot.phase2_async = env_exact_one("MLX_REDLINE_PHASE2_ASYNC");
        g_rms_hot.async_hostwait = env_exact_one("MLX_REDLINE_ASYNC_HOSTWAIT");
        g_rms_hot.profile = env_exact_one("MLX_REDLINE_RMS_PROFILE");
        g_rms_hot.inited = true;
    }

    // Path B flags early: dual IB acquire needs to know if async is intended.
    const RedlinePreSync pre_mode = pre_sync_mode();
    const bool want_phase2 = g_rms_hot.phase2;
    const bool want_async =
        want_phase2 && g_rms_hot.phase2_async && g_hip_stream_async && hip_stream &&
        pre_mode != RedlinePreSync::Off && pre_mode != RedlinePreSync::Device;
    // Hostwait is explicit only — do not silently rewrite ASYNC into host join.
    const bool async_hostwait = want_async && g_rms_hot.async_hostwait;

    // Dual retained IBs when true Path B WaitValue (not hostwait): overlap set_k
    // on free slot while prior IB is still in flight on the RL queue.
    void* ib = acquire_rms_ib(dtype_code, n_rows, /*dual_for_async=*/want_async && !async_hostwait);
    if (!ib) {
        ++g_rms_fallback_count;
        return false;
    }

    const bool profile = g_rms_hot.profile;
    using clock = std::chrono::steady_clock;
    auto mark = []() { return clock::now(); };
    auto ns_since = [](clock::time_point t0) {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0)
                .count());
    };

    // P12c: pack + set_kernargs BEFORE pre-sync. Kernarg patch only stores
    // device pointers / scalars — it does not read producer VRAM — so this
    // host work can overlap in-flight product HIP producers. Pre-sync must
    // still complete immediately before replay (dual-queue producer drain).
    // Single contiguous set_k (was 3×24/8/8) — one memcpy into kernarg buffer.
    uint8_t karg[512];
    pack_rms_karg(karg, sizeof(karg), x, w, out, eps, axis_size, w_stride);
    auto t0 = profile ? mark() : clock::time_point{};
    if (g_fn_set_k(ib, 0, 0, karg, kRmsKargLive) != kRlOk) {
        ++g_rms_fallback_count;
        return false;
    }
    if (profile) {
        g_rms_ns_setk += ns_since(t0);
        t0 = mark();
    }

    // Path B async (opt-in): MLX_REDLINE_PHASE2=1 and MLX_REDLINE_PHASE2_ASYNC=1
    // → submit_after + hipStreamWaitValue32 on product stream (true Path B).
    // Consumer fence: HSA completion signal (redline b7ebc83+). Dual IBs cut
    // host rl_pm4_wait on consecutive same-shape owns. ASYNC_HOSTWAIT=1 is
    // diagnostic only. Else phase2 sync / phase1 ordered path.
    bool used_bridge = false;
    bool used_async = false;
    bool used_async_hostwait = false;

    if (profile) {
        t0 = mark();
    }

    if (want_async) {
        // Idle product stream: skip WriteValue+WAIT_REG_MEM (no outstanding
        // producers). Still publish completion WaitValue for consumers.
        // Note: mid-layer decode almost never Query-idle (WaitValue + product
        // work keep stream dirty) — pre_query_skip stays 0 for first token.
        bool submitted = false;
        uint64_t call_submit_ns = 0;
        uint64_t call_cwait_ns = 0;
        if (g_fn_pm4_submit && hip_stream) {
            auto* st = static_cast<hipStream_t>(hip_stream);
            if (hipStreamQuery(st) == hipSuccess) {
                ++g_pre_query_skip;
                submitted = (g_fn_pm4_submit(ib) == kRlOk);
            } else {
                (void)hipGetLastError();
                ++g_pre_sync_wait; // Query not-ready → submit_after path
            }
        } else {
            ++g_pre_sync_wait;
        }
        if (!submitted) {
            submitted = (g_fn_submit_after_p2(ib, hip_stream) == kRlOk);
        }
        if (profile) {
            call_submit_ns = ns_since(t0);
            g_rms_ns_submit += call_submit_ns;
            t0 = mark();
        }
        if (!submitted) {
            // Fall back to sync phase2 if available, else phase1.
            if (g_fn_replay_after_hip_p2 &&
                g_fn_replay_after_hip_p2(ib, hip_stream) == kRlOk) {
                used_bridge = true;
            } else if (g_fn_replay_after_hip &&
                       g_fn_replay_after_hip(ib, hip_stream) == kRlOk) {
                used_bridge = true;
            } else {
                ++g_rms_fallback_count;
                return false;
            }
            if (profile) {
                // Fallback path still counts full ordered wall in g_rms_ns_ordered.
                g_rms_ns_ordered += call_submit_ns + ns_since(t0);
                t0 = mark();
            }
        } else if (async_hostwait) {
            g_async_ib_needs_wait[ib] = true;
            if (!g_fn_pm4_wait || g_fn_pm4_wait(ib) != kRlOk) {
                ++g_rms_fallback_count;
                return false;
            }
            g_async_ib_needs_wait[ib] = false;
            used_bridge = true;
            used_async_hostwait = true;
            if (profile) {
                g_rms_ns_ordered += call_submit_ns + ns_since(t0);
                t0 = mark();
            }
        } else {
            g_async_ib_needs_wait[ib] = true;
            if (g_fn_consumer_wait(hip_stream) != kRlOk) {
                if (g_fn_pm4_wait) {
                    (void)g_fn_pm4_wait(ib);
                    g_async_ib_needs_wait[ib] = false;
                }
                ++g_rms_fallback_count;
                return false;
            }
            if (profile) {
                call_cwait_ns = ns_since(t0);
                g_rms_ns_cwait += call_cwait_ns;
                g_rms_ns_ordered += call_submit_ns + call_cwait_ns;
                t0 = mark();
            }
            used_bridge = true;
            used_async = true;
        }
    } else {
        // Prefer phase1 StreamSynchronize for gen unless PHASE2=1 (sync phase2).
        auto* ordered = g_fn_replay_after_hip;
        if (want_phase2 && g_fn_replay_after_hip_p2) {
            ordered = g_fn_replay_after_hip_p2;
        }
        if (ordered && pre_mode != RedlinePreSync::Off &&
            pre_mode != RedlinePreSync::Device) {
            if (pre_mode == RedlinePreSync::Stream && hip_stream) {
                auto* st = static_cast<hipStream_t>(hip_stream);
                if (hipStreamQuery(st) == hipSuccess) {
                    ++g_pre_query_skip;
                    if (profile) {
                        g_rms_ns_pre += ns_since(t0);
                        t0 = mark();
                    }
                    if (g_fn_replay(ib) != kRlOk) {
                        ++g_rms_fallback_count;
                        return false;
                    }
                } else {
                    (void)hipGetLastError();
                    ++g_pre_sync_wait;
                    if (profile) {
                        g_rms_ns_pre += ns_since(t0);
                        t0 = mark();
                    }
                    if (ordered(ib, hip_stream) != kRlOk) {
                        ++g_rms_fallback_count;
                        return false;
                    }
                    used_bridge = true;
                }
            } else {
                ++g_pre_sync_wait;
                if (profile) {
                    g_rms_ns_pre += ns_since(t0);
                    t0 = mark();
                }
                if (ordered(ib, hip_stream) != kRlOk) {
                    ++g_rms_fallback_count;
                    return false;
                }
                used_bridge = true;
            }
        } else {
            redline_pre_sync(hip_stream);
            if (profile) {
                g_rms_ns_pre += ns_since(t0);
                t0 = mark();
            }
            if (g_fn_replay(ib) != kRlOk) {
                ++g_rms_fallback_count;
                return false;
            }
        }
    }
    // Non-async: ordered_join is remaining wall after pre (submit+host wait).
    // Async paths already folded submit+cwait into g_rms_ns_ordered above.
    if (profile && !want_async) {
        g_rms_ns_ordered += ns_since(t0);
        t0 = mark();
    } else if (profile) {
        t0 = mark();
    }

    // P12d: post fence. Async WaitValue path: product stream already ordered —
    // skip device post (auto is no-op; force device would re-serialize).
    // Hostwait async already joined Redline; still allow post if forced.
    if (!used_async) {
        redline_post_sync(hip_stream);
    }
    if (profile) {
        g_rms_ns_post += ns_since(t0);
    }
    ++g_rms_own_count;
    if (!g_rms_logged) {
        g_rms_logged = true;
        const char* mode = "none";
        if (used_async) {
            mode = "phase2-async-used";
        } else if (used_async_hostwait) {
            mode = "phase2-async-hostwait";
        } else if (used_bridge && want_phase2 && g_fn_replay_after_hip_p2) {
            mode = "phase2-used";
        } else if (used_bridge) {
            mode = "phase1-used";
        }
        std::cerr
            << "[redline] OWN_RMSNORM packed launch handled by Redline retained PM4 "
               "(product HIP RMSNorm skipped; P12d/P13 bridge="
            << (g_hip_stream_bridge ? "yes" : "no") << " " << mode
            << "; set_k-before-pre; PRE_SYNC=" << pre_sync_label()
            << " POST_SYNC=" << post_sync_label() << "; NOT gen t/s)\n";
    }
    // One-shot host-phase profile after ~1 token of packed RMSNorms (~31).
    // Prints TOTAL and MEAN so "2022us" is not misread as 2ms per call.
    if (profile && !g_rms_profile_logged && g_rms_own_count >= 31) {
        g_rms_profile_logged = true;
        const double n = static_cast<double>(g_rms_own_count);
        auto us_total = [](uint64_t ns) { return static_cast<double>(ns) / 1000.0; };
        auto us_mean = [n](uint64_t ns) {
            return (n > 0.0) ? (static_cast<double>(ns) / 1000.0) / n : 0.0;
        };
        std::cerr << "[redline] OWN_RMSNORM host profile (n=" << g_rms_own_count
                  << " TOTAL): set_k=" << us_total(g_rms_ns_setk)
                  << "us pre_sync=" << us_total(g_rms_ns_pre)
                  << "us ordered_join=" << us_total(g_rms_ns_ordered)
                  << "us submit=" << us_total(g_rms_ns_submit)
                  << "us cwait=" << us_total(g_rms_ns_cwait)
                  << "us post_sync=" << us_total(g_rms_ns_post)
                  << "us ib_host_wait=" << us_total(g_rms_ns_ib_wait)
                  << "us | MEAN/call: set_k=" << us_mean(g_rms_ns_setk)
                  << "us pre_sync=" << us_mean(g_rms_ns_pre)
                  << "us ordered_join=" << us_mean(g_rms_ns_ordered)
                  << "us submit=" << us_mean(g_rms_ns_submit)
                  << "us cwait=" << us_mean(g_rms_ns_cwait)
                  << "us post_sync=" << us_mean(g_rms_ns_post)
                  << "us ib_host_wait=" << us_mean(g_rms_ns_ib_wait)
                  << "us | ordered_join=PRE_drain+RL_submit+WaitValue_enqueue"
                     " (async split submit|cwait; NOT pure doorbell; host wall; NOT gen t/s)"
                     " host_cut=setk1+flag_cache"
                  << " pre_query_skip=" << g_pre_query_skip
                  << " pre_wait=" << g_pre_sync_wait
                  << " ib_wait_n=" << g_rms_ib_wait_count
                  << " dbl_skip_wait=" << g_rms_ib_dbl_skip_wait << "\n";
    }
    return true;
#endif
}

} // namespace mlx_lm

// ---------------------------------------------------------------------------
// P12 C ABI — strong symbol for MLX weak hook (libmlx.a links into chat).
// ---------------------------------------------------------------------------
extern "C" bool mlx_redline_try_own_rmsnorm(
    const void* x,
    const void* w,
    void* out,
    float eps,
    uint32_t axis_size,
    int64_t w_stride,
    uint32_t n_rows,
    int dtype_code,
    void* hip_stream) {
    return mlx_lm::redline_try_own_rmsnorm_packed(
        x, w, out, eps, axis_size, w_stride, n_rows, dtype_code, hip_stream);
}
