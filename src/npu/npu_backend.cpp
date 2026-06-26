// NPU backend — invokes IRON JIT for NPU compute
// Open-source version uses Peano-compiled kernels (Apache 2.0)
// Full 31 TFLOPS version requires Chess license + xclbin
//
// For now, delegates to Python IRON JIT via subprocess.
// This ensures correctness (the JIT handles all XRT details)
// and avoids duplicating the complex XRT instruction-buffer flow.
//
// Future: direct C++ XRT path for lower latency.

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "mlx-lm/npu/npu_backend.h"

namespace npu {

namespace {

struct NPUState {
    bool initialized = false;
    std::string name;
    float peak_tflops = 0.0f;
    bool has_chess = false;  // true if Chess-compiled xclbin available
};

NPUState& state() {
    static NPUState s;
    return s;
}

// Path to the IRON JIT helper script
std::string jit_script_path() {
    const char* env = std::getenv("NPU_JIT_SCRIPT");
    if (env) return env;
    return NPU_INSTALL_DIR "/bin/npu_jit.py";
}

// Run a GEMM via the IRON JIT Python helper
bool run_jit_gemm(const int32_t* A, const int32_t* B, int32_t* C,
                  int M, int K, int N) {
    // Write inputs to temp files
    char a_path[] = "/tmp/npu_gemm_a_XXXXXX";
    char b_path[] = "/tmp/npu_gemm_b_XXXXXX";
    char c_path[] = "/tmp/npu_gemm_c_XXXXXX";
    int fd_a = mkstemp(a_path);
    int fd_b = mkstemp(b_path);
    int fd_c = mkstemp(c_path);
    
    if (fd_a < 0 || fd_b < 0 || fd_c < 0) return false;
    
    size_t bytes_a = M * K * sizeof(int32_t);
    size_t bytes_b = K * N * sizeof(int32_t);
    size_t bytes_c = M * N * sizeof(int32_t);
    
    write(fd_a, A, bytes_a); close(fd_a);
    write(fd_b, B, bytes_b); close(fd_b);
    write(fd_c, C, bytes_c); close(fd_c);
    
    // Call the IRON JIT Python script
    std::string cmd = "python3 " + jit_script_path()
        + " --a " + a_path
        + " --b " + b_path
        + " --c " + c_path
        + " --M " + std::to_string(M)
        + " --K " + std::to_string(K)
        + " --N " + std::to_string(N);
    
    int ret = std::system(cmd.c_str());
    
    // Read result
    std::ifstream result_f(c_path, std::ios::binary);
    if (result_f) {
        result_f.read(reinterpret_cast<char*>(C), bytes_c);
    }
    
    // Cleanup
    std::remove(a_path);
    std::remove(b_path);
    std::remove(c_path);
    
    return ret == 0;
}

} // anonymous namespace

bool init() {
    if (state().initialized) return true;

    // Check if NPU is accessible via XRT Python bindings
    // by checking that the IRON JIT can detect the device
    FILE* pipe = popen("python3 -c \"from aie.utils import has_xrt, get_current_device; print(has_xrt); d=get_current_device(); print(d.name if d else 'none')\" 2>/dev/null", "r");
    if (!pipe) {
        std::fprintf(stderr, "[NPU] Failed to probe NPU\n");
        return false;
    }
    
    std::array<char, 256> buf;
    bool xrt_ok = false;
    std::string dev_name;
    if (std::fgets(buf.data(), buf.size(), pipe)) {
        xrt_ok = (std::string(buf.data()) == "True\n");
    }
    if (std::fgets(buf.data(), buf.size(), pipe)) {
        dev_name = std::string(buf.data());
        if (!dev_name.empty() && dev_name.back() == '\n')
            dev_name.pop_back();
    }
    pclose(pipe);
    
    if (!xrt_ok) {
        std::fprintf(stderr, "[NPU] XRT not available\n");
        return false;
    }
    
    state().name = dev_name.empty() ? "RyzenAI" : dev_name;
    
    // Detect NPU type for peak TFLOPS
    if (state().name.find("npu5") != std::string::npos ||
        state().name.find("NPU5") != std::string::npos) {
        state().peak_tflops = 31.2f;
    } else if (state().name.find("npu4") != std::string::npos ||
               state().name.find("NPU4") != std::string::npos) {
        state().peak_tflops = 23.0f;
    } else if (state().name.find("npu3") != std::string::npos ||
               state().name.find("NPU3") != std::string::npos) {
        state().peak_tflops = 16.0f;
    } else {
        state().peak_tflops = 10.0f;
    }
    
    // Check for Chess-compiled xclbin (31 TFLOPS path)
    const char* xclbin_env = std::getenv("NPU_XCLBIN_PATH");
    if (xclbin_env) {
        std::ifstream f(xclbin_env);
        state().has_chess = f.good();
    }
    
    state().initialized = true;
    std::fprintf(stderr, "[NPU] %s (%.1f TFLOPS peak)%s\n",
                 state().name.c_str(), state().peak_tflops,
                 state().has_chess ? " [Chess xclbin available]" : "");
    return true;
}

bool is_available() {
    return state().initialized;
}

const char* device_name() {
    return state().name.c_str();
}

bool matmul(const int32_t* A, const int32_t* B, int32_t* C,
            int M, int K, int N) {
    if (!state().initialized) return false;
    
    // For now, use the JIT Python path for correctness.
    // The JIT caches compiled xclbins, so repeated calls
    // with the same shape are fast (no recompilation).
    return run_jit_gemm(A, B, C, M, K, N);
}

float peak_tflops() {
    return state().peak_tflops;
}

} // namespace npu
