// E2: HIP multi-kernel host-wall baseline (eager launches vs hipGraph).
// No-op floor_k chain — dispatch/submit cost, not compute.
//   hipcc --offload-arch=gfx1150 e2_hip_multi_wall.hip -o e2_hip_multi_wall
//   ./e2_hip_multi_wall [N=64] [M=100] [WARMUP=20]

#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" __global__ void floor_k() {}

#define CHECK(x)                                                                   \
    do {                                                                           \
        hipError_t _e = (x);                                                       \
        if (_e != hipSuccess) {                                                    \
            fprintf(stderr, "hip error %s:%d: %s\n", __FILE__, __LINE__,           \
                    hipGetErrorString(_e));                                        \
            return 1;                                                              \
        }                                                                          \
    } while (0)

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 64;
    int M = argc > 2 ? atoi(argv[2]) : 100;
    int WARMUP = argc > 3 ? atoi(argv[3]) : 20;

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    printf("e2_hip_multi_wall: device=%s arch=%d.%d N=%d M=%d warmup=%d\n",
           prop.name, prop.major, prop.minor, N, M, WARMUP);

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // ---- Arm A: eager HIP N launches + stream sync (host wall) ----
    for (int i = 0; i < WARMUP; i++) {
        for (int k = 0; k < N; k++) {
            floor_k<<<dim3(1), dim3(1), 0, stream>>>();
        }
        CHECK(hipStreamSynchronize(stream));
    }
    std::vector<double> eager_us;
    eager_us.reserve(M);
    for (int r = 0; r < M; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < N; k++) {
            floor_k<<<dim3(1), dim3(1), 0, stream>>>();
        }
        CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        eager_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double eager_med = median(eager_us);
    printf("HIP_eager:     N=%d  median host us/replay = %.3f  us/disp = %.4f\n",
           N, eager_med, eager_med / N);

    // ---- Arm B: hipGraph capture of N, launch+sync (host wall) ----
    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int k = 0; k < N; k++) {
        floor_k<<<dim3(1), dim3(1), 0, stream>>>();
    }
    hipGraph_t graph;
    CHECK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t gexec;
    CHECK(hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));

    for (int i = 0; i < WARMUP; i++) {
        CHECK(hipGraphLaunch(gexec, stream));
        CHECK(hipStreamSynchronize(stream));
    }
    std::vector<double> graph_us;
    graph_us.reserve(M);
    for (int r = 0; r < M; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        CHECK(hipGraphLaunch(gexec, stream));
        CHECK(hipStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        graph_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    double graph_med = median(graph_us);
    printf("HIP_graph:     N=%d  median host us/replay = %.3f  us/disp = %.4f\n",
           N, graph_med, graph_med / N);

    printf("ratio HIP_eager/HIP_graph = %.3fx  (higher => graph faster)\n",
           (graph_med > 0) ? eager_med / graph_med : 0.0);

    CHECK(hipGraphExecDestroy(gexec));
    CHECK(hipGraphDestroy(graph));
    CHECK(hipStreamDestroy(stream));
    return 0;
}
