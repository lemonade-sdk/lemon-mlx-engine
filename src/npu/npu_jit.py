#!/usr/bin/env python3
"""IRON JIT helper for NPU GEMM — called by the C++ NPU backend."""

import argparse
import numpy as np
import sys
import os

os.environ.setdefault("NPU_CACHE_HOME", "/tmp/npu_cache")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    args = parser.parse_args()

    # Read inputs
    A = np.fromfile(args.a, dtype=np.int32).reshape(args.M, args.K)
    B = np.fromfile(args.b, dtype=np.int32).reshape(args.K, args.N)

    import aie.iron as iron
    from aie.iron import In, Out, ExternalFunction, ObjectFifo, Program, Runtime, Worker
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from aie.utils import get_current_device

    # Check NPU availability
    dev = get_current_device()
    if dev is None:
        print("[NPU] No NPU device available", file=sys.stderr)
        sys.exit(1)

    M, K, N = args.M, args.K, args.N
    a_ty = np.ndarray[(M, K), np.dtype[np.int32]]
    b_ty = np.ndarray[(K, N), np.dtype[np.int32]]
    c_ty = np.ndarray[(M, N), np.dtype[np.int32]]

    # Kernel source — Peano-compiled vectorized GEMM
    kernel_src = f"/tmp/npu_gemm_{M}x{K}x{N}.cc"
    if not os.path.exists(kernel_src):
        with open(kernel_src, "w") as f:
            f.write(f'''
#include <stdint.h>
#include <aie2pintrin.h>
extern "C" void gemm(int32_t* a, int32_t* b, int32_t* c,
                      int32_t M, int32_t K, int32_t N) {{
    for (int i = 0; i < M; i++) {{
        int32_t* row_a = &a[i * K];
        for (int j = 0; j < N; j++) {{
            int32_t sum = 0;
            int k = 0;
            for (; k + 16 <= K; k += 16) {{
                v16int32 va = *(v16int32 *)&row_a[k];
                for (int v = 0; v < 16; v++) {{
                    sum += ((int32_t *)&va)[v] * b[(k + v) * N + j];
                }}
            }}
            for (; k < K; k++) sum += row_a[k] * b[k * N + j];
            c[i * N + j] = sum;
        }}
    }}
}}
''')

    @iron.jit
    def gemm_fn(a_in: In, b_in: In, c_out: Out):
        kfn = ExternalFunction("gemm", source_file=kernel_src,
                               arg_types=[a_ty, b_ty, c_ty, np.int32, np.int32, np.int32])
        oa = ObjectFifo(a_ty, name="a")
        ob = ObjectFifo(b_ty, name="b")
        oc = ObjectFifo(c_ty, name="c")
        def cf(a, b, c, kfn):
            ea = a.acquire(1); eb = b.acquire(1); ec = c.acquire(1)
            kfn(ea, eb, ec, M, K, N)
            c.release(1); b.release(1); a.release(1)
        w = Worker(cf, [oa.cons(), ob.cons(), oc.prod(), kfn], tile=Tile(0, 2))
        rt = Runtime()
        with rt.sequence(a_ty, b_ty, c_ty) as (a, b, c):
            rt.start(w)
            rt.fill(oa.prod(), a)
            rt.fill(ob.prod(), b)
            rt.drain(oc.cons(), c, wait=True)
        return Program(iron.get_current_device(), rt).resolve_program()

    # Run on NPU
    gemm_fn(A, B, np.zeros((M, N), dtype=np.int32))

    # Write output (already modified in-place by XRTTensor sync)
    C = np.zeros((M, N), dtype=np.int32)
    from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
    c_tensor = XRTTensor(np.zeros((M, N), dtype=np.int32), device="npu")
    gemm_fn(A, B, c_tensor)
    C = c_tensor.numpy()

    C.tofile(args.c)
    print(f"[NPU] GEMM {M}x{K}x{N} done", file=sys.stderr)


if __name__ == "__main__":
    main()
