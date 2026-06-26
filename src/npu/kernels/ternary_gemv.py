#!/usr/bin/env python3
"""NPU ternary GEMV — host-unpacked float32 dot product on IRON JIT.

The IRON JIT pipelined_loop requires all stores to output inside the loop body.
We use a read-modify-write accumulator pattern: each output element is written
at EVERY inner loop step, accumulating via read-modify-write.

Usage:
    python3 ternary_gemv.py --weights <u8.npy> --acts <f32.npy> \
        --scale <float> --invert <0|1> --out <result.npy>
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mlir-aie/.venv/lib/python3.14/site-packages"))
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_

_kernel_cache = {}


def _get_gemv_kernel(N: int, K: int):
    """Float32 GEMV kernel. Each output element is accumulated IN the output
    buffer via read-modify-write at each inner loop step."""
    key = (N, K)
    if key in _kernel_cache:
        return _kernel_cache[key]

    w_ty = np.ndarray[(N, K), np.dtype[np.float32]]
    a_ty = np.ndarray[(K,), np.dtype[np.float32]]
    o_ty = np.ndarray[(N,), np.dtype[np.float32]]

    @iron.jit
    def _gemv(weights_mat: In, acts_vec: In, result_vec: Out):
        of_w = ObjectFifo(w_ty, name="w")
        of_a = ObjectFifo(a_ty, name="a")
        of_o = ObjectFifo(o_ty, name="o")

        def core_fn(of_w, of_a, of_o):
            wb = of_w.acquire(1)
            ab = of_a.acquire(1)
            ob = of_o.acquire(1)
            # Initialize all outputs to 0
            for oc in range_(N):
                ob[oc] = 0.0
            # Accumulate into output at each k step
            for oc in range_(N):
                for k in range_(K):
                    ob[oc] = ob[oc] + wb[oc, k] * ab[k]
                # ob[oc] has the final value after inner loop completes
            of_o.release(1)
            of_a.release(1)
            of_w.release(1)

        w = Worker(core_fn, [of_w.cons(), of_a.cons(), of_o.prod()])
        rt = Runtime()
        with rt.sequence(w_ty, a_ty, o_ty) as (wp, ap, op):
            rt.start(w)
            rt.fill(of_w.prod(), wp)
            rt.fill(of_a.prod(), ap)
            rt.drain(of_o.cons(), op, wait=True)
        return Program(iron.get_current_device(), rt).resolve_program()

    _kernel_cache[key] = _gemv
    return _gemv


def _unpack_weights(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    """Unpack U8 ternary codes to float32 {-1, 0, +1}."""
    packed_rows = packed.shape[0]
    result = np.zeros((N, K), dtype=np.float32)
    for oc in range(N):
        row = oc // 4
        lane = oc % 4
        shift = lane * 2
        for k in range(K):
            code = (packed[row, k] >> shift) & 0x03
            result[oc, k] = float(code - 1)
    return result


def run(
    packed_weights: np.ndarray,
    activations: np.ndarray,
    weight_scale: float = 1.0,
    invert_scale: bool = False,
    kernel_obj: str = "",
) -> np.ndarray:
    """Run ternary GEMV: host-unpack then NPU dot product."""
    N = packed_weights.shape[0] * 4
    K = packed_weights.shape[1]

    # Host-unpack
    weights_f32 = _unpack_weights(packed_weights, N, K)

    # NPU dot product
    kernel = _get_gemv_kernel(N, K)
    w_npu = iron.tensor(weights_f32, device="npu")
    a_npu = iron.tensor(activations.astype(np.float32), device="npu")
    o_npu = iron.zeros(N, dtype=np.float32, device="npu")
    kernel(w_npu, a_npu, o_npu)

    # Apply scale
    scale = (1.0 / weight_scale) if invert_scale else weight_scale
    return o_npu.numpy() * scale


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--acts", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--invert", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    w = np.load(args.weights)
    a = np.load(args.acts)
    result = run(w, a, args.scale, bool(args.invert))
    np.save(args.out, result)
