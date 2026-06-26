# Implementation Plan: NPU Ternary GEMM Dispatch

## Overview
Add NPU (AI Engine) acceleration for 1-bit ternary matmuls in lemon-mlx-engine.
Replaces Python IRON JIT subprocess with direct C++ XRT + pre-compiled AIE kernel.

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/npu/kernels/ternary_gemv_aie.cpp` | CREATE | AIE kernel: ternary GEMV on NPU |
| `src/npu/npu_backend.cpp` | REWRITE | Native C++ XRT backend (replace Python) |
| `include/mlx-lm/npu/npu_backend.h` | UPDATE | Add ternary_gemv, matmul_bf16 APIs |
| `CMakeLists.txt` | UPDATE | Build AIE kernel + link XRT |
| `include/mlx-lm/common/quantized_linear.h` | UPDATE | NPU dispatch in linear_forward |
| `tests/test_bitnet_quant.cpp` | UPDATE | NPU dispatch tests |

## Tasks

### Task 1: AIE Kernel — ternary_gemv_aie.cpp
Write the AIE kernel that consumes packed U8 ternary weights and produces fp16 output.

Kernel: for each output row `oc ∈ [0, N)`:
- Unpack 4 ternary codes per byte from `packed_weights[oc/4][k]`
- Map codes {0,1,2} → ternary {-1,0,+1}
- Multiply by `activations[k]` and accumulate
- Apply `weight_scale[oc]` (or inverse)
- Write fp16 result to `output[oc]`

Use `aie::mmul` or manual vector ops for the multiply-accumulate.

### Task 2: XRT Backend — npu_backend.cpp rewrite
Replace Python subprocess with:

1. `init()` — Find NPU device, load XCLBIN, open HW context
2. `ternary_gemv()` — Create BOs, run kernel, sync, read back
3. `matmul_bf16()` — Same for bf16×bf16 GEMM (if needed)

### Task 3: Build System — CMakeLists.txt
- Find XRT libraries
- Add custom command to compile AIE kernel with `clang++ --target=aie2`
- Package into XCLBIN

### Task 4: MLX Integration — quantized_linear.h
Add NPU dispatch path for ternary (2-bit) weights:
- Check `QuantizedWeightRegistry` for 2-bit weight
- If NPU available, dispatch to `npu::ternary_gemv`
- Fall back to GPU `quantized_matmul` otherwise

### Task 5: Test
- Run existing test_bitnet_quant (34 tests, 8280 assertions)
- Test Qwen3-8B-BitNet with NPU enabled
- Verify output correctness
