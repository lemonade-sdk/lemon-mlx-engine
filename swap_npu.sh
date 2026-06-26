#!/bin/bash
# Swap between community NPU (JIT) and full NPU (Chess xclbin)
# Usage: source swap_npu.sh [community|full]

MODE=${1:-community}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case $MODE in
    community)
        unset NPU_XCLBIN_PATH
        echo "[NPU] Community mode: using IRON JIT (open-source)"
        echo "[NPU] Requires: pip install mlir-aie"
        ;;
    full)
        XCLBIN="$SCRIPT_DIR/build_full/kernels/chess_gemm.xclbin"
        if [ -f "$XCLBIN" ]; then
            export NPU_XCLBIN_PATH="$XCLBIN"
            echo "[NPU] Full mode: using Chess xclbin (31 TFLOPS)"
            echo "[NPU] xclbin: $XCLBIN"
        else
            echo "[NPU] Full mode: Chess xclbin not found at $XCLBIN"
            echo "[NPU] Build the full binary first or set NPU_XCLBIN_PATH manually"
            return 1
        fi
        ;;
    *)
        echo "Usage: source swap_npu.sh [community|full]"
        return 1
        ;;
esac
