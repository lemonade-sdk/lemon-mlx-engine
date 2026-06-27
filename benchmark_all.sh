#!/bin/bash
# Comprehensive benchmark across all fixed models on Strix Halo (gfx1151)
set -e

export ROCm_DIR=/tmp/rocm_sdk_core
source /tmp/rocm_venv/bin/activate
export LD_LIBRARY_PATH=$ROCm_DIR/lib:$LD_LIBRARY_PATH

CHAT=/home/bcloud/lemon-mlx-engine/build/chat
MAX_TOKENS=100
PROMPT="What is the capital of France? Explain in one sentence."

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║           BENCHMARK: lemon-mlx-engine on Strix Halo (gfx1151)           ║"
echo "║           Commit 26aad7e — All fixes applied                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Prompt: \"$PROMPT\""
echo "Max tokens: $MAX_TOKENS, Temperature: 0.0 (greedy)"
echo ""

benchmark() {
    local name="$1"
    local model_path="$2"
    shift 2
    local extra_args="$@"

    echo "──────────────────────────────────────────────────────────────────────────"
    echo "▶ $name"
    echo "  Path: $model_path"
    [ -n "$extra_args" ] && echo "  Args: $extra_args"
    echo ""

    local output
    output=$(echo "$PROMPT" | timeout 120 $CHAT "$model_path" --max-tokens $MAX_TOKENS --temperature 0.0 $extra_args 2>&1) || true

    echo "$output" | grep -E "(Loading model|bound HIP|Model loaded|Prompt:|Generation:|Assistant:|Error|error|Fatal|Segmentation|Unsupported)" | head -10
    echo ""
}

# 1. BASELINE: Llama-3.2-1B-Instruct-4bit
benchmark "Llama-3.2-1B-Instruct-4bit (baseline)" /home/bcloud/models/llama-1b

# 2. BitNet b1.58-2B-4T (1.58-bit ternary)
benchmark "BitNet b1.58-2B-4T (1.58-bit ternary)" /home/bcloud/models/bitnet-2b

# 3. Bonsai 1.7B (1-bit affine)
benchmark "Bonsai 1.7B (1-bit)" /home/bcloud/models/bonsai-1.7b

# 4. Bonsai 4B (1-bit affine)
benchmark "Bonsai 4B (1-bit)" /home/bcloud/models/bonsai-4b

# 5. Bonsai 8B (1-bit affine) — needs more VRAM
benchmark "Bonsai 8B (1-bit)" /home/bcloud/models/bonsai-8b

# 6. Qwen3-1.7B MXFP4 (issue #10 fix)
benchmark "Qwen3-1.7B-MLX-MXFP4 (MXFP4 quant)" /home/bcloud/models/qwen3-1.7b-mxfp4

# 7. OpenELM-3B (issue #7 segfault fix)
benchmark "OpenELM-3B (issue #7 segfault fix)" /home/bcloud/models/openelm-3b --raw

# 8. Granite-4.0-H-Tiny (issue #6 crash fix)
benchmark "Granite-4.0-H-Tiny (issue #6 crash fix)" /home/bcloud/models/granite-4.0-h-tiny --raw

# 9. Lille-130M (issue #9 dequant fix)
benchmark "Lille-130M (issue #9 dequant fix)" /home/bcloud/models/lille-130m --raw

# 10. Falcon-E-3B (1.58-bit, inverse-scale BitLinear)
benchmark "Falcon-E-3B (1.58-bit, inverse-scale BitLinear)" /home/bcloud/models/falcon-e-3b

echo "════════════════════════════════════════════════════════════════════════════"
echo "Benchmark complete."
