#!/bin/bash
# Circular MTP Inference Test
# Tests Multi-Token Prediction with continuous generation load
# Uses mlx-community/Qwen3.5-4B-MTP-4bit model

set -e

export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

MODEL="mlx-community/Qwen3.5-4B-MTP-4bit"
CHAT_BIN="/home/antmi/lemon-mlx-engine/build/chat"
LOG_DIR="/home/antmi/lemon-mlx-engine/mtp_test_logs"
LOG_FILE="$LOG_DIR/mtp_circular_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "=== MTP Circular Inference Test ==="
echo "Model: $MODEL"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo ""

# Test prompts that exercise different aspects of generation
PROMPTS=(
    "What is 2+2? Reply with just the number."
    "Name the 3 primary colors. Be brief."
    "Write a haiku about coding."
    "What is the capital of France? One word answer."
    "Explain what MTP means in one sentence."
    "Count from 1 to 10, separated by commas."
    "What are the first 5 prime numbers?"
    "Write a Python function to add two numbers. Keep it short."
    "What is the meaning of life? Be philosophical but brief."
    "Describe the color blue in one sentence."
)

N_DRAFT=4
MAX_TOKENS=256
TEMPERATURE=0.7
LOOPS=3

echo "Configuration:"
echo "  n_draft: $N_DRAFT"
echo "  max_tokens: $MAX_TOKENS"
echo "  temperature: $TEMPERATURE"
echo "  loops: $LOOPS"
echo "  prompts per loop: ${#PROMPTS[@]}"
echo "  total generations: $((LOOPS * ${#PROMPTS[@]}))"
echo ""

# Start the log
{
    echo "=== MTP Circular Inference Test ==="
    echo "Model: $MODEL"
    echo "Start time: $(date)"
    echo "Config: n_draft=$N_DRAFT max_tokens=$MAX_TOKENS temp=$TEMPERATURE loops=$LOOPS"
    echo "======================================="
    echo ""
} | tee "$LOG_FILE"

# Build the input for all loops
INPUT=""
for loop in $(seq 1 $LOOPS); do
    echo ""
    echo "========================================="
    echo "LOOP $loop of $LOOPS"
    echo "========================================="
    echo ""
    {
        echo ""
        echo "=== LOOP $loop of $LOOPS ==="
        echo ""
    } >> "$LOG_FILE"

    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        echo "[$loop/$LOOPS] Prompt $((i+1))/${#PROMPTS[@]}: $prompt"
        echo "[$loop/$LOOPS] Prompt: $prompt" >> "$LOG_FILE"
        INPUT+="$prompt"$'\n'
    done
done

INPUT+="quit"$'\n'

# Run the chat with all prompts piped in
echo ""
echo "Starting MTP inference run..."
echo ""

echo "$INPUT" | timeout 600 "$CHAT_BIN" "$MODEL" \
    --use-mtp \
    --n-draft $N_DRAFT \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --no-think \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "======================================="
echo "Test completed at: $(date)"
echo "Log saved to: $LOG_FILE"
echo "======================================="
echo ""
echo "End time: $(date)"
echo "======================================="
