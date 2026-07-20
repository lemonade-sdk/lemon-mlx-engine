#!/bin/bash
# Thinking Mode Verification Test
# Tests that thinking/reasoning mode works properly with MTP
# Uses optimal parameters to prevent repetition loops

set -e

export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"

MODEL="mlx-community/Qwen3.5-4B-MTP-4bit"
CHAT_BIN="/home/antmi/lemon-mlx-engine/build/chat"
LOG_DIR="/home/antmi/lemon-mlx-engine/mtp_test_logs"
LOG_FILE="$LOG_DIR/thinking_mode_test_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "=== Thinking Mode Verification Test ==="
echo "Model: $MODEL"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"
echo ""

# Test questions that require reasoning
declare -a QUESTIONS=(
    "If a train travels 60 miles in 2 hours, what is its speed in miles per hour? Show your work step by step."
    "What are the pros and cons of renewable energy sources? Provide a balanced analysis."
    "Explain the difference between supervised and unsupervised learning with examples."
    "If x + 5 = 12, what is x? Explain your reasoning."
)

# Optimal parameters for thinking mode
N_DRAFT=4
MAX_TOKENS=512
TEMPERATURE=0.3
REPETITION_PENALTY=1.2

echo "Configuration:"
echo "  n_draft: $N_DRAFT"
echo "  max_tokens: $MAX_TOKENS"
echo "  temperature: $TEMPERATURE"
echo "  repetition_penalty: $REPETITION_PENALTY"
echo "  Total questions: ${#QUESTIONS[@]}"
echo ""

# Start the log
{
    echo "=== Thinking Mode Verification Test ==="
    echo "Model: $MODEL"
    echo "Start time: $(date)"
    echo "Config: n_draft=$N_DRAFT max_tokens=$MAX_TOKENS temp=$TEMPERATURE rep_penalty=$REPETITION_PENALTY"
    echo "======================================="
    echo ""
} | tee "$LOG_FILE"

echo "========================================="
echo "Testing Thinking Mode with MTP"
echo "========================================="
echo ""

# Build input for all questions
INPUT=""
for i in "${!QUESTIONS[@]}"; do
    question="${QUESTIONS[$i]}"
    echo "[$((i+1))/${#QUESTIONS[@]}] Question: $question"
    echo "[$((i+1))/${#QUESTIONS[@]}] Question: $question" >> "$LOG_FILE"
    INPUT+="$question"$'\n'
done

INPUT+="quit"$'\n'

# Run the chat with thinking enabled (NO --no-think flag)
echo ""
echo "Starting thinking mode test..."
echo "NOTE: Thinking is ENABLED (no --no-think flag)"
echo ""

echo "$INPUT" | timeout 300 "$CHAT_BIN" "$MODEL" \
    --use-mtp \
    --n-draft $N_DRAFT \
    --max-tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --repetition-penalty $REPETITION_PENALTY \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "========================================="
echo "Test completed at: $(date)"
echo "Log saved to: $LOG_FILE"
echo "========================================="
echo ""
echo "VERIFICATION CHECKLIST:"
echo "✅ Model generated 'Thinking Process:' or similar reasoning"
echo "✅ Reasoning is structured (numbered steps, bullet points)"
echo "✅ Reasoning is relevant to the question"
echo "✅ Final answer is provided after thinking"
echo "✅ No infinite repetition loops"
echo "✅ No garbled/encoded tokens (◎ characters)"
echo ""
echo "If all checks pass, thinking mode is WORKING PROPERLY!"
echo ""
echo "Key Parameters for Thinking Mode:"
echo "  --temperature 0.3 (not 0.0 to avoid loops)"
echo "  --repetition-penalty 1.2 (critical for long reasoning)"
echo "  --max-tokens 512 (enough space for thinking)"
echo "  --use-mtp (enables MTP speculative decoding)"
echo ""
echo "DO NOT use --no-think if you want to see the reasoning!"
