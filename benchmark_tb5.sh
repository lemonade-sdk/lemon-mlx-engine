#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  TB5 + R9700 eGPU Benchmark Script
#  Tests HIP graph configurations vs. no-graphs baseline
#  Target: Thunderbolt 5 connected Radeon AI PRO R9700 (gfx1201)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# ── Environment ──────────────────────────────────────────────────────────────
export ROCm_DIR=/tmp/rocm_sdk_core
source /tmp/rocm_venv/bin/activate
export LD_LIBRARY_PATH=$ROCm_DIR/lib:$LD_LIBRARY_PATH

# Ensure gfx1201 is used (RDNA4 discrete)
unset HSA_OVERRIDE_GFX_VERSION

CHAT=/home/bcloud/lemon-mlx-engine/build/chat

# ── Test parameters ──────────────────────────────────────────────────────────
PROMPT="Explain the concept of cache coherence in modern multi-core processors in 3-4 paragraphs."
MAX_TOKENS=200
TEMP=0.0

# ── Models under test ─────────────────────────────────────────────────────────
declare -A MODELS
MODELS["Llama-1B"]="/home/bcloud/models/llama-1b"
MODELS["Qwen3-1.7B-MXFP4"]="/home/bcloud/models/qwen3-1.7b-mxfp4"
MODELS["BitNet-2B"]="/home/bcloud/models/bitnet-2b"
MODELS["Qwen3-4B-4bit"]="mlx-community/Qwen3-4B-4bit"

# ── Graph configuration variants ──────────────────────────────────────────────
declare -A GRAPH_LABELS
GRAPH_LABELS["no_graphs"]="MLX_USE_HIP_GRAPHS=0 (no graphs)"
GRAPH_LABELS["prefill_only"]="MLX_USE_HIP_GRAPHS=1 MLX_GRAPH_DECODE=0 (graphs prefill, no decode)"
GRAPH_LABELS["full"]="Default (graphs full)"
GRAPH_LABELS["replay"]="MLX_GRAPH_REPLAY=1 (build-once replay)"

# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

log()  { echo -e "\n$*"; }
warn() { echo "⚠  $*" >&2; }
die()  { echo "✖ FATAL: $*" >&2; exit 1; }

# Check required binary
[ -x "$CHAT" ] || die "chat binary not found at $CHAT"

# Check rocm-smi availability
if ! command -v rocm-smi &>/dev/null; then
    warn "rocm-smi not in PATH — GPU utilisation will not be collected"
fi

# Find GPU bus ID for rocm-smi (expects one discrete R9700 on TB5)
get_gpu_bus() {
    rocm-smi --showbus 2>/dev/null | grep -v 'Bus' | awk '{print $1}' | head -1 || echo ""
}

# Parse tokens/second from chat output
# Expected format: "Prompt tokens: 42  (X.XX tokens/s)"
#                 "Generated: 200 tokens  (Y.YY tokens/s)"
parse_prompt_tps()   { echo "$1" | grep -oP 'Prompt tokens:.*?\(\K[0-9.]+'; }
parse_gen_tps()      { echo "$1" | grep -oP 'Generated:.*?\(\K[0-9.]+'; }
parse_peak_vram_mb() { echo "$1" | grep -oP 'Peak GPU.*?(\d+) MB' | grep -oP '\d+'; }

# Collect GPU memory and utilisation in background, return PID
monitor_gpu() {
    local outfile="$1"
    local gpu_bus
    gpu_bus=$(get_gpu_bus)
    > "$outfile"
    while kill -0 "$MON_PID" 2>/dev/null; do
        if [ -n "$gpu_bus" ] && command -v rocm-smi &>/dev/null; then
            local vram_used vram_total util
            vram_used=$(rocm-smi --showbus "$gpu_bus" --showmeminfo vram --json 2>/dev/null \
                            | grep -oP "\"GPU.*?\"vram_used\":\s*\K[0-9]+" | head -1 || echo "0")
            vram_total=$(rocm-smi --showbus "$gpu_bus" --showmeminfo vram --json 2>/dev/null \
                            | grep -oP "\"GPU.*?\"vram_total\":\s*\K[0-9]+" | head -1 || echo "0")
            util=$(rocm-smi --showbus "$gpu_bus" --showutilization 2>/dev/null \
                      | grep -v 'GPU\|---\|util' | awk '{print $2}' | grep '%' | head -1 || echo "0%")
            echo "$(date +%s.%N),${vram_used},${vram_total},${util}" >> "$outfile"
        fi
        sleep 0.2
    done
}

# Run one benchmark config and collect all metrics
run_benchmark() {
    local label="$1"
    local model_path="$2"
    local model_name="$3"
    shift 3
    local env_vars=("$@")

    local tmp_out
    tmp_out=$(mktemp)
    local mon_out
    mon_out=$(mktemp)

    echo "──────────────────────────────────────────────────────────────────────────"
    echo "▶ $label"
    echo "  Model : $model_name"
    echo "  Prompt: ${PROMPT:0:60}..."
    echo ""

    # Build env string for logging
    local env_str=""
    for v in "${env_vars[@]}"; do
        env_str+=" $v"
    done
    [ -n "$env_str" ] && echo "  Env   :$env_str"

    # Start GPU monitoring in background
    monitor_gpu "$mon_out" &
    local MON_PID=$!

    # Run inference
    local raw_output
    local start_ts end_ts elapsed
    start_ts=$(date +%s.%N)
    raw_output=$(echo "$PROMPT" | \
                     env "${env_vars[@]}" \
                     timeout 300 "$CHAT" "$model_path" \
                         --max-tokens $MAX_TOKENS \
                         --temperature $TEMP 2>&1) || {
        warn "Command failed or timed out for '$label'"
        echo "$raw_output"
    }
    end_ts=$(date +%s.%N)
    elapsed=$(echo "$end_ts - $start_ts" | bc)

    # Stop monitoring
    kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null || true

    # ── Extract metrics ──────────────────────────────────────────────────────
    local prompt_tps gen_tps peak_vram peak_util avg_util

    prompt_tps=$(parse_prompt_tps "$raw_output")
    gen_tps=$(parse_gen_tps "$raw_output")

    # Peak VRAM from chat output if present, else from monitoring log
    peak_vram=$(parse_peak_vram_mb "$raw_output")
    if [ -z "$peak_vram" ]; then
        peak_vram=$(awk -F, '
            BEGIN { max=0 }
            $2 ~ /^[0-9]+$/ && $2>max { max=$2 }
            END { print max }' "$mon_out" 2>/dev/null || echo "N/A")
    fi

    # Average GPU utilisation from monitoring log
    avg_util=$(awk -F, '
        BEGIN { sum=0; n=0 }
        $4 ~ /[0-9]+%/ {
            sub(/%/,"",$4)
            sum+=$4; n++
        }
        END { if(n>0) printf "%.1f%%", sum/n; else print "N/A" }' "$mon_out" 2>/dev/null)

    # ── Print results ───────────────────────────────────────────────────────
    printf "  %-22s : %s\n" "Prompt tokens/s"   "${prompt_tps:-N/A}"
    printf "  %-22s : %s\n" "Generation tokens/s" "${gen_tps:-N/A}"
    printf "  %-22s : %s MB\n" "Peak VRAM"        "${peak_vram:-N/A}"
    printf "  %-22s : %s\n"   "Avg GPU util"     "${avg_util:-N/A}"
    printf "  %-22s : %s s\n"  "Wall time"       "${elapsed:-N/A}"
    echo ""

    # ── Write CSV row ───────────────────────────────────────────────────────
    echo "\"$model_name\",\"$label\",\"${prompt_tps:-NA}\",\"${gen_tps:-NA}\",\"${peak_vram:-NA}\",\"${avg_util:-NA}\",\"${elapsed:-NA}\"" >> "$CSV"

    # Append monitoring log
    if [ -s "$mon_out" ]; then
        local gpu_csv="${CSV%.csv}.gpu_stats.csv"
        tail -n +2 "$mon_out" >> "$gpu_csv"
    fi

    rm -f "$tmp_out" "$mon_out"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════════════

CSV="${0%.sh}_results.csv"
GPU_CSV="${CSV%.csv}.gpu_stats.csv"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║   TB5 + R9700 eGPU Benchmark — HIP Graph Configuration Comparison       ║"
echo "║   Prompt: ${PROMPT:0:60}...                    ║"
echo "║   Max tokens: $MAX_TOKENS  |  Temperature: $TEMP                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Init CSV
echo "model,config,prompt_tps,gen_tps,peak_vram_mb,avg_gpu_util,wall_time_s" > "$CSV"
echo "timestamp,vram_used_kb,vram_total_kb,gpu_util" > "$GPU_CSV"

# ═══════════════════════════════════════════════════════════════════════════════
#  Main benchmark loop
# ═══════════════════════════════════════════════════════════════════════════════

for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"

    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "  MODEL: $model_name"
    echo "════════════════════════════════════════════════════════════════════════════"

    # ── 1. No graphs ──────────────────────────────────────────────────────────
    run_benchmark \
        "${GRAPH_LABELS[no_graphs]}" \
        "$model_path" "$model_name" \
        "MLX_USE_HIP_GRAPHS=0"

    # ── 2. Prefill-only (graphs for prefill, no decode-mode) ──────────────────
    run_benchmark \
        "${GRAPH_LABELS[prefill_only]}" \
        "$model_path" "$model_name" \
        "MLX_USE_HIP_GRAPHS=1" "MLX_GRAPH_DECODE=0"

    # ── 3. Default (graphs full) — no extra env vars needed ──────────────────
    run_benchmark \
        "${GRAPH_LABELS[full]}" \
        "$model_path" "$model_name"

    # ── 4. Build-once replay ─────────────────────────────────────────────────
    run_benchmark \
        "${GRAPH_LABELS[replay]}" \
        "$model_path" "$model_name" \
        "MLX_USE_HIP_GRAPHS=1" "MLX_GRAPH_REPLAY=1"

    # Small pause to let GPU cool between model switches
    sleep 2
done

# ═══════════════════════════════════════════════════════════════════════════════
#  PCIe Bandwidth Analysis — BitNet-2B (memory-bandwidth-bound highlight)
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  PCIe BANDWIDTH ANALYSIS — BitNet-2B (memory-bandwidth-bound)"
echo "════════════════════════════════════════════════════════════════════════════"

log "BitNet-2B is chosen because its 1.58-bit quantized compute is extremely"
log "lightweight — performance is almost entirely limited by VRAM bandwidth."
log "On a TB5 eGPU link the PCIe overhead is maximised, so no-graph vs."
log "graph comparison directly quantifies the PCIe benefit."
echo ""

# Quick side-by-side comparison for BitNet-2B only
for env_desc in \
    "No graphs"         "MLX_USE_HIP_GRAPHS=0" \
    "Graphs (default)"  ""; do

    local label_tpl
    local env_arg
    if [ "$env_desc" = "No graphs" ]; then
        label_tpl="MLX_USE_HIP_GRAPHS=0"
        env_arg="MLX_USE_HIP_GRAPHS=0"
    else
        label_tpl="MLX_USE_HIP_GRAPHS=1 (graphs)"
        env_arg=""
    fi

    local tmp_out mon_out
    tmp_out=$(mktemp); mon_out=$(mktemp)

    monitor_gpu "$mon_out" &
    local MON_PID=$!

    local start end elapsed raw
    start=$(date +%s.%N)
    raw=$(echo "$PROMPT" | env $env_arg timeout 120 "$CHAT" "${MODELS[BitNet-2B]}" \
                --max-tokens $MAX_TOKENS --temperature $TEMP 2>&1) || true
    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null || true

    local tps
    tps=$(parse_gen_tps "$raw")
    echo "  [$label_tpl]  Generation: ${tps:-N/A} tokens/s  (${elapsed}s wall)"
    rm -f "$tmp_out" "$mon_out"
done

# ═══════════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "  Results saved to: $CSV"
echo "  GPU stats saved to: $GPU_CSV"
echo ""
echo "  CSV columns:"
echo "    model, config, prompt_tps, gen_tps, peak_vram_mb, avg_gpu_util, wall_time_s"
echo ""
echo "  GPU stats columns:"
echo "    timestamp (unix), vram_used_kb, vram_total_kb, gpu_util_pct"
echo ""
echo "Benchmark complete."
