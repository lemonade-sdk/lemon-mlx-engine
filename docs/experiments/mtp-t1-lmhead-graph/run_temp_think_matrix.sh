#!/usr/bin/env bash
# Product-mode gen matrix: temperature × thinking × (eager|MTP)
# HARD BAN: log only; no invented TPS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUTDIR="$ROOT/docs/experiments/mtp-t1-lmhead-graph"
MODEL="${MODEL:-LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit}"
CHAT="${CHAT:-$ROOT/build/chat}"
ENV_BASE="MLX_ENABLE_QUANT_FUSE=1 MLX_LOAD_MTP_HEAD=1"
PROMPT_SHORT='Write a technical overview of the Fourier Transform for engineers.'
PROMPT_THINK='In 6-8 short sentences, explain Maxwell Gauss law for electric fields (differential form) and what it means physically.'

run() {
  local name="$1"; shift
  local envx="$1"; shift
  local prompt="$1"; shift
  local log="$OUTDIR/${name}.txt"
  echo "=== RUN $name ===" | tee "$log"
  echo "date=$(date -Iseconds) tip=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)" | tee -a "$log"
  echo "args: $*" | tee -a "$log"
  # shellcheck disable=SC2086
  env $envx $ENV_BASE "$CHAT" "$MODEL" "$@" >>"$log" 2>&1 <<EOF
$prompt
quit
EOF
  echo "=== DONE $name exit=$? ===" | tee -a "$log"
  grep -E 'Generation:|Prompt:|MTP enabled|temperature|Error' "$log" | tail -15 || true
}

# E0: greedy no-think
run T_E0_temp0_nothink "" "$PROMPT_SHORT" \
  --temperature 0 --top-p 1 --max-tokens 128 --no-think --ignore-eos

# E07: product sample no-think
run T_E07_temp07_nothink "" "$PROMPT_SHORT" \
  --temperature 0.7 --top-p 0.9 --max-tokens 128 --no-think --ignore-eos

# E07T: product sample + thinking
run T_E07T_temp07_think "" "$PROMPT_THINK" \
  --temperature 0.7 --top-p 0.9 --max-tokens 512 --ignore-eos

# M0: MTP greedy
run T_M0_mtp_temp0_nothink "MTP_TIMING=1" "$PROMPT_SHORT" \
  --use-mtp --n-draft 2 --temperature 0 --top-p 1 --max-tokens 128 --no-think --ignore-eos

# M07: MTP rejection sampling
run T_M07_mtp_temp07_nothink "MTP_TIMING=1" "$PROMPT_SHORT" \
  --use-mtp --n-draft 2 --temperature 0.7 --top-p 0.9 --max-tokens 128 --no-think --ignore-eos

# M07T: MTP RS + thinking
run T_M07T_mtp_temp07_think "MTP_TIMING=1" "$PROMPT_THINK" \
  --use-mtp --n-draft 2 --temperature 0.7 --top-p 0.9 --max-tokens 512 --ignore-eos

echo '==== MATRIX SUMMARY ==='
for f in "$OUTDIR"/T_E*.txt "$OUTDIR"/T_M*.txt; do
  [[ -f "$f" ]] || continue
  echo -n "$(basename "$f"): "
  grep 'Generation:' "$f" | tail -1 || echo MISSING
done
