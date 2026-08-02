#!/usr/bin/env bash
# T1 long-context KV retest — serial only (HARD BAN dual-load).
# Kill bar: kv4 or kv8 must beat safe-fuse baseline by ≥5% gen t/s (within session).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUTDIR="$ROOT/docs/experiments/mtp-t1-attack"
MODEL="${MODEL:-LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit}"
CHAT="${CHAT:-$ROOT/build/chat}"
PROMPT_FILE="${PROMPT_FILE:-$OUTDIR/longctx_prompt.txt}"
COMMON="MLX_ENABLE_QUANT_FUSE=1 MLX_LOAD_MTP_HEAD=1"
STATUS="$OUTDIR/T1L_STATUS.txt"

if [[ ! -x "$CHAT" ]]; then
  echo "missing chat binary: $CHAT" | tee "$STATUS"
  exit 1
fi
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "missing prompt file: $PROMPT_FILE" | tee "$STATUS"
  exit 1
fi

{
  echo "T1L start date=$(date -Iseconds) tip=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "model=$MODEL chat=$CHAT prompt_file=$PROMPT_FILE"
  echo "kill_bar: ≥5% gen t/s vs T1L_eager_safe_fuse within-session"
} | tee "$STATUS"

run() {
  local name="$1"; shift
  local envx="$1"; shift
  local args=("$@")
  local log="$OUTDIR/${name}.txt"
  echo "=== RUN $name env=[$envx] args=[${args[*]}] ===" | tee "$log" | tee -a "$STATUS"
  echo "date=$(date -Iseconds) tip=$(git rev-parse --short HEAD)" | tee -a "$log"
  # shellcheck disable=SC2086
  # chat.cpp uses std::getline — ONE physical line = ONE user turn.
  # Collapse prompt to a single line, then quit. Do NOT feed multi-line files.
  local one_line
  one_line="$(tr '\n' ' ' <"$PROMPT_FILE" | tr -s ' ')"
  { printf '%s\n' "$one_line"; printf 'quit\n'; } | env $envx "$CHAT" "$MODEL" "${args[@]}" >>"$log" 2>&1 || {
    echo "=== FAIL $name exit=$? ===" | tee -a "$log" | tee -a "$STATUS"
    return 0
  }
  echo "=== DONE $name ===" | tee -a "$log" | tee -a "$STATUS"
  grep -E 'Generation:|Prompt:|kv|quant-fuse|Model loaded' "$log" | tail -20 | tee -a "$STATUS" || true
}

BASE_ARGS=(--temperature 0 --top-p 1 --max-tokens 256 --no-think --ignore-eos)

# Eager only — isolate T1/KV; do not credit MTP.
run T1L_eager_safe_fuse "$COMMON" "${BASE_ARGS[@]}"
run T1L_eager_safe_kv8 "$COMMON" "${BASE_ARGS[@]}" --kv-bits 8
run T1L_eager_safe_kv4 "$COMMON" "${BASE_ARGS[@]}" --kv-bits 4

{
  echo '==== T1L SUMMARY ==='
  for f in T1L_eager_safe_fuse T1L_eager_safe_kv8 T1L_eager_safe_kv4; do
    echo -n "$f: "
    grep 'Generation:' "$OUTDIR/${f}.txt" 2>/dev/null | tail -1 || echo MISSING
  done
  echo "T1L complete date=$(date -Iseconds)"
} | tee -a "$STATUS"
