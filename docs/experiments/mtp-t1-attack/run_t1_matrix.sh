#!/usr/bin/env bash
# T1 attack A/B matrix — absolute gen t/s (eager + MTP). Credit T1, not MTP magic.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUTDIR="$ROOT/docs/experiments/mtp-t1-attack"
MODEL="${MODEL:-LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit}"
PROMPT="${PROMPT:-Write a technical overview of the Fourier Transform for engineers.}"
CHAT="${CHAT:-$ROOT/build/chat}"
COMMON="MLX_ENABLE_QUANT_FUSE=1 MLX_LOAD_MTP_HEAD=1"
# GDN in_proj fuse ON for max fuse cell (temp0 only — quality thrash at 0.7 known)
FULL_FUSE="MLX_ENABLE_QUANT_FUSE=1 MLX_ENABLE_QUANT_FUSE_GDN=1 MLX_LOAD_MTP_HEAD=1"

run() {
  local name="$1"; shift
  local envx="$1"; shift
  local args=("$@")
  local log="$OUTDIR/${name}.txt"
  echo "=== RUN $name env=[$envx] args=[${args[*]}] ===" | tee "$log"
  echo "date=$(date -Iseconds) tip=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)" | tee -a "$log"
  # shellcheck disable=SC2086
  env $envx "$CHAT" "$MODEL" "${args[@]}" >>"$log" 2>&1 <<EOF
$PROMPT
quit
EOF
  echo "=== DONE $name exit=$? ===" | tee -a "$log"
  grep -E 'Generation:|Prompt:|MTP enabled|dense_kept|quant-fuse|GDN|kv' "$log" | tail -25 || true
}

BASE_ARGS=(--temperature 0 --top-p 1 --max-tokens 256 --no-think --ignore-eos)

# A1 eager baseline (SAFE fuse: attn/MLP, no GDN in_proj)
run T1_eager_safe_fuse "$COMMON" "${BASE_ARGS[@]}"

# A2 eager full fuse incl GDN in_proj
run T1_eager_full_fuse "$FULL_FUSE" "${BASE_ARGS[@]}"

# A3 eager + KV quant 8
run T1_eager_safe_kv8 "$COMMON" "${BASE_ARGS[@]}" --kv-bits 8

# A4 eager + KV quant 4
run T1_eager_safe_kv4 "$COMMON" "${BASE_ARGS[@]}" --kv-bits 4

# A5 MTP seq n2 baseline (product-like)
run T1_mtp_safe_fuse "$COMMON MTP_TIMING=1 MTP_DEBUG=1" "${BASE_ARGS[@]}" --use-mtp --n-draft 2

# A6 MTP + full fuse
run T1_mtp_full_fuse "$FULL_FUSE MTP_TIMING=1 MTP_DEBUG=1" "${BASE_ARGS[@]}" --use-mtp --n-draft 2

# A7 MTP + kv8
run T1_mtp_safe_kv8 "$COMMON MTP_TIMING=1 MTP_DEBUG=1" "${BASE_ARGS[@]}" --use-mtp --n-draft 2 --kv-bits 8

# A8 MTP + kv4
run T1_mtp_safe_kv4 "$COMMON MTP_TIMING=1 MTP_DEBUG=1" "${BASE_ARGS[@]}" --use-mtp --n-draft 2 --kv-bits 4

echo '==== MATRIX SUMMARY ==='
for f in "$OUTDIR"/T1_*.txt; do
  echo -n "$(basename "$f"): "
  grep 'Generation:' "$f" | tail -1 || echo MISSING
done
