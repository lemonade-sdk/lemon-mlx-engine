#!/usr/bin/env bash
# Systematic C1 two-stage config research: many (r, power, K) cells.
# For each cell:
#   1) CHECK run max_tokens=64 → mismatch rate
#   2) PERF run max_tokens=128 (no CHECK) → gen t/s
# Always runs a ctrl once. Logs + CSV under sweep_out/.
#
# Usage:
#   ./docs/experiments/mtp-t1-lmhead-graph/run_c1_config_sweep.sh
#   SWEEP_QUICK=1 ./...   # smaller grid
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUTDIR="$ROOT/docs/experiments/mtp-t1-lmhead-graph/sweep_out"
mkdir -p "$OUTDIR"
MODEL="${MODEL:-LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit}"
CHAT="${CHAT:-$ROOT/build/chat}"
PROMPT="${PROMPT:-Write a technical overview of the Fourier Transform for engineers.}"
CSV="$OUTDIR/SWEEP_$(date +%Y%m%d_%H%M%S).csv"
echo "cell,r,power,K,check_steps,mismatch,mismatch_rate_pct,tps,delta_pct_vs_ctrl,quality_ok,log_check,log_perf" > "$CSV"

if [[ "${SWEEP_QUICK:-0}" == "1" ]]; then
  RS=(64 128)
  POWERS=(0 1)
  KS=(4096 8192)
else
  # Broad research grid — expand anytime
  RS=(64 96 128 160 192)
  POWERS=(0 1 2)
  KS=(4096 8192 16384)
fi

run_chat() {
  local log="$1"; shift
  local max="$1"; shift
  local envx="$1"; shift
  # shellcheck disable=SC2086
  env MLX_ENABLE_QUANT_FUSE=1 MLX_LOAD_MTP_HEAD=1 $envx \
    "$CHAT" "$MODEL" --temperature 0 --top-p 1 --max-tokens "$max" --no-think --ignore-eos \
    >"$log" 2>&1 <<EOF
$PROMPT
quit
EOF
}

# --- control ---
CTRL_LOG="$OUTDIR/ctrl.txt"
echo "=== CTRL ===" | tee "$OUTDIR/STATUS.txt"
run_chat "$CTRL_LOG" 128 ""
CTRL_TPS=$(grep 'Generation:' "$CTRL_LOG" | tail -1 | sed -E 's/.* ([0-9.]+) tokens\/s.*/\1/')
echo "ctrl_tps=$CTRL_TPS" | tee -a "$OUTDIR/STATUS.txt"
echo "ctrl,,,,0,0,0,$CTRL_TPS,0,1,$CTRL_LOG," >> "$CSV"

quality_ok() {
  # crude: reject obvious garble loops
  local log="$1"
  if grep -qE 'TheTheThe|Overview Overview Overview|Discretion Signal Analysis the Discretion' "$log"; then
    echo 0; return
  fi
  echo 1
}

for r in "${RS[@]}"; do
  for p in "${POWERS[@]}"; do
    for K in "${KS[@]}"; do
      cell="r${r}_p${p}_K${K}"
      echo "=== CELL $cell ===" | tee -a "$OUTDIR/STATUS.txt"
      envx="MLX_LM_HEAD_TWOSTAGE=1 MLX_LM_HEAD_STAGE1_R=$r MLX_LM_HEAD_STAGE1_POWER=$p MLX_LM_HEAD_STAGE1_K=$K"

      logc="$OUTDIR/${cell}_check.txt"
      run_chat "$logc" 64 "$envx MLX_LM_HEAD_TWOSTAGE_CHECK=1" || true
      # parse last CHECK line
      last=$(grep 'CHECK v2 step=' "$logc" | tail -1 || true)
      steps=0; mism=0; rate=100
      if [[ -n "$last" ]]; then
        steps=$(echo "$last" | sed -E 's/.*step=([0-9]+).*/\1/')
        mism=$(echo "$last" | sed -E 's/.*mismatch_total=([0-9]+).*/\1/')
        rate=$(echo "$last" | sed -E 's/.*rate=([0-9.]+)%.*/\1/')
      fi
      echo "  check steps=$steps mism=$mism rate=$rate%" | tee -a "$OUTDIR/STATUS.txt"

      logp="$OUTDIR/${cell}_perf.txt"
      run_chat "$logp" 128 "$envx" || true
      tps=$(grep 'Generation:' "$logp" | tail -1 | sed -E 's/.* ([0-9.]+) tokens\/s.*/\1/' || echo "")
      qok=$(quality_ok "$logp")
      delta=""
      if [[ -n "$tps" && -n "$CTRL_TPS" ]]; then
        delta=$(python3 -c "print(round(100*($tps-$CTRL_TPS)/$CTRL_TPS, 3))")
      fi
      echo "  tps=$tps delta=$delta% quality_ok=$qok" | tee -a "$OUTDIR/STATUS.txt"
      echo "$cell,$r,$p,$K,$steps,$mism,$rate,$tps,$delta,$qok,$logc,$logp" >> "$CSV"
    done
  done
done

echo "=== SWEEP DONE csv=$CSV ===" | tee -a "$OUTDIR/STATUS.txt"
# Top hits: quality_ok + mismatch_rate<=3 + best delta
python3 - <<PY
import csv
from pathlib import Path
csv_path = Path("$CSV")
rows = list(csv.DictReader(csv_path.open()))
print("\n=== ALL CELLS (sorted by delta desc) ===")
def fnum(x, d=0.0):
    try: return float(x)
    except: return d
scored = []
for r in rows:
    if r["cell"]=="ctrl": continue
    scored.append(r)
scored.sort(key=lambda r: fnum(r.get("delta_pct_vs_ctrl")), reverse=True)
for r in scored:
    print(f"{r['cell']:20} r={r['r']:3} p={r['power']} K={r['K']:5} "
          f"mism%={r['mismatch_rate_pct']:>6} tps={r['tps']:>7} "
          f"d%={r['delta_pct_vs_ctrl']:>7} q={r['quality_ok']}")
print("\n=== CANDIDATES: quality_ok=1 AND mism%<=3.0 AND delta>0 ===")
cands=[r for r in scored if r.get("quality_ok")=="1" and fnum(r.get("mismatch_rate_pct"),99)<=3.0 and fnum(r.get("delta_pct_vs_ctrl"))>0]
for r in cands[:10]:
    print(r["cell"], "delta", r["delta_pct_vs_ctrl"], "mism%", r["mismatch_rate_pct"])
if not cands:
    print("(none — relax bars or expand grid)")
print("\n=== CANDIDATES relaxed: quality_ok=1 AND delta>0 ===")
cands2=[r for r in scored if r.get("quality_ok")=="1" and fnum(r.get("delta_pct_vs_ctrl"))>0]
for r in cands2[:10]:
    print(r["cell"], "delta", r["delta_pct_vs_ctrl"], "mism%", r["mismatch_rate_pct"])
PY
