#!/usr/bin/env bash
# P2 N-sweep / multi-run host-wall for retained AQL (NOT gen t/s).
# Uses out-of-process p1_load_hsaco binary (warpfront redline-dispatch example).
#
# Env:
#   REDLINE_P1_HSACO   required (.co/.hsaco)
#   REDLINE_P1_SYMBOL  default floor_k.kd
#   REDLINE_P2_BIN     path to p1_load_hsaco (default: /tmp/redline-warpfront-target/release/examples/p1_load_hsaco)
#   REDLINE_P2_NS      space-separated N list (default: "2 4 8 16 32 64")
#   REDLINE_P2_RUNS    independent process runs per (N,policy) (default: 3)
#   REDLINE_P2_ITERS   timed replays per run (default: 40)
#   REDLINE_P2_WARMUP  warmup replays (default: 10)
#   REDLINE_P2_POLICIES space-separated (default: "BoundarySerialized SystemEveryDispatch")
#
# Output: machine-readable lines P2_ROW n=.. policy=.. run=.. host_median_us=..
# Summary: P2_SUMMARY ...
set -euo pipefail

BIN="${REDLINE_P2_BIN:-/tmp/redline-warpfront-target/release/examples/p1_load_hsaco}"
HSACO="${REDLINE_P1_HSACO:?REDLINE_P1_HSACO required}"
SYMBOL="${REDLINE_P1_SYMBOL:-floor_k.kd}"
NS="${REDLINE_P2_NS:-2 4 8 16 32 64}"
RUNS="${REDLINE_P2_RUNS:-3}"
ITERS="${REDLINE_P2_ITERS:-40}"
WARMUP="${REDLINE_P2_WARMUP:-10}"
POLICIES="${REDLINE_P2_POLICIES:-BoundarySerialized SystemEveryDispatch}"

if [[ ! -x "$BIN" ]]; then
  echo "P2_FAIL missing binary: $BIN" >&2
  exit 2
fi
if [[ ! -f "$HSACO" ]]; then
  echo "P2_FAIL missing HSACO: $HSACO" >&2
  exit 2
fi

echo "[p2] bin=$BIN"
echo "[p2] hsaco=$HSACO symbol=$SYMBOL"
echo "[p2] N=[$NS] policies=[$POLICIES] runs=$RUNS iters=$ITERS warmup=$WARMUP"
echo "[p2] NOTE: host_median_us only — NOT model gen t/s"

# Collect rows: n policy us
declare -a ALL_ROWS=()

for pol in $POLICIES; do
  for n in $NS; do
    for ((r=1; r<=RUNS; r++)); do
      # Independent process = multi-run "seed" (no-op kernel; process isolation variance)
      out="$(
        REDLINE_P1_HSACO="$HSACO" \
        REDLINE_P1_SYMBOL="$SYMBOL" \
        REDLINE_P1_N="$n" \
        REDLINE_P1_POLICY="$pol" \
        REDLINE_P1_ITERS="$ITERS" \
        REDLINE_P1_WARMUP="$WARMUP" \
        "$BIN" 2>/dev/null
      )" || {
        echo "P2_ROW n=$n policy=$pol run=$r status=FAIL"
        continue
      }
      us="$(echo "$out" | sed -n 's/.*host_median_us=\([0-9.]*\).*/\1/p' | head -1)"
      if [[ -z "$us" ]]; then
        echo "P2_ROW n=$n policy=$pol run=$r status=PARSE_FAIL out=$out"
        continue
      fi
      echo "P2_ROW n=$n policy=$pol run=$r host_median_us=$us"
      ALL_ROWS+=("$n|$pol|$us")
    done
  done
done

# Median of run medians per (n,policy)
python3 - <<'PY' "${ALL_ROWS[@]}"
import sys
from collections import defaultdict

rows = sys.argv[1:]
g = defaultdict(list)
for r in rows:
    n, pol, us = r.split("|")
    g[(int(n), pol)].append(float(us))

print("P2_TABLE n policy runs med_of_med_us min_us max_us us_per_dispatch")
keys = sorted(g.keys(), key=lambda k: (k[1], k[0]))
# For speedup vs SystemEveryDispatch at same n
sys_by_n = {}
for (n, pol), vals in g.items():
    if pol == "SystemEveryDispatch":
        vals_s = sorted(vals)
        m = vals_s[len(vals_s)//2]
        sys_by_n[n] = m

for (n, pol) in keys:
    vals = sorted(g[(n, pol)])
    m = vals[len(vals)//2]
    mn, mx = vals[0], vals[-1]
    per = m / n if n else 0.0
    line = f"P2_SUMMARY n={n} policy={pol} runs={len(vals)} med_of_med_us={m:.3f} min_us={mn:.3f} max_us={mx:.3f} us_per_dispatch={per:.4f}"
    if pol == "BoundarySerialized" and n in sys_by_n and m > 0:
        line += f" vs_system_every={sys_by_n[n]/m:.3f}x"
    print(line)

print("P2_OK nsweep complete (host µs only; not gen t/s)")
PY
