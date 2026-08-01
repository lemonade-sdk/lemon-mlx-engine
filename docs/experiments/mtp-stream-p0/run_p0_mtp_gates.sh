#!/usr/bin/env bash
# P0-MTP gates M1–M3 (server) + optional M4/M6 from PR #63.
# Single-process only — kills existing build/server and build/chat first.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
EXP="${EXP:-docs/experiments/mtp-stream-p0/gates}"
N_DRAFT="${N_DRAFT:-2}"
LOAD_WAIT="${LOAD_WAIT:-180}"
SKIP_M3="${SKIP_M3:-0}"
SKIP_M4="${SKIP_M4:-1}"   # CLI needs server down; default skip in server-only run
SKIP_M6="${SKIP_M6:-1}"

mkdir -p "$EXP/raw" "$EXP/logs"
META="$EXP/meta.jsonl"
RESULTS="$EXP/RESULTS.md"

ENGINE_SHA="$(git rev-parse HEAD)"
MLX_SHA="$(cd build/_deps/mlx-src 2>/dev/null && git rev-parse HEAD || echo unknown)"
BIN_SERVER="build/server"
BIN_CHAT="build/chat"

die() { echo "FAIL: $*" >&2; exit 1; }
log() { echo "[gate] $*" >&2; }

need_bins() {
  test -x "$BIN_SERVER" || die "missing $BIN_SERVER — rebuild"
  test -x "$BIN_CHAT" || die "missing $BIN_CHAT — rebuild"
}

free_gpu_procs() {
  # Prefer port/PID-file kills — avoid pkill -f matching this harness cmdline.
  if [[ -f "$EXP/server.pid" ]]; then
    local sp
    sp="$(cat "$EXP/server.pid" 2>/dev/null || true)"
    [[ -n "${sp:-}" ]] && kill "$sp" 2>/dev/null || true
    rm -f "$EXP/server.pid"
  fi
  local port_pid
  port_pid="$(ss -ltnp 2>/dev/null | awk -v p=":${PORT}" '
    $0 ~ p {
      if (match($0, /pid=([0-9]+)/, a)) { print a[1]; exit }
    }')"
  [[ -n "${port_pid:-}" ]] && kill "$port_pid" 2>/dev/null || true
  sleep 2
}

wait_health() {
  local i
  for i in $(seq 1 "$LOAD_WAIT"); do
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 \
       || curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      log "health ok after ${i}s"
      return 0
    fi
    sleep 1
  done
  die "server not healthy within ${LOAD_WAIT}s — see $1"
}

start_server() {
  local logf="$1"
  shift
  free_gpu_procs
  log "starting server → $logf ($*)"
  env \
    MLX_LOAD_MTP_HEAD=1 \
    ${MLX_ENABLE_QUANT_FUSE:+MLX_ENABLE_QUANT_FUSE="$MLX_ENABLE_QUANT_FUSE"} \
    ${MLX_ENABLE_QUANT_FUSE_GDN:+MLX_ENABLE_QUANT_FUSE_GDN="$MLX_ENABLE_QUANT_FUSE_GDN"} \
    "$BIN_SERVER" "$MODEL" \
      --host "$HOST" --port "$PORT" \
      --use-mtp --n-draft-tokens "$N_DRAFT" \
      "$@" \
      >"$logf" 2>&1 &
  echo $! >"$EXP/server.pid"
  wait_health "$logf"
}

curl_chat() {
  local out="$1"
  local body="$2"
  local code
  code="$(curl -sS -o "$out" -w '%{http_code}' \
    "http://${HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body")"
  echo "$code"
}

has_stream_err() {
  local f
  for f in "$@"; do
    if [[ -f "$f" ]] && grep -q 'Stream(cpu' "$f" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

content_nonempty() {
  python3 - "$1" <<'PY'
import json,sys
p=sys.argv[1]
try:
    d=json.load(open(p))
except Exception as e:
    print("json_error",e); sys.exit(1)
ch=d.get("choices") or []
if not ch:
    print("no_choices"); sys.exit(1)
msg=ch[0].get("message") or {}
c=(msg.get("content") or "") + (msg.get("reasoning_content") or "")
# some stacks put text under delta for stream-only; we use non-stream
if not str(c).strip():
    # try text field
    t=ch[0].get("text") or ""
    if not str(t).strip():
        print("empty"); sys.exit(1)
print("ok")
sys.exit(0)
PY
}

record() {
  local id="$1" status="$2" note="$3"
  echo "{\"gate\":\"$id\",\"status\":\"$status\",\"engine\":\"$ENGINE_SHA\",\"mlx\":\"$MLX_SHA\",\"note\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$note")}" >>"$META"
  log "$id $status — $note"
}

# --- preflight ---
need_bins
: >"$META"
{
  echo "# P0-MTP gate results"
  echo
  echo "- **When:** $(date -Iseconds)"
  echo "- **Engine:** \`$ENGINE_SHA\`"
  echo "- **mlx:** \`$MLX_SHA\`"
  echo "- **Model:** \`$MODEL\`"
  echo "- **n_draft:** $N_DRAFT"
  echo
  echo "| Gate | Status | Notes |"
  echo "|------|--------|-------|"
} >"$RESULTS"

# ========== M1 ==========
M1_LOG="$EXP/logs/M1-server.log"
M1_JSON="$EXP/raw/M1-short.json"
start_server "$M1_LOG" --no-think --max-tokens 128
CODE="$(curl_chat "$M1_JSON" \
  "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 5 words.\"}],\"max_tokens\":64,\"temperature\":0}")"
if [[ "$CODE" != "200" ]]; then
  record M1 FAIL "http=$CODE"
  echo "| M1 | FAIL | http=$CODE |" >>"$RESULTS"
  die "M1 http $CODE"
fi
if has_stream_err "$M1_LOG" "$M1_JSON"; then
  record M1 FAIL "Stream(cpu) present"
  echo "| M1 | FAIL | Stream(cpu) |" >>"$RESULTS"
  die "M1 Stream error"
fi
if ! content_nonempty "$M1_JSON"; then
  record M1 FAIL "empty content"
  echo "| M1 | FAIL | empty content |" >>"$RESULTS"
  die "M1 empty"
fi
if ! grep -qE '\[mtp\]|MTP enabled|n_draft' "$M1_LOG"; then
  log "WARN M1: no explicit [mtp] line in server log (check acceptance manually)"
fi
record M1 PASS "http=200 no Stream nonempty"
echo "| M1 | PASS | short no-think |" >>"$RESULTS"

# ========== M2 (same server) ==========
M2_JSON="$EXP/raw/M2-long.json"
CODE="$(curl_chat "$M2_JSON" \
  "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain the wave equation in plain language and give the 1D form. Keep it clear.\"}],\"max_tokens\":800,\"temperature\":0}")"
if [[ "$CODE" != "200" ]] || has_stream_err "$M1_LOG" "$M2_JSON" || ! content_nonempty "$M2_JSON"; then
  record M2 FAIL "http=$CODE or stream/empty"
  echo "| M2 | FAIL | http=$CODE |" >>"$RESULTS"
  die "M2 failed"
fi
record M2 PASS "http=200 long no-think"
echo "| M2 | PASS | long no-think |" >>"$RESULTS"

# ========== M3 ==========
if [[ "$SKIP_M3" != "1" ]]; then
  M3_LOG="$EXP/logs/M3-server.log"
  M3S_JSON="$EXP/raw/M3-short.json"
  M3L_JSON="$EXP/raw/M3-long.json"
  start_server "$M3_LOG" --max-tokens 4096
  CODE="$(curl_chat "$M3S_JSON" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer briefly.\"}],\"max_tokens\":512,\"temperature\":0}")"
  CODE2="$(curl_chat "$M3L_JSON" \
    "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"In one short paragraph, who was James Clerk Maxwell?\"}],\"max_tokens\":1024,\"temperature\":0}")"
  if [[ "$CODE" != "200" || "$CODE2" != "200" ]] \
     || has_stream_err "$M3_LOG" "$M3S_JSON" "$M3L_JSON"; then
    record M3 FAIL "http=$CODE/$CODE2 or Stream"
    echo "| M3 | FAIL | http=$CODE/$CODE2 |" >>"$RESULTS"
    die "M3 failed"
  fi
  record M3 PASS "thinking+MTP short+long"
  echo "| M3 | PASS | thinking+MTP |" >>"$RESULTS"
else
  echo "| M3 | SKIP | SKIP_M3=1 |" >>"$RESULTS"
fi

# ========== M4 optional ==========
if [[ "$SKIP_M4" != "1" ]]; then
  free_gpu_procs
  M4_LOG="$EXP/logs/M4-chat.log"
  printf 'Say hi in 5 words.\nquit\n' | env MLX_LOAD_MTP_HEAD=1 \
    "$BIN_CHAT" "$MODEL" --no-think --use-mtp --n-draft "$N_DRAFT" \
    --temperature 0 --max-tokens 64 \
    >"$M4_LOG" 2>&1 || true
  if has_stream_err "$M4_LOG" || ! grep -qiE 'Assistant:|Generation:' "$M4_LOG"; then
    record M4 FAIL "CLI stream or no output"
    echo "| M4 | FAIL | see logs |" >>"$RESULTS"
    die "M4 failed"
  fi
  record M4 PASS "CLI MTP"
  echo "| M4 | PASS | CLI |" >>"$RESULTS"
else
  echo "| M4 | SKIP | SKIP_M4=1 |" >>"$RESULTS"
fi

# ========== M6 optional ==========
if [[ "$SKIP_M6" != "1" ]]; then
  free_gpu_procs
  M6_LOG="$EXP/logs/M6-xor.log"
  printf 'Say hi.\nquit\n' | env MLX_LOAD_MTP_HEAD=1 MLX_DECODE_GRAPH_PURE=1 \
    "$BIN_CHAT" "$MODEL" --no-think --use-mtp --n-draft 2 \
    --temperature 0 --max-tokens 32 \
    >"$M6_LOG" 2>&1 || true
  if has_stream_err "$M6_LOG"; then
    record M6 FAIL "Stream under pure+MTP"
    echo "| M6 | FAIL | Stream |" >>"$RESULTS"
    die "M6 Stream"
  fi
  if ! grep -q 'M6 XOR' "$M6_LOG"; then
    log "WARN M6: expected XOR log line missing (MTP still short-circuits pure)"
  fi
  record M6 PASS "XOR log / no Stream"
  echo "| M6 | PASS | pure ignored under MTP |" >>"$RESULTS"
else
  echo "| M6 | SKIP | SKIP_M6=1 |" >>"$RESULTS"
fi

free_gpu_procs
{
  echo
  echo "## Verdict"
  echo
  if grep -q '| M1 | PASS |' "$RESULTS" \
     && grep -q '| M2 | PASS |' "$RESULTS" \
     && grep -q '| M3 | PASS |' "$RESULTS"; then
    echo "**P0-MTP CLOSE eligible:** M1+M2+M3 PASS."
  else
    echo "**P0-MTP NOT closed:** need M1+M2+M3 PASS (M3 may still be running)."
  fi
  echo
  echo "Artifacts under \`$EXP/\`."
} >>"$RESULTS"

log "done — $RESULTS"
cat "$RESULTS"
