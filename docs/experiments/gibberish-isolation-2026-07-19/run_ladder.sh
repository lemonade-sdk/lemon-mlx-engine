#!/usr/bin/env bash
# Isolation ladder for LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit ONLY.
# Saves every HTTP response to raw/*.json and builds RESULTS.md for manual verify.
set -euo pipefail

MODEL="LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit"
RES="$(cd "$(dirname "$0")" && pwd)"
RAW="$RES/raw"
LOGS="$RES/logs"
SERVER_BIN="/home/antmi/lemon-mlx-engine/build/server"
HOST="127.0.0.1"
PORT="8080"
BASE="http://${HOST}:${PORT}"
PIDFILE="$LOGS/server.pid"
MANIFEST="$RES/RESULTS.md"
META="$RES/meta.jsonl"

mkdir -p "$RAW" "$LOGS"
: > "$META"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/runner.log"; }

kill_server() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid=$(cat "$PIDFILE" || true)
    if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      log "Stopping server pid=$pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        ps -p "$pid" >/dev/null 2>&1 || break
        sleep 0.5
      done
      if ps -p "$pid" >/dev/null 2>&1; then
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
      fi
    fi
    rm -f "$PIDFILE"
  fi
  # fallback: free 8080 if our server still bound
  local p
  p=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)
  if [[ -n "${p:-}" ]]; then
    log "Killing leftover listener pid=$p on :$PORT"
    kill -TERM "$p" 2>/dev/null || true
    sleep 2
  fi
}

start_server() {
  local tag="$1"
  shift
  # remaining: env assignments as KEY=VAL, then -- for server args
  kill_server
  local env_args=()
  local server_args=()
  local mode=env
  for a in "$@"; do
    if [[ "$a" == "--" ]]; then mode=args; continue; fi
    if [[ "$mode" == env ]]; then env_args+=("$a"); else server_args+=("$a"); fi
  done

  local slog="$LOGS/${tag}-server.log"
  log "Starting server tag=$tag env=${env_args[*]:-(defaults)} args=${server_args[*]}"
  # shellcheck disable=SC2086
  nohup env HF_HUB_OFFLINE=1 "${env_args[@]}" \
    "$SERVER_BIN" "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --no-download \
    "${server_args[@]}" \
    >"$slog" 2>&1 &
  echo $! >"$PIDFILE"
  local pid
  pid=$(cat "$PIDFILE")
  log "server pid=$pid log=$slog"

  local i
  for i in $(seq 1 90); do
    if curl -sS --max-time 2 "$BASE/health" >/dev/null 2>&1; then
      log "health ok attempt=$i"
      curl -sS "$BASE/health" >"$RAW/${tag}-health.json" || true
      return 0
    fi
    if ! ps -p "$pid" >/dev/null 2>&1; then
      log "SERVER DIED during load"
      tail -50 "$slog" | tee -a "$LOGS/runner.log"
      return 1
    fi
    sleep 3
  done
  log "health timeout"
  return 1
}

# Append full response content into RESULTS.md for manual verification
append_result_md() {
  local step="$1"
  local config="$2"
  local json_path="$3"
  local prompt="$4"
  python3 - "$step" "$config" "$json_path" "$prompt" "$MANIFEST" <<'PY'
import json, sys, datetime
from pathlib import Path
step, config, jp, prompt, md_path = sys.argv[1:6]
p = Path(jp)
md = Path(md_path)
raw = p.read_text(encoding="utf-8", errors="replace")
try:
    d = json.loads(raw)
except Exception as e:
    block = f"""
## {step}

- **config:** `{config}`
- **file:** `{p.name}`
- **status:** JSON PARSE FAIL: {e}

```
{raw[:4000]}
```

"""
    with md.open("a", encoding="utf-8") as f:
        f.write(block)
    print("PARSE_FAIL")
    sys.exit(0)

if "error" in d:
    content = json.dumps(d, indent=2)
    finish = "error"
    usage = {}
    body = content
else:
    ch = d["choices"][0]
    finish = ch.get("finish_reason")
    usage = d.get("usage") or {}
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    body = content
    if reasoning:
        body = f"[reasoning_content]\n{reasoning}\n\n[content]\n{content}"

block = f"""
## {step}

| Field | Value |
|-------|--------|
| time (UTC) | {datetime.datetime.utcnow().isoformat()}Z |
| model | LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit |
| config | `{config}` |
| raw JSON | `raw/{p.name}` |
| finish_reason | `{finish}` |
| usage | `{usage}` |
| content_chars | {len(content) if 'content' in dir() else 'n/a'} |

### Prompt

```
{prompt}
```

### Full model response (manual verify)

```
{body}
```

---
"""
with md.open("a", encoding="utf-8") as f:
    f.write(block)
print(f"OK finish={finish} chars={len(body)}")
PY
}

chat() {
  local step="$1"
  local config="$2"
  local out_json="$3"
  local max_tokens="$4"
  local prompt="$5"
  local extra_json="${6:-}"  # optional extra fields e.g. "\"use_mtp\": true"

  local body
  body=$(python3 - "$MODEL" "$max_tokens" "$prompt" "$extra_json" <<'PY'
import json, sys
model, max_tokens, prompt, extra = sys.argv[1:5]
req = {
  "model": model,
  "temperature": 0,
  "max_tokens": int(max_tokens),
  "stream": False,
  "enable_thinking": False,
  "messages": [{"role": "user", "content": prompt}],
}
if extra.strip():
    # extra is JSON object fragment or full object string
    extra_obj = json.loads("{" + extra + "}" if not extra.strip().startswith("{") else extra)
    req.update(extra_obj)
print(json.dumps(req))
PY
)

  log "REQUEST $step max_tokens=$max_tokens -> $out_json"
  local http t
  http=$(curl -sS --max-time 900 -o "$out_json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d "$body" \
    "$BASE/v1/chat/completions") || true
  t=$(date -Iseconds)
  echo "{\"step\":\"$step\",\"http\":$http,\"time\":\"$t\",\"file\":\"$out_json\",\"config\":\"$config\"}" >>"$META"
  log "RESPONSE $step http=$http size=$(wc -c <"$out_json" || echo 0)"
  append_result_md "$step" "$config" "$out_json" "$prompt"
}

PROMPT_SHORT='In 3 short steps, state Maxwell equations in vacuum (differential form) and what each means. Be concise.'
PROMPT_LONG='Derive the wave equation for E from Maxwell equations in vacuum (mu0, eps0). Show intermediate steps clearly, then state phase velocity. Keep math clean.'

# --- RESULTS.md header ---
cat >"$MANIFEST" <<EOF
# Isolation results — manual verification pack

**Model (ONLY):** \`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit\`  
**Date:** $(date -Iseconds)  
**Device:** ROCm (see server logs)  
**Engine:** \`/home/antmi/lemon-mlx-engine/build/server\`  
**Raw JSON directory:** \`raw/\`  
**Server logs:** \`logs/\`

How to verify: open each section below; full model text is inlined. Matching machine JSON is under \`raw/<step>.json\`.

---
EOF

log "=== LADDER START model=$MODEL ==="

# ========== L0: pure OFF + SYNC + no MTP + no-think ==========
start_server L0 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  MLX_SYNC_DECODE=1 \
  -- \
  --no-think --max-tokens 2048

chat "L0-short" "PURE_OFF=1 SYNC=1 no-mtp no-think temp=0 max_tokens=512" \
  "$RAW/L0-short-maxwell.json" 512 "$PROMPT_SHORT"

chat "L0-long" "PURE_OFF=1 SYNC=1 no-mtp no-think temp=0 max_tokens=800" \
  "$RAW/L0-long-wave.json" 800 "$PROMPT_LONG"

# ========== L1: pure OFF, no SYNC ==========
start_server L1 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  -- \
  --no-think --max-tokens 2048

chat "L1-short" "PURE_OFF=1 no-SYNC no-mtp no-think temp=0 max_tokens=512" \
  "$RAW/L1-short-maxwell.json" 512 "$PROMPT_SHORT"

chat "L1-long" "PURE_OFF=1 no-SYNC no-mtp no-think temp=0 max_tokens=800" \
  "$RAW/L1-long-wave.json" 800 "$PROMPT_LONG"

# ========== L2: pure ON (default), no SYNC ==========
start_server L2 \
  -- \
  --no-think --max-tokens 2048

chat "L2-short" "PURE default ON no-SYNC no-mtp no-think temp=0 max_tokens=512" \
  "$RAW/L2-short-maxwell.json" 512 "$PROMPT_SHORT"

chat "L2-long" "PURE default ON no-SYNC no-mtp no-think temp=0 max_tokens=800" \
  "$RAW/L2-long-wave.json" 800 "$PROMPT_LONG"

# ========== L3: thinking ON, pure OFF + SYNC (safe trunk + thinking) ==========
start_server L3 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  MLX_SYNC_DECODE=1 \
  -- \
  --max-tokens 2048
# no --no-think

chat "L3-short-thinking" "PURE_OFF=1 SYNC=1 thinking DEFAULT ON no-mtp temp=0 max_tokens=512" \
  "$RAW/L3-short-thinking.json" 512 "$PROMPT_SHORT" '"enable_thinking": true'

chat "L3-long-thinking" "PURE_OFF=1 SYNC=1 thinking ON no-mtp temp=0 max_tokens=800" \
  "$RAW/L3-long-thinking.json" 800 "$PROMPT_LONG" '"enable_thinking": true'

# ========== L4: MTP ON (pure inactive), pure off env still set but MTP short-circuits ==========
start_server L4 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  MLX_SYNC_DECODE=1 \
  -- \
  --no-think --max-tokens 2048 --use-mtp

chat "L4-short-mtp" "use-mtp PURE_OFF SYNC no-think temp=0 max_tokens=512" \
  "$RAW/L4-short-mtp.json" 512 "$PROMPT_SHORT" '"use_mtp": true'

chat "L4-long-mtp" "use-mtp PURE_OFF SYNC no-think temp=0 max_tokens=800" \
  "$RAW/L4-long-mtp.json" 800 "$PROMPT_LONG" '"use_mtp": true'

# Summary table footer
python3 - "$RAW" "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys
raw, md = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for p in sorted(raw.glob("L*.json")):
    try:
        d = json.loads(p.read_text())
        if "error" in d:
            rows.append((p.name, "ERROR", str(d.get("error"))[:80], 0))
            continue
        ch = d["choices"][0]
        c = (ch.get("message") or {}).get("content") or ""
        rows.append((p.name, ch.get("finish_reason"), d.get("usage"), len(c)))
    except Exception as e:
        rows.append((p.name, "PARSE", str(e), 0))
lines = ["\n## Summary table\n", "| file | finish_reason | usage | content_chars |", "|------|---------------|-------|---------------|"]
for name, fr, us, n in rows:
    lines.append(f"| `{name}` | `{fr}` | `{us}` | {n} |")
lines.append("\n*Server left running with last ladder config (L4 MTP).*\n")
with md.open("a") as f:
    f.write("\n".join(lines) + "\n")
print("summary rows", len(rows))
PY

log "=== LADDER COMPLETE results=$MANIFEST ==="
ls -la "$RAW"
echo "RESULTS_MD=$MANIFEST"
