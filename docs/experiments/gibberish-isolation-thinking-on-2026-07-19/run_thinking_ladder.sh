#!/usr/bin/env bash
# Isolation with THINKING ON — LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit ONLY.
# Branch label passed as arg1: feat | main
set -euo pipefail

BRANCH_LABEL="${1:?usage: run_thinking_ladder.sh feat|main SERVER_BIN [PORT]}"
SERVER_BIN="${2:?path to server binary}"
PORT="${3:-8080}"
MODEL="LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit"
RES="$(cd "$(dirname "$0")" && pwd)/${BRANCH_LABEL}"
RAW="$RES/raw"
LOGS="$RES/logs"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"
PIDFILE="$LOGS/server.pid"
MANIFEST="$RES/RESULTS.md"
META="$RES/meta.jsonl"

mkdir -p "$RAW" "$LOGS"
: > "$META"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOGS/runner.log"; }

kill_server() {
  if [[ -f "$PIDFILE" ]]; then
    local pid; pid=$(cat "$PIDFILE" || true)
    if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
      log "Stopping server pid=$pid"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 30); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 0.5; done
      if ps -p "$pid" >/dev/null 2>&1; then kill -KILL "$pid" 2>/dev/null || true; sleep 1; fi
    fi
    rm -f "$PIDFILE"
  fi
  local p
  p=$(ss -tlnp 2>/dev/null | grep ":${PORT}" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)
  if [[ -n "${p:-}" ]]; then
    log "Killing leftover :$PORT pid=$p"
    kill -TERM "$p" 2>/dev/null || true
    sleep 2
  fi
}

start_server() {
  local tag="$1"; shift
  kill_server
  local env_args=() server_args=() mode=env
  for a in "$@"; do
    if [[ "$a" == "--" ]]; then mode=args; continue; fi
    if [[ "$mode" == env ]]; then env_args+=("$a"); else server_args+=("$a"); fi
  done
  local slog="$LOGS/${tag}-server.log"
  log "START tag=$tag branch=$BRANCH_LABEL bin=$SERVER_BIN env=${env_args[*]:-(defaults)} args=${server_args[*]}"
  nohup env HF_HUB_OFFLINE=1 "${env_args[@]}" \
    "$SERVER_BIN" "$MODEL" \
    --host "$HOST" --port "$PORT" \
    --no-download \
    --max-tokens 2048 \
    "${server_args[@]}" \
    >"$slog" 2>&1 &
  echo $! >"$PIDFILE"
  local pid; pid=$(cat "$PIDFILE")
  log "pid=$pid"
  local i
  for i in $(seq 1 90); do
    if curl -sS --max-time 2 "$BASE/health" >/dev/null 2>&1; then
      log "health ok attempt=$i"
      curl -sS "$BASE/health" >"$RAW/${tag}-health.json" || true
      return 0
    fi
    if ! ps -p "$pid" >/dev/null 2>&1; then
      log "SERVER DIED"; tail -60 "$slog" | tee -a "$LOGS/runner.log"; return 1
    fi
    sleep 3
  done
  log "health timeout"; return 1
}

append_md() {
  local step="$1" config="$2" json_path="$3" prompt="$4"
  python3 - "$step" "$config" "$json_path" "$prompt" "$MANIFEST" "$BRANCH_LABEL" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
step, config, jp, prompt, md_path, branch = sys.argv[1:7]
p = Path(jp); md = Path(md_path)
raw = p.read_text(encoding="utf-8", errors="replace")
try:
    d = json.loads(raw)
except Exception as e:
    md.open("a").write(f"\n## {step}\n\nPARSE FAIL: {e}\n\n```\n{raw[:4000]}\n```\n\n---\n")
    print("PARSE_FAIL"); raise SystemExit(0)
if "error" in d:
    content = json.dumps(d, indent=2)
    finish, usage, body = "error", {}, content
    c_len = 0
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
    c_len = len(content)
md.open("a").write(f"""
## {step}

| Field | Value |
|-------|--------|
| branch_label | `{branch}` |
| time (UTC) | {datetime.now(timezone.utc).isoformat()} |
| model | LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit |
| config | `{config}` |
| raw JSON | `raw/{p.name}` |
| finish_reason | `{finish}` |
| usage | `{usage}` |
| content_chars | {c_len} |

### Prompt

```
{prompt}
```

### Full model response (manual verify)

```
{body}
```

---
""")
print(f"OK finish={finish} chars={len(body)}")
PY
}

chat() {
  local step="$1" config="$2" out_json="$3" max_tokens="$4" prompt="$5"
  local extra="${6:-}"
  local body
  body=$(python3 - "$MODEL" "$max_tokens" "$prompt" "$extra" <<'PY'
import json, sys
model, max_tokens, prompt, extra = sys.argv[1:5]
req = {
  "model": model,
  "temperature": 0,
  "max_tokens": int(max_tokens),
  "stream": False,
  "enable_thinking": True,  # THINKING ON always
  "messages": [{"role": "user", "content": prompt}],
}
if extra.strip():
    obj = json.loads("{" + extra + "}" if not extra.strip().startswith("{") else extra)
    req.update(obj)
print(json.dumps(req))
PY
)
  log "REQUEST $step max_tokens=$max_tokens -> $out_json"
  local http
  http=$(curl -sS --max-time 1200 -o "$out_json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d "$body" \
    "$BASE/v1/chat/completions") || true
  echo "{\"step\":\"$step\",\"http\":$http,\"file\":\"$out_json\",\"config\":\"$config\"}" >>"$META"
  log "RESPONSE $step http=$http size=$(wc -c <"$out_json" || echo 0)"
  append_md "$step" "$config" "$out_json" "$prompt"
}

PROMPT_SHORT='In 3 short steps, state Maxwell equations in vacuum (differential form) and what each means. Be concise.'
PROMPT_LONG='Derive the wave equation for E from Maxwell equations in vacuum (mu0, eps0). Show intermediate steps clearly, then state phase velocity. Keep math clean.'

cat >"$MANIFEST" <<EOF
# Isolation RESULTS — THINKING ON — branch \`${BRANCH_LABEL}\`

**Model ONLY:** \`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit\`  
**Thinking:** **ON** (\`enable_thinking: true\`, server **without** \`--no-think\`)  
**Server binary:** \`${SERVER_BIN}\`  
**Date:** $(date -Iseconds)

Raw JSON under \`raw/\`. Logs under \`logs/\`.

---
EOF

log "=== THINKING-ON LADDER START branch=$BRANCH_LABEL ==="

# T0: pure OFF + SYNC + no MTP + thinking ON (server allows thinking)
start_server T0 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  MLX_SYNC_DECODE=1 \
  --
chat "T0-short" "THINKING_ON PURE_OFF SYNC no-mtp max_tokens=1024" \
  "$RAW/T0-short.json" 1024 "$PROMPT_SHORT"
chat "T0-long" "THINKING_ON PURE_OFF SYNC no-mtp max_tokens=1600" \
  "$RAW/T0-long.json" 1600 "$PROMPT_LONG"

# T2: pure ON default + thinking ON (Discord-like defaults minus MTP)
start_server T2 \
  --
chat "T2-short" "THINKING_ON PURE_default_ON no-mtp max_tokens=1024" \
  "$RAW/T2-short.json" 1024 "$PROMPT_SHORT"
chat "T2-long" "THINKING_ON PURE_default_ON no-mtp max_tokens=1600" \
  "$RAW/T2-long.json" 1600 "$PROMPT_LONG"

# T4: MTP + thinking ON (may 500)
start_server T4 \
  MLX_DECODE_GRAPH_PURE_OFF=1 \
  MLX_SYNC_DECODE=1 \
  -- \
  --use-mtp
chat "T4-short-mtp" "THINKING_ON use-mtp PURE_OFF SYNC max_tokens=1024" \
  "$RAW/T4-short-mtp.json" 1024 "$PROMPT_SHORT" '"use_mtp": true'
chat "T4-long-mtp" "THINKING_ON use-mtp PURE_OFF SYNC max_tokens=1600" \
  "$RAW/T4-long-mtp.json" 1600 "$PROMPT_LONG" '"use_mtp": true'

python3 - "$RAW" "$MANIFEST" "$BRANCH_LABEL" <<'PY'
import json
from pathlib import Path
import sys
raw, md, branch = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows=[]
for p in sorted(raw.glob("T*.json")):
    if "health" in p.name: continue
    try:
        d=json.loads(p.read_text())
        if "error" in d:
            rows.append((p.name,"ERROR",d.get("error"),0)); continue
        ch=d["choices"][0]; c=(ch.get("message") or {}).get("content") or ""
        r=(ch.get("message") or {}).get("reasoning_content") or ""
        rows.append((p.name, ch.get("finish_reason"), d.get("usage"), len(c), len(r or "")))
    except Exception as e:
        rows.append((p.name,"PARSE",str(e),0,0))
lines=["\n## Summary\n", f"branch=`{branch}`\n",
"| file | finish | usage | content_chars | reasoning_chars |",
"|------|--------|-------|---------------|-----------------|"]
for row in rows:
    if len(row)==4:
        lines.append(f"| `{row[0]}` | `{row[1]}` | `{row[2]}` | {row[3]} |  |")
    else:
        lines.append(f"| `{row[0]}` | `{row[1]}` | `{row[2]}` | {row[3]} | {row[4]} |")
md.open("a").write("\n".join(lines)+"\n")
print("rows", len(rows))
PY

log "=== COMPLETE branch=$BRANCH_LABEL results=$MANIFEST ==="
kill_server || true
