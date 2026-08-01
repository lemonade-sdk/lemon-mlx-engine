# P0-MTP gates (PR #63 → `fix/mtp-stream-p0`)

**Source:** `docs/FIX_AND_VERIFY_GUIDE.md` on PR #63 (`fix/eager-no-mtp-correctness`)  
**Branch under test:** `fix/mtp-stream-p0`  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` · gfx1150  

## Goal

Close **P0-MTP** (`Stream(cpu, 0)` HTTP 500 under `--use-mtp`) only when **M1+M2+M3** are green.  
M1 alone is a **false close**. M4/M6 are required for confidence; M6 is XOR hygiene.

## Code status (pre-gate)

| Item | Status on this branch |
|------|------------------------|
| `StreamGuard` in `mtp_speculative_step` | **Yes** (`generate.cpp`) |
| Own gen stream default (ROCm) | **Yes** (`generation_stream()`, opt-out `MLX_GEN_OWN_STREAM=0`) |
| MTP head default skip | **Yes** (`MLX_LOAD_MTP_HEAD=1` required) |
| Pure default OFF | **Yes** (`MLX_DECODE_GRAPH_PURE=1` opt-in) |
| M6 XOR (pure ignored when MTP) | **Yes** — `next()` short-circuits to MTP; log on conflict |
| Server `--use-mtp` / `--n-draft-tokens` | **Yes** (`examples/server.cpp`) |
| Dual-load ban (chat+server) | **Ops rule** — one process only on ~8 GB VRAM class |

## Close criteria (from #63)

| ID | Test | PASS | FAIL |
|----|------|------|------|
| **M1** | short no-think MTP HTTP | HTTP **200**, non-empty coherent, **no** `Stream(cpu`, log shows MTP | 500 / Stream / empty / crash |
| **M2** | long no-think MTP HTTP | 200; coherent math start; no Stream | 500 / soup |
| **M3** | thinking + MTP HTTP | 200 both short+long; no Stream (budget length OK) | any 500 Stream |
| **M4** | CLI single-turn MTP | Completes; non-empty | exception / Stream |
| **M6** | pure XOR | MTP-only path; pure not mixed; log when both set | silent pure+MTP mix |

**Close P0-MTP only if M1+M2+M3 green.**

## Operator recipe (single process)

```bash
# Free GPU: one model process only
pkill -f './build/(server|chat)' 2>/dev/null || true
sleep 2
ss -ltnp | grep 8080 || true

export MLX_DECODE_GRAPH_PURE_OFF=1   # belt; pure already default off
# Optional full fuse (orthogonal to Stream fix):
# export MLX_ENABLE_QUANT_FUSE=1 MLX_ENABLE_QUANT_FUSE_GDN=1
export MLX_LOAD_MTP_HEAD=1

MODEL=LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit
EXP=docs/experiments/mtp-stream-p0/gates

# --- M1: short no-think ---
./build/server "$MODEL" \
  --host 127.0.0.1 --port 8080 \
  --no-think --use-mtp --n-draft-tokens 2 --max-tokens 128 \
  > "$EXP/M1-server.log" 2>&1 &
# wait for /health, then:
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":64,"temperature":0}' \
  | tee "$EXP/M1-short.json"
grep -n 'Stream(cpu' "$EXP/M1-server.log" "$EXP/M1-short.json" || true

# --- M2: long no-think (same server) ---
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Explain the wave equation in plain language and give the 1D form. Keep it clear."}],"max_tokens":800,"temperature":0}' \
  | tee "$EXP/M2-long.json"

# --- M3: thinking + MTP (restart WITHOUT --no-think) ---
pkill -f './build/server' ; sleep 2
./build/server "$MODEL" \
  --host 127.0.0.1 --port 8080 \
  --use-mtp --n-draft-tokens 2 --max-tokens 4096 \
  > "$EXP/M3-server.log" 2>&1 &
# short + long with thinking on (default)

# --- M4: CLI (kill server first — dual-load ban) ---
# echo "Say hi in 5 words." | ./build/chat $MODEL --no-think --use-mtp --n-draft 2 --temperature 0 --max-tokens 64

# --- M6: pure XOR log (kill prior; set BOTH flags) ---
# MLX_DECODE_GRAPH_PURE=1 MLX_LOAD_MTP_HEAD=1 ./build/chat ... --use-mtp ...
# expect: "[MTP] M6 XOR: MLX_DECODE_GRAPH_PURE=1 ignored..."
```

## Harness

```bash
bash docs/experiments/mtp-stream-p0/run_p0_mtp_gates.sh
```

## Results

See `gates/RESULTS.md` after runs (written by harness / operator).
