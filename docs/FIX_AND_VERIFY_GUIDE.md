# Fix ownership, test matrix, and verification guide

**Status date:** 2026-07-19 (revised — operator constraints)  
**Audience:** operator / implementer  
**Canonical model (empirical decode tests ONLY):** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Active product posture (operator):**
- **No MTP** for now (`--use-mtp` deferred; P0-MTP not in flight)
- **Eager decode only** — pure-graph not used; pure-graph is now **opt-in** (`MLX_DECODE_GRAPH_PURE=1`)
- Work in **`/home/antmi/lemon-mlx-engine` only**; verify locally

**Baselines already captured:**

| Pack | Path |
|------|------|
| No-think ladder L0–L4 | `docs/experiments/gibberish-isolation-2026-07-19/` |
| Thinking-ON feat + main | `docs/experiments/gibberish-isolation-thinking-on-2026-07-19/` |
| Analysis | `docs/ISSUE_OPENWEBUI_GIBBERISH_KV_ANALYSIS.md` |

---

## Triple supervisor + Clear Thought sign-off

| Role | Agent / tool | Verdict |
|------|----------------|---------|
| **Decode/ROCm supervisor** | Clear Thought collaborative + explore | Engine-first for MTP/pure; mlx only after pure-off+sync or stream hygiene |
| **API/Client supervisor** | Clear Thought + planning strategist | Path-specific tests (HTTP ≠ CLI ≠ OWUI); thinking/tools separate |
| **QA/Regression supervisor** | quality-reviewer | Strict pass/fail, false greens forbidden, full regression order |
| **Clear Thought** | scientific method, divide-and-conquer, first principles, sequential, metacognitive | One independent variable per restart; save artifacts; not “feat is clean” |

**Consensus:** You still have **real P0 issues**. feat tools branch did **not** uniquely break decode vs main (thinking-on content matched). MTP 500 is **shared**. Verify every fix with the tests below.

### Operator scope note (2026-07-19, current)

| Feature | Operator stance | Implication for this guide |
|---------|-----------------|----------------------------|
| **MTP** | **Not enabling** — deferred | Do **not** spend cycles on P0-MTP until asked |
| **Pure-graph** | **Not enabling** — **eager only** (graph not worth it) | Pure-graph is **not** the active repro path; policy opt-in remains optional product hygiene, not the Discord fix |
| **Active path** | Eager decode, no MTP | Focus: **thinking budget**, **ChatSession CLI**, **EOS/stop**, **OWUI Memory/tools** |

**Eager / no-MTP server shape (matches operator):**

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1   # or future default OFF
# optional full sync each step:
# export MLX_SYNC_DECODE=1
# do NOT pass --use-mtp

./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 4096
```

---

## 1. What issues you have (plain)

| ID | Issue | Severity | Status |
|----|--------|----------|--------|
| **I1 / P0-MTP** | `--use-mtp` → HTTP 500 `There is no Stream(cpu, 0) in current thread` | **P0** | Reproduced feat **and** main — **DEFERRED** (operator: not enabling MTP) |
| **I2 / P0-thinking** | Thinking ON burns budget → `finish_reason=length`, looks like collapse | **P0** | **Mitigated:** defaults 4096 + server floor when thinking=on and max_tokens&lt;4096; still use `--no-think` for short Q&A |
| **I3 / P0-ChatSession** | CLI multi-turn full re-template onto non-empty KV (double-prefill) | **P0** | **FIXED** (full re-prefill + fresh KV each turn; unit test) |
| **I4 / P1-pure-graph** | Pure-graph default ON + sticky `graph_external_pos` | **P1** | **Default OFF** (opt-in `MLX_DECODE_GRAPH_PURE=1`); operator runs eager only |
| **I5 / P1-EOS-stop** | Multi-id EOS collapse; request `stop` parsed but not applied | **P1** | **Partially fixed**: multi-id EOS merge (no singleton replace); `stop` strings honored in server |
| **I6 / P1-OWUI** | Memory/tools/history / wrong backend look like “memory corrupt” | **P1** | Client + protocol |

**Not your primary issue:** OpenAI inventing tokens; HTTP multi-turn server KV reuse; “feat decode is worse than main” (content-matched on thinking-on ladder).

---

## 2. Where to make updates (repos)

| Layer | Repo / place | Local path |
|-------|----------------|------------|
| **Engine** | **`lemon-mlx-engine`** | `/home/antmi/lemon-mlx-engine` |
| **MLX ROCm** | **[NripeshN/mlx `rocm-support`](https://github.com/NripeshN/mlx/tree/rocm-support)** | `build/_deps/mlx-src` (FetchContent) |
| **Client** | OpenWebUI (+ lemonade if it sets ports) | Config / UI, not mlx |

Engine pins mlx in `CMakeLists.txt`:

```cmake
GIT_REPOSITORY https://github.com/NripeshN/mlx.git
GIT_TAG        rocm-support   # prefer pin to a SHA for repro
```

### Ownership table (explore supervisor)

| Fix ID | Primary | Secondary | Primary paths | Escalate to mlx when |
|--------|---------|-----------|---------------|----------------------|
| **P0-MTP** | Engine | mlx | `src/common/generate.cpp` (`mtp_speculative_step`, MTP branch of `TokenIterator::next`; missing `StreamGuard` vs `step`/`prepare`); `examples/server.cpp` `--use-mtp`; `qwen35_moe` / `llm_factory` MTP load | After engine stream/thread fix still `Stream(cpu, 0)`; error in mlx `backend/cpu/encoder.cpp` / stream TLS |
| **P0-thinking** | Engine | Client | `src/common/server.cpp` (`ThinkingContextGuard`); `model_manager.cpp`; `openai_types.h`; `--no-think` | **Never** for pure UX; only if no-think + high max_tokens still token soup → reclassify as pure/async |
| **P0-ChatSession** | Engine only | — | `src/common/chat_session.cpp`; `chat_session.h`; `llm_default_prepare`; `examples/chat.cpp`; `tests/test_chat_session.cpp` | **Never** |
| **P1-pure-graph** | Engine first | mlx | `generate.cpp` pure default; `graph_decode.cpp` `g_external`; `qwen35_moe.cpp`; `gated_delta.cpp` | Pure **OFF** + `MLX_SYNC_DECODE=1` + no MTP still soup → capture/arena/pos in `device.cpp` / `allocator.cpp` / `indexing.hip` |
| **P1-EOS-stop** | Engine only | — | `llm_factory.cpp` EOS merge; `server.cpp` (apply `stop`); `openai_types.h` | **Never** |
| **P1-OWUI** | Client + engine protocol | — | OWUI URL/Memory/tools; `server.cpp` tools/`role:tool` 400; `tool_calling.cpp` | Only if raw curl (no OWUI) still soup after pure-off+sync |

### Do **not** change for these fixes

| Wrong target | Why |
|--------------|-----|
| GGUF / llama.cpp / lemonade `:8001` | Different stack |
| mlx Metal/CUDA-only trees | ROCm path |
| mlx for ChatSession / EOS / stop / thinking defaults | Engine (+ client) |
| OpenWebUI for MTP Stream error | Reproduced via raw curl |
| Random FetchContent deps (json, fmt, …) | Irrelevant |

---

## 3. Global test harness (do this every time)

### 3.1 Hard rules

1. **Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` only for decode acceptance.  
2. **One independent variable** per full process restart.  
3. **MTP XOR pure-graph** — never claim both fixed from one run (`TokenIterator::next` short-circuits).  
4. **HTTP = fresh KV**; **CLI ChatSession = multi-turn KV** — never prove ChatSession via HTTP alone.  
5. **Not GGUF `:8001`** for engine evidence.  
6. **`temperature=0`** for regression.  
7. Save **raw JSON + server log + engine SHA + mlx SHA** under `docs/experiments/…`.  
8. **Full process restart** when changing env flags.

### 3.2 Pre-flight

```bash
# Ports / VRAM
ss -ltnp | grep -E '8001|8080' || true
rocm-smi 2>/dev/null | head -20

# Free GGUF only if you own it / consented
# curl -sS -X POST http://127.0.0.1:13305/api/v1/unload -H 'Content-Type: application/json' -d '{}'

# Model + binary
ls ~/.cache/huggingface/hub/models--LemonMLXE--Qwen3.6-35B-A3B-MTP-mlx-4bit/ | head
test -x ./build/server

# Pins
git rev-parse HEAD
(cd build/_deps/mlx-src && git rev-parse HEAD && git log -1 --oneline)
```

**Pre-flight PASS:** engine can load canonical model; VRAM free enough; health OK.  
**Pre-flight FAIL:** wrong stack / OOM / GGUF hog → stop.

### 3.3 Safe baseline server (until P0-MTP + pure policy land)

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1
export MLX_SYNC_DECODE=1
# do NOT pass --use-mtp until P0-MTP is green

./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 \
  --no-think \
  --max-tokens 2048
```

### 3.4 Artifact layout after a fix

```text
docs/experiments/verify-<fix-id>-YYYY-MM-DD/
  RESULTS.md
  raw/*.json
  logs/*-server.log
  meta.jsonl   # engine SHA, mlx SHA, binary md5, flags
```

---

## 4. Dependency order (what to fix first)

### 4a. Operator-active order (no MTP, eager only)

```
0. Pre-flight / H0 backend (engine URL, not GGUF :8001)
1. P0-ChatSession double-prefill ← DONE (full re-prefill + unit test)
2. P0-thinking budget            ← ops: high max_tokens or --no-think
3. P1-EOS multi-id + stop        ← DONE (merge EOS; honor request stop)
4. P1-OWUI Memory/tools          ← “memory corrupt” client path
5. Pure-graph                    ← default OFF (eager); do not enable
6. (Deferred) P0-MTP Stream(cpu,0) — do not enable --use-mtp
```

### 4b. Full product order (if shipping all features)

```
0. Pre-flight / H0 backend
1. P0-MTP Stream(cpu,0)          ← hard 500 when --use-mtp (deferred for operator)
2. P0-thinking budget
3. P0-ChatSession double-prefill
4. P1-pure-graph policy + RAII
5. P1-EOS multi-id + stop
6. P1-OWUI Memory/tools
```

After **any** code change: run **§8 regression order** for the **active** suites only (skip M* MTP tests while MTP deferred).

---

## 5. Per-priority: implement + verify

---

### P0-MTP — `Stream(cpu, 0)` HTTP 500

#### Goal
`--use-mtp` returns **HTTP 200** with real tokens; no Stream error.

#### Where to change

| Primary | `lemon-mlx-engine` `src/common/generate.cpp` (MTP path: stream discipline / `StreamGuard` like `step`/`prepare`) |
| Secondary | mlx `stream.cpp`, `backend/cpu/encoder.cpp`, ROCm worker thread CPU stream registration |

#### Implement (high-level)

1. Reproduce with `--use-mtp`, pure off, no-think.  
2. Ensure MTP runs under explicit **GPU** stream; do not assume `Stream(cpu, 0)` exists on the server worker thread.  
3. Optional probe: `MLX_GEN_OWN_STREAM=1` (restart required).  
4. Escalate to **NripeshN/mlx** only if engine stream hygiene is done and error still throws from mlx encoder.

#### REQUIRED verify tests

| ID | Test | What you do | PASS | FAIL |
|----|------|-------------|------|------|
| **M1** | short no-think MTP | Restart: `PURE_OFF=1 SYNC=1 ./build/server $MODEL --no-think --use-mtp --max-tokens 512`. curl Maxwell short, temp=0. | HTTP **200**, non-empty coherent content, **no** Stream string, log shows MTP | 500 / Stream / empty / crash |
| **M2** | long no-think MTP | Same server; wave prompt; max_tokens=800 | 200; coherent math start; no Stream | 500 / soup |
| **M3** | thinking + MTP | Restart without `--no-think`; keep `--use-mtp`; short+long | 200 both; no Stream (budget length OK) | any 500 Stream |
| **M4** | CLI single-turn MTP | `./build/chat $MODEL --no-think --use-mtp` one prompt | Completes; non-empty | exception / Stream |
| **M6** | XOR guard | `--use-mtp` with pure default ON | MTP-only or clean refuse; pure not silently mixed | pure+MTP same request |

**Close P0-MTP only if M1+M2+M3 green.** M1 alone is a **false close**.

#### Copy-paste verify core

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1 MLX_SYNC_DECODE=1
pkill -f './build/server.*8080' 2>/dev/null || true; sleep 1
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --no-think --use-mtp --max-tokens 128 \
  > /tmp/mtp-p0-server.log 2>&1 &
sleep 30
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit","messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":64,"temperature":0}' \
  | tee /tmp/mtp-p0-short.json
grep -n 'Stream(cpu' /tmp/mtp-p0-server.log /tmp/mtp-p0-short.json || true
# PASS: no matches + HTTP 200 + non-empty content
```

#### Baseline before fix

`docs/experiments/gibberish-isolation-2026-07-19/raw/L4-*-mtp.json`  
`docs/experiments/gibberish-isolation-thinking-on-2026-07-19/{feat,main}/raw/T4-*-mtp.json`  
→ HTTP 500 Stream error.

---

### P0-thinking — budget / “collapse” UX

#### Goal
Thinking ON + adequate `max_tokens` yields a **final answer** (prefer `finish_reason=stop`); low budget clearly `length`. Flag `enable_thinking` honored.

#### Where to change

| Primary | Engine `server.cpp`, `model_manager.cpp`, defaults / docs |
| Client | OpenWebUI max_tokens / reasoning toggle |

**Not mlx.**

#### Implement (high-level)

1. Policy: raise recommended/default max_tokens for thinking models **and/or** document client settings **and/or** thinking budget cut then answer.  
2. Log `effective_thinking=on|off`.  
3. No sticky thinking polarity across HTTP requests.

#### REQUIRED verify tests

| ID | Test | What you do | PASS | FAIL |
|----|------|-------------|------|------|
| **T1** | think-on sufficient budget | PURE_OFF+SYNC, no MTP, no `--no-think`, Maxwell short, max_tokens=**2048**, `enable_thinking:true` | 200; final content has Maxwell; prefer stop; if length still has answer start | thinking only / empty final |
| **T2** | think-on long | Wave prompt; max_tokens=1600 **and** 4096 | 1600: coherent if length; 4096: stop + phase velocity / complete | mid-stream soup; empty after think |
| **T3** | think-off parity | `enable_thinking:false` / `--no-think` | L0-class coherent; no leaked think tags | sticky think / leaked tags |
| **T4** | request override | Server `--no-think` + body `enable_thinking:true` (and reverse) | Precedence as documented; next request not sticky wrong | sticky process polarity |
| **T5** | think + pure ON | pure default ON, think ON, long wave | Not worse than baseline T2-long; no soup | T2-long-class garbage |

**Close only if T1+T2+T5 green.**

#### Copy-paste verify core

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1 MLX_SYNC_DECODE=1
# NO --no-think
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --max-tokens 4096 &
# V1 low budget → expect length
# V2 high budget → expect stop + complete answer
# V3 enable_thinking false → short coherent
```

---

### P0-ChatSession — CLI multi-turn double-prefill

#### Goal
Turn 2+ CLI does **not** append full history tokens onto non-empty KV. HTTP path unchanged (already correct).

#### Where to change

| Only | `src/common/chat_session.cpp` (+ tests). Optional: clear KV and full re-prefill (slower but correct). |

**Not mlx. Not OWUI.**

#### Implement (high-level)

1. Track cached length / message boundaries.  
2. Prefill **delta only** **or** clear cache and full re-prefill each turn.  
3. Unit test must fail on old code, pass on new; still run C1 on 35B.

#### REQUIRED verify tests

| ID | Test | What you do | PASS | FAIL |
|----|------|-------------|------|------|
| **C1** | CLI turn2 instrumented | chat multi-turn: name Ada → “what is my name?” Log prefill length / cache offset | Turn2 delta-only (or empty-cache full); answers Ada | Full history into non-empty KV; nonsense |
| **C2** | CLI turn3 | Continue chain | Coherent chain | explode / gibberish |
| **C3** | HTTP multiturn control | Same 3 turns via full messages HTTP | Correct | HTTP fail = different bug |
| **C4** | CLI ≥ HTTP quality | Side-by-side turn2 | CLI OK | CLI only broken |
| **C5** | unit test | `ctest -R chat_session` | Fails pre-fix, passes post | unit green, C1 fails = **false green** |

**Close only with C1 instrumented proof + C2.**

```bash
cd build && ctest -R chat_session --output-on-failure
export MLX_DECODE_GRAPH_PURE_OFF=1 MLX_SYNC_DECODE=1
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit --no-think --max-tokens 64
# Turn1: My name is Ada. Reply OK.
# Turn2: What is my name? One word.
```

---

### P1-pure-graph — default policy + sticky `graph_external_pos`

#### Goal

1. Pure-graph **opt-in** (default OFF).  
2. Clear `graph_external_pos` + destroy capture on **every** gen end (RAII).

#### Where to change

| Primary | `generate.cpp`, `graph_decode.cpp` / `.h`, model gates `qwen35_moe.cpp` |
| mlx if needed | `device.cpp` decode_capture_*, `allocator.cpp` arena, `indexing.hip` |

#### Implement (high-level)

1. Invert default: enable only `MLX_DECODE_GRAPH_PURE=1` (or CLI flag).  
2. RAII clear external pos + capture destroy on iterator end / exception.  
3. Document restart-on-env-change.

#### REQUIRED verify tests

| ID | Test | PASS | FAIL |
|----|------|------|------|
| **P1** | pure-off baseline hold (L0 short+long) | ≥ baseline L0 | regression |
| **P2** | pure-on short | coherent Maxwell | soup/crash |
| **P3** | pure-on long stress max_tokens 800/1600/2048 | no mid-stream soup | soup after N tokens |
| **P4** | pure-on + thinking long | not T2-long garbage | baseline soup |
| **P5** | external-pos lifecycle | flag false after gen; next request sane | sticky pos |
| **P6** | pure async vs SYNC | both coherent or document race | only SYNC works → not “pure fixed” |

**Close only if P3 (≥1600) + P4 + P5.** Short-only pure-ON green is **insufficient**.

#### Escalate to mlx

Only if **pure OFF + SYNC + no MTP** still has token-level soup on canonical model via raw curl.

---

### P1-EOS-stop — multi-id EOS + honor `stop`

#### Goal

1. Keep multi-id EOS (model: `248046`, `248044`).  
2. Apply request `stop` strings in generate loops (stream + non-stream).

#### Where to change

| Only | `llm_factory.cpp` (union EOS, never replace multi with singleton); `server.cpp` (apply `chat_req.stop`); tests |

#### REQUIRED verify tests

| ID | Test | PASS | FAIL |
|----|------|------|------|
| **E1** | multi-id EOS set after load | both ids present | collapsed to one |
| **E2** | natural short stop (“Reply exactly: OK”) | `finish_reason=stop`, tokens ≪ max | always length + junk |
| **E3** | `stop: ["###END###"]` or count-to-N stop | stops at string; no overrun | stop ignored |
| **E4** | early answer high max_tokens | clean stop; no overshoot junk | good then junk to length |
| **E5** | thinking then EOS | clean after final answer | soup after answer |

**Close if E1+E2+E3 green.**

---

### P1-OWUI — Memory / tools / wrong backend

#### Goal

H0 correct backend; clean chat matches curl; tools/Memory either work or **clear** 4xx—not token soup 200.

#### Where to change

| Client | OWUI base URL, Memory off, tools off/on per support |
| Engine | tools parse/emit (feat); `role:tool` → 400 with clear body; docs |

#### REQUIRED verify tests

| ID | Test | PASS | FAIL |
|----|------|------|------|
| **O0** | H0 backend gate | models/chat hit engine + canonical model id | GGUF :8001 only |
| **O1** | OWUI single-turn no tools/Memory | matches curl quality | UI soup, curl clean → SSE/proxy |
| **O2** | OWUI multiturn no tools | correct history re-send | “memory” fail with clean curl |
| **O3** | Memory on vs off | A/B attributed correctly | blame pure-graph without A/B |
| **O4** | tools tier-1 (feat) | tool_calls or clean 4xx | 500 Stream / free-text “success” |
| **O5** | `role:tool` multi-turn | clear 400 | 200 nonsense |
| **O6** | product argv | no accidental global `--no-think`/`--use-mtp` | silent product flags |

**O0 is a gate.** No OWUI pass without it.

```bash
curl -sS http://127.0.0.1:8001/v1/models | head -c 200; echo
curl -sS http://127.0.0.1:8080/v1/models | head -c 400; echo
# Tool role must 400 on freeze:
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit","messages":[
    {"role":"user","content":"hi"},
    {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]},
    {"role":"tool","tool_call_id":"c1","content":"{}"}
  ],"max_tokens":32}'
```

---

## 6. Full regression order after ANY fix

Full **process restart** between env/flag changes. Always canonical model.

| Order | Suite | Config | Minimum tests | Stop if fail |
|------:|-------|--------|---------------|--------------|
| 0 | Env gate | free VRAM, health, models | health + models | wrong stack |
| 1 | L0 hold | PURE_OFF+SYNC, no MTP, no-think | short+long Maxwell | baseline broken |
| 2 | L1 async | PURE_OFF, no SYNC | short+long | async regression |
| 3 | ChatSession | L0 env; **CLI** | C1, C2, C3 | P0-ChatSession open |
| 4 | Thinking | PURE_OFF+SYNC, think ON | T1, T2 | P0-thinking open |
| 5 | Pure-graph | pure ON default | P2, P3, P4 | P1-pure open |
| 6 | EOS/stop | pure OFF, no MTP | E1–E4 | P1-EOS open |
| 7 | MTP | PURE_OFF+SYNC, `--use-mtp` | M1, M2, M3, M6 | P0-MTP open |
| 8 | XOR / sticky | M6, P5 | | policy bug |
| 9 | OWUI | if product | O0→O6 | client/protocol |
| 10 | Pack | RESULTS.md vs baselines | | new soup/500 |

### Narrow fix extras

| Fix | Also run |
|-----|----------|
| MTP only | M* + L0 + M6 |
| Thinking only | T* + L0 no-think + E5 |
| ChatSession only | C* + L0 HTTP |
| Pure only | P* + L0 + T5; **not** MTP |
| EOS only | E* + T1 |
| OWUI only | O0–O6 + L0 curl twin |

---

## 7. False greens (forbid)

| ID | Claim | Why invalid | Require instead |
|----|--------|-------------|-----------------|
| FG1 | L0–L2 coherent ⇒ clean | MTP/thinking/CLI still open | Full inventory |
| FG2 | feat == main ⇒ no work | Shared bugs remain | Shared P0 still open |
| FG3 | Proxy model green | Wrong model | Canonical 35B only |
| FG4 | GGUF :8001 green | Wrong stack | Engine + MLX model |
| FG5 | HTTP multiturn OK ⇒ ChatSession fixed | Fresh KV | **C1** |
| FG6 | Pure OFF green ⇒ pure fixed | Never exercised pure | **P3/P4** |
| FG7 | Pure ON short only | Insufficient | long ≥1600 + think |
| FG8 | MTP 200 empty | Not integrity | M1 content bar |
| FG9 | MTP silent fallback | Hides break | Log MTP path |
| FG10 | stop with empty final content | Budget lie | non-empty final |
| FG11 | unit only without C1 35B | Miss double-prefill | C1 |
| FG12 | stop field accepted but ignored | Parse ≠ honor | **E3** |
| FG13 | single EOS “works” | Latent multi-id | **E1** |
| FG14 | Memory off green ⇒ Memory OK | | O3 A/B |
| FG15 | tools free-text as success | Protocol lie | structured tool_calls |
| FG16 | hipBLASLt gone ⇒ tokens fixed | stderr ≠ tokens | raw JSON |
| FG17 | no process restart | sticky env/flags | restart matrix |
| FG18 | no Stream but MTP not entered | Flag dropped | log MTP active |
| FG19 | temp>0 single sample | Non-repro | temp=0 |
| FG20 | pure+MTP one bucket | XOR | separate M* and P* |

---

## 8. When to open a PR on NripeshN/mlx

All of:

1. Repro on **canonical model** via raw `curl` to `./build/server` (not OWUI-only).  
2. Engine mitigations tried: pure OFF, SYNC, MTP off (for pure bugs) **or** stream fix attempts (for MTP).  
3. Failure is stream/capture/arena/kernel — **not** EOS, ChatSession, stop, Memory, thinking budget.  
4. Package: error, gfx, mlx SHA, engine SHA, curl, A/B table.

| Symptom after engine work | mlx? | Focus |
|---------------------------|------|--------|
| MTP still Stream(cpu,0) after stream hygiene | **Yes** | CPU stream TLS / encoder |
| Pure-OFF+SYNC still soup | **Yes** | kernels / async |
| Pure-ON fails after RAII; pure-OFF OK | **Maybe** | capture/arena/pos |
| CLI multiturn only | **No** | ChatSession |
| length / stop ignored | **No** | EOS/stop |
| OWUI Memory/tools | **No** | client + protocol |

---

## 9. Sign-off checklist (cannot ship as “fixed” until)

| Priority | Cannot sign off until |
|----------|------------------------|
| **P0-MTP** | M1+M2+M3 + M6 |
| **P0-thinking** | T1+T2+T5 |
| **P0-ChatSession** | C1 instrumented + C2 + C3 control |
| **P1-pure-graph** | P3 (≥1600) + P4 + P5 |
| **P1-EOS-stop** | E1+E2+E3 |
| **P1-OWUI** | O0 + O1 + O4/O5 honesty |

**Reject:** “looks fine,” “same as main,” “short Maxwell works,” “unit tests pass,” “no 500 on non-MTP.”  
**Accept:** config matrix, artifacts under `docs/experiments/…`, baseline deltas, no FG* conditions.

---

## 10. What you do day-to-day (current posture)

1. **Do not** use `--use-mtp` (deferred).  
2. Raise **max_tokens** for thinking, or use `--no-think` / `enable_thinking:false` for short Q&A.  
3. CLI multi-turn: **ChatSession full re-prefill is fixed** — still prefer high max_tokens with thinking ON.  
4. OWUI: new chat, Memory/tools off, URL = **engine** serving canonical model.  
5. **Eager only:** leave pure-graph unset (default OFF). Optional `MLX_SYNC_DECODE=1` for max safety.

---

## 11. Related docs

| Doc | Use |
|-----|-----|
| `docs/ISSUE_OPENWEBUI_GIBBERISH_KV_ANALYSIS.md` | Full diagnosis |
| `docs/OWUI_OPS_CHECKLIST.md` | OpenWebUI H0 backend gate + Memory/tools posture |
| `docs/LOOP_STATUS.md` | Branch progress (eager no-MTP track) |
| `docs/experiments/gibberish-isolation-2026-07-19/` | No-think baselines |
| `docs/experiments/gibberish-isolation-thinking-on-2026-07-19/` | Thinking-ON + feat vs main |
| This file | **Fix + verify only** |

---

## 12. Supervisor attestation

| Supervisor | Agreed |
|------------|--------|
| Decode/ROCm (Clear Thought + explore) | Engine-first; mlx bar documented |
| API/Client (planning) | Path-specific verify; dependency order |
| QA (quality-reviewer) | Required tests, false greens, regression order |

*Generated with Clear Thought MCP (scientific method, divide-and-conquer, first principles, collaborative reasoning, sequential, metacognitive) + triple domain supervisors. No claim that feat is issue-free.*
