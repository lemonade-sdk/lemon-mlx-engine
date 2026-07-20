# Investigation: Gibberish / “memory corrupt” via OpenWebUI + engine

**Status date:** 2026-07-19 (revised)  
**Symptom (Discord):** Math/reasoning starts coherent; then output becomes nonsense. Suspect KV/sync divergence, not pure algorithm. Path: **OpenWebUI + lemon-mlx-engine**. Screenshot also shows **CLI `./run.sh`** on **Qwen3.6-35B-A3B-MTP** (gfx1152).

**Supervisors / tools this revision:** explore (ownership map), quality-reviewer (critical MD audit), planning-analysis-strategist (experiment ladder), Clear Thought (scientific method + collaborative reasoning + divide-and-conquer).

---

## ⚠ CANONICAL TEST MODEL (MANDATORY)

**All isolation experiments, CLI runs, HTTP/curl runs, OpenWebUI repros, pure-graph A/B, MTP A/B, and server launches for this investigation MUST use only:**

| Field | Value |
|-------|--------|
| **HF model id** | **`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`** |
| Local cache | `~/.cache/huggingface/hub/models--LemonMLXE--Qwen3.6-35B-A3B-MTP-mlx-4bit/` |
| Snapshot | `5f638dff286ea1a97a6a0b673f50acc9b3c9aa4b` |
| Weights | `model.safetensors` ~**20 G** on disk |
| Architecture | `qwen3_5_moe` hybrid (`linear_attention` + `full_attention`), MTP head present |

### Explicit non-targets (do **not** use for diagnosis conclusions)

| Forbidden as primary test model | Why it is invalid for this ticket |
|---------------------------------|-----------------------------------|
| `mlx-community/Qwen3.5-0.8B-4bit` (and other tiny proxies) | Wrong size/arch; prior short A/B is historical only, **not** authoritative |
| `mlx-community/Qwen3.5-4B-*`, `Qwen3.6-27B-*`, other MLX ids | Different weights/behavior |
| `unsloth/...GGUF` / lemonade **llama-server :8001** | **Different stack** (llama.cpp Vulkan), not lemon-mlx-engine + MLX ROCm |
| Any “stand-in until 35B fits” | Operator mandate: **use the downloaded 35B MTP MLX model for all tests** |

**Server launch template (only valid model string):**

```bash
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 \
  --no-think \
  --max-tokens 2048
```

**CLI template:**

```bash
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit --no-think --max-tokens 2048
# or project run.sh equivalent with the same model id
```

Until this model is loaded under lemon-mlx-engine on free GPU VRAM, **no new empirical isolation results are accepted**. Code review / ownership analysis may continue; **token-level pass/fail requires this model only**.

---

## 0. Two-repo stack (answer: engine **and/or** MLX)

This is **not** a single-repo issue. lemon-mlx-engine **vendors MLX ROCm** via CMake `FetchContent`:

| Layer | Repo | Local tree | Role |
|-------|------|------------|------|
| **Engine** | this repo (`lemon-mlx-engine`) | `/home/antmi/lemon-mlx-engine` | Orchestration: pure-graph policy, GDN/MTP, ChatSession, HTTP/OpenAI server, EOS/stop, thinking |
| **MLX ROCm** | [NripeshN/mlx `rocm-support`](https://github.com/NripeshN/mlx/tree/rocm-support) | `build/_deps/mlx-src` only (no separate clone) | HIP graph capture/replay, decode arena, `gpu_kv_pos_*`, streams/events, gfx115x attention kernels |

```cmake
# CMakeLists.txt — always tracks branch tip (not a fixed SHA)
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/NripeshN/mlx.git
    GIT_TAG        rocm-support
)
```

| Pin fact | Value |
|----------|--------|
| **Declared tag** | branch `rocm-support` (moves) |
| **This machine’s checkout** | `0dadb703d77301af29405cf7e12627efb88a6d0f` |
| **Tip message** | `fix(rocm): WMMA flash FA wave size must be WARP_SIZE (not 64 fallback)` |
| **MLX version header** | 0.32.0 |

**Implication:** mid-stream “KV/sync” gibberish can require **engine** policy fixes (default pure-graph off, flag cleanup, ChatSession) **and/or** **mlx** capture/arena/kernel fixes. Engine can **mitigate without an mlx PR** via env defaults; root HIP capture bugs ship only in [NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support).

**lemonade** under `/home/antmi/lemonade` is a multi-backend launcher/proxy. It does **not** vendor this MLX tree. Inference brains for MLX path live in **lemon-mlx-engine + FetchContent mlx**.

---

## 1. Evidence from the screenshot / environment

| Observation | Implication |
|-------------|-------------|
| Model `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` | Hybrid **Qwen3.5-MoE** trunk + **MTP** head (`model_type: qwen3_5_moe`) |
| HF cache | Snapshot `5f638dff…`, weights ~**20G** on disk (downloaded) |
| `layer_types` | Mix of `linear_attention` (GDN-class) + `full_attention` (interval pattern) — **40 layers** |
| `eos_token_id` | **Multi-id** `[248046, 248044]` in `generation_config.json` |
| `[MTP] Found 0 quantized weight groups` / dequantized 20 weights | MTP head load path non-quant / fallback-ish |
| Inferred config via **base model fallback** | Risk of config mismatch if base ≠ trunk |
| gfx1152, cus=4 | Low-CU APU path (different from gfx1150) |
| Prompt: Maxwell’s equations | Coherent thinking outline starts |
| hipBLASLt caps line interleaved with “thinking process” | Stderr noise mixed into UX (**not** token corruption) |

User narrative: “math gets done but the memory gets corrupt” / “crazy gibberish” / “maybe OpenAI implementation?”

### Live machine confounders (2026-07-19)

| Fact | Status |
|------|--------|
| GPU VRAM | ~**8.5 GiB** total; **~98% used** |
| Process holding GPU | lemonade **`llama-server` GGUF** (Vulkan) on **`127.0.0.1:8001`** — `unsloth/Qwen3.5-35B-A3B-GGUF` |
| lemon-mlx `server` / 35B MLX | **Not loaded** (cannot load while :8001 holds VRAM) |
| Local lemond MLX registry | `127.0.0.1:13306` (recipe `lemon-mlx`, small models registered) |
| Residual VRAM | ~**170 MiB** — **not** enough for 0.8B or 35B MLX reloads |

**H0 — backend attribution (must pass before engine/mlx blame):**  
OpenWebUI may be pointed at **GGUF :8001**, **lemond :13306**, or a raw **lemon-mlx-engine server** port. Pure-graph / MTP / ChatSession hypotheses apply **only** when the request hits lemon-mlx-engine (or lemond → engine). Gibberish on **GGUF llama.cpp** is a **different stack** (not this MD’s pure-graph track).

---

## 2. Critical architecture facts

### 2.1 Client path: HTTP vs CLI KV

| Path | KV behavior |
|------|-------------|
| **OpenAI server** (`server.cpp` → `generate_text` / `TokenIterator`) | **Fresh KV every request** (`new_cache_fn`). Full `messages` re-templated every turn. |
| **CLI chat** (`ChatSession`) | **Persistent multi-turn KV** across turns. |

So classic “session KV reuse corruption” on the **HTTP/OpenWebUI path is largely refuted** for *cross-request* KV reuse.  
Gibberish can still come from **in-request decode** (pure-graph / MTP / async), **EOS overshoot**, **thinking budget**, **tool-role protocol**, or **client history pollution**.

### 2.2 Decode-path decision tree (mutually exclusive forks)

In `TokenIterator::next()` (`src/common/generate.cpp`):

```
use_mtp_ ?
  YES → MTP speculative path only  (pure-graph NOT entered)
  NO  → ROCm && !MLX_DECODE_GRAPH_PURE_OFF && pure_graph_state_ != 9 ?
          YES → step_pure_graph()   (DEFAULT ON when env unset)
          NO  → step() + async_eval (or MLX_SYNC_DECODE=1 fully sync)
```

| Flags | Active decode path |
|-------|--------------------|
| `--use-mtp` / `use_mtp: true` | **MTP only** (pure-graph skipped) |
| no MTP, ROCm, pure env not OFF | **Pure-graph** (default) |
| `MLX_DECODE_GRAPH_PURE_OFF=1` | Standard `step()` (+ optional sync) |

**Isolation rule:** turning pure-graph off does **nothing** if MTP is on, and disabling MTP does **nothing** about pure-graph. Never A/B “pure + MTP” as one bucket.

**Code honesty:** comment near pure-graph says “opt-in”; behavior is **opt-out** (`PURE_OFF` absence enables it). Product should match safer polarity (see §9).

### 2.3 Process-global `graph_external_pos`

| Behavior | Detail |
|----------|--------|
| Set `true` | Pure-graph warmup (`step_pure_graph`, state 0) |
| Cleared `false` | **Only** in pure-graph `disable()` (state → 9) |
| **Not** cleared | On normal generation end / iterator destroy after successful pure path |
| Scope | `static bool g_external` in `graph_decode.cpp` — **process-global** |

Model path (`qwen35_moe.cpp`): device-pos RoPE/KV and **GDN in-place** gate on `graph_external_pos()` (and related env). Sticky `true` after pure warmup can affect later L=1 paths in the same process even when pure is not the intended mode.

---

## 3. Ranked root-cause hypotheses (with ownership)

Legend: **E** = lemon-mlx-engine · **M** = NripeshN/mlx `rocm-support` · **B** = both · **C** = OpenWebUI / client · **N/A** = other stack

### H0 — Wrong backend (OWUI → GGUF :8001) — **GATE**

If OWUI talks to lemonade **llama.cpp Vulkan GGUF** on :8001, pure-graph/MTP/ChatSession are **out of scope**. Diagnose GGUF/client separately.

**Isolation:** OWUI connection URL/port/model id; raw `curl` to that same base URL.

---

### H1 — Decode path desync (ROCm pure-graph / GDN / async) — **HIGH** when **MTP off** · owner **B**

| Sub | Layer | What |
|-----|-------|------|
| **H1 policy** | **E** | Default ON; env kill switch; state machine in `step_pure_graph` |
| **H1 capture/arena** | **M** | `decode_capture_*`, `decode_arena_*` in `mlx/backend/rocm/` |
| **H1 pos kernels** | **M** | `gpu_kv_pos_*` / scalar inject (`indexing.hip`) |
| **H1 model path** | **E** | `qwen35_moe` device-pos + GDN inplace when `graph_external_pos` |
| **H1 GDN kernels** | **E** | `gated_delta.cpp` custom HIP (runtime via mlx) |
| **H1 external-pos sticky** | **E** | process-global flag lifecycle |
| **H1e kernels** | **M** | gfx115x WMMA FA / MoE / stream races if pure-off+sync still wrong |

**Isolation (MTP must be off):**

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1
export MLX_SYNC_DECODE=1   # optional: fully sync each step
# full process restart
```

If gibberish **disappears** with pure off → pure-graph stack (E first for default/cleanup; M if capture/arena proven).  
If still wrong under pure-off **and** sync → not pure-graph policy; consider weights/EOS/kernels (**E/M**).

---

### H2 — MTP speculative decode state — **HIGH when MTP enabled** · owner **E** (mostly)

- MTP **short-circuits** pure-graph (same request cannot blame both).
- Partial accept + hybrid GDN rollback is subtle; `MambaCache::set_position` is effectively a no-op (residual risk; snapshots/rollback also exist — do not overclaim “set_position alone dooms MTP”).
- Log: “Found 0 quantized weight groups” suggests non-standard MTP weight handling.

**Isolation:** ensure server **not** started with `--use-mtp` / request `use_mtp` false.  
If gibberish **only** with MTP → H2 (**E**).

---

### H3 — Incomplete EOS / ignored `stop` → continues after good answer — **HIGH** · owner **E**

- Generation stops on integer EOS ids only.
- Request `stop` strings are **parsed but not applied** in handlers.
- **Chat-template path** can replace multi-id EOS with a **single** template eos id (`llm_factory.cpp` ~321–325). This is **general loader behavior**, not MTP-only. MTP delta path only fills EOS if missing (~613–616).
- Model ships multi-id EOS `[248046, 248044]` — collapse risk is real for “coherent then junk until length.”

**Check:** `finish_reason` / `completion_tokens` vs `max_tokens`. If always `length`, audit EOS set for this model.

---

### H4 — Thinking budget (Qwen3.x + enable_thinking) — **HIGH for “half sense then collapse”** · owner **E** + **C**

- Default thinking ON unless `--no-think` / request override.
- CoT burns tokens; tail truncated or rushed (`finish_reason=length`).
- OpenWebUI multi-turn grows prompt → less room for final answer after think.

**Isolation:** `--no-think` or `enable_thinking: false`; raise `max_tokens`.

---

### H5 — OpenWebUI client context / tools protocol — **HIGH for “memory” feature path** · owner **C** (+ **E** protocol)

OpenWebUI is **stateless full-history** client:

| OWUI behavior | Engine reality |
|---------------|----------------|
| Memory / RAG / system tools inflate `messages` | Full re-prefill; no silent trim |
| Native tools → second request with `role: "tool"` | Engine **400**s tool roles (v1 freeze) |
| Tools markup stored as assistant text | Next turn polluted |
| Stream + reverse proxy buffering | UI “gibberish” that isn’t token-level |
| Stop / continue | Partial assistant text re-fed as full turn (not true continue) |

**“Math works, memory gibberish”** often means short math never hits tools/history growth; Memory/tools path hits **tool multi-turn** or **huge injection** → protocol/context failure looks like model “lost memory.”

This is **not** “GPU RAM of the model forgot facts” on HTTP path — the model only sees what OWUI re-sends.

---

### H6 — CLI ChatSession multi-turn full re-template on non-empty KV — **CONFIRMED defect (CLI)** · owner **E**

Not conditional speculation. On multi-turn with `CacheState::KVCache`:

1. Full history is re-templated (`build_messages` → full `turn_messages`).
2. Entire token sequence is passed into `TokenIterator` with **non-empty** `kv_cache_`.
3. `llm_default_prepare` **always** feeds **all** prompt tokens into that cache (chunked append; **no delta**).

→ **Deterministic double-prefill** for CLI turn 2+.  
**OpenWebUI does not use this path.** CLI `./run.sh` **does**.

Screenshot is CLI; user also reports OWUI — **both must be diagnosed separately**.

---

### H7 — OpenAI “implementation” alone — **LOWER as sole cause** · owner **E** (low)

- Fresh cache, full template each request (correct for multi-turn HTTP).
- Thinking guard restore is correct (no sticky thinking across HTTP requests).
- Tools Tier-1 limitations matter only if tools are used.

Unlikely pure “OpenAI wrapper invents gibberish tokens” without H0–H5.

---

### H8 — Non-pure async pipeline race — **MEDIUM** · owner **B**

`async_eval(token)` + `eval(previous)` pipeline lag. Isolate with `MLX_SYNC_DECODE=1`.

---

## 4. Ownership matrix (file pointers)

### A. Fix in **lemon-mlx-engine only**

| Topic | Paths |
|-------|--------|
| Pure-graph **policy** (default off / opt-in) | `src/common/generate.cpp` |
| `graph_external_pos` lifecycle / clear on gen end | `src/common/graph_decode.cpp`, pure `disable` + RAII |
| GDN / mamba layer math & custom HIP | `src/common/gated_delta.cpp`, `ssm_utils.cpp` |
| Hybrid model device-pos / inplace gates | `src/llm/models/qwen35_moe.cpp` |
| KV / MambaCache rollback | `include/mlx-lm/common/kv_cache.h`, `src/common/kv_cache.cpp` |
| MTP speculative decode | `src/common/generate.cpp`, `mtp_delta_kernel.cpp` |
| CLI multi-turn double-prefill | `src/common/chat_session.cpp` |
| HTTP API, stop/EOS, thinking | `src/common/server.cpp`, `src/llm/llm_factory.cpp` |
| CMake pin mlx SHA (repro) | `CMakeLists.txt` |

### B. Fix in **mlx `rocm-support` only** → https://github.com/NripeshN/mlx/tree/rocm-support

| Topic | Paths under `build/_deps/mlx-src/` |
|-------|-------------------------------------|
| HIP graph capture/replay | `mlx/backend/rocm/device.cpp` (`decode_capture_*`) |
| Decode arena address stability | `mlx/backend/rocm/allocator.cpp`, `allocator.h` |
| In-place pos/token GPU kernels | `mlx/backend/rocm/indexing.hip` |
| Stream sync / event fences | `eval.cpp`, `event.*`, `fence.cpp`, `host_stage.cpp` |
| Attention correctness on gfx115x | `flash_attention_wmma.hip`, SDPA |
| Async eval / scheduler | `mlx/transforms.cpp`, `scheduler.cpp` |
| MoE fused path hazards | `moe_swiglu.cpp` |

### C. **Needs both**

| Split | Engine | mlx |
|-------|--------|-----|
| Pure-graph end-to-end token validity | state machine, when capture/replay, external pos, tensors | capture, arena, pos kernels, graph mode |
| Device-position attention | `graph_decode_pos()`, model `update_at_pos` | pos buffers + SDPA |
| Async decode races | pipeline in `TokenIterator::next` | non-blocking streams + events |

### D. **Client only**

| Topic | Notes |
|-------|--------|
| OWUI Memory / RAG / native tools | Full history re-send; tool multi-turn unsupported → “memory gibberish” UX |
| Proxy SSE buffering | Looks like token corruption without being it |
| Backend URL mispointed at GGUF | H0 — not engine pure-graph |

---

## 5. OpenAI vs KV: what the user got right / wrong

| Claim | Assessment |
|-------|------------|
| “Something to do with syncing and/or KV” | **Partially right** for **in-request** pure-graph / GDN / MTP / async — **not** for HTTP multi-turn KV reuse |
| “Math done but memory corrupt” | Fits **mid-decode desync**, **EOS overshoot**, **thinking cut**, or **OWUI history/tools pollution** (or wrong backend) |
| “Doubt pure algorithm” | Agree — more systems/state than pure math kernels |
| “Maybe OpenAI implementation?” | Possible **protocol/history/tools** mismatch; less likely pure message mapping invents random tokens |
| “Is it lemon-mlx-engine or mlx?” | **Both possible**; see ownership matrix. Engine mitigates first; mlx only with evidence bar (§8). |

---

## 6. Isolation matrix (do this order)

**Every step below uses only `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` on lemon-mlx-engine (not GGUF, not 0.8B).**

| Step | Action | If fixed… |
|------|--------|-----------|
| **0** | Confirm OWUI hits engine serving **this** model (not GGUF :8001) | H0 wrong-backend |
| 1 | Single-turn, stream off, high `max_tokens`, `temperature=0`, no tools, `--no-think` | Client history / tools / thinking |
| 2a | If MTP on → **MTP off only** (do not toggle pure yet) | H2 |
| 2b | If MTP off → `MLX_DECODE_GRAPH_PURE_OFF=1` (+ optional `MLX_SYNC_DECODE=1`) | H1 pure-graph / GDN / async |
| 3 | Same prompt via raw `curl` vs OpenWebUI | OWUI assembly / proxy SSE |
| 4 | New chat, no Memory/RAG/native tools | OWUI injection / tool loop |
| 5 | Check `finish_reason` and token counts | EOS vs length vs cancel |
| 6 | CLI multi-turn only (turn 2+) | ChatSession double-prefill (H6) |

### Symptom taxonomy

| Symptom shape | Prefer |
|---------------|--------|
| Mid-stream **token soup** | H1 **or** H2 (XOR by config), not both as one cause |
| Coherent then junk until `length` | H3 EOS/stop, H4 thinking |
| “Memory” feature nonsense | H5 client (+ H0) |
| CLI turn 2+ only; HTTP OK | H6 engine |
| Only when OWUI tools/Memory on | H5 |
| hipBLASLt in the stream | stderr UX noise — not tokens |

---

## 7. Experiment log

### 7.1 Historical only (INVALID for conclusions — wrong model)

| ID | Config | Model | Path | Result | Validity |
|----|--------|-------|------|--------|----------|
| **A–C** | pure ON / OFF / OFF+SYNC | `Qwen3.5-0.8B-4bit` | CLI ~80 toks | Coherent | **INVALID** — not canonical model |
| **H0a/b** | thinking / no-think | GGUF on `:8001` | curl | length vs stop | **INVALID** for engine pure-graph — wrong stack |

Log retained for archaeology: `/tmp/gibberish-diag.log`, `/tmp/gibberish-h0/`.  
**Do not cite A–C or H0a/b as evidence that pure-graph is safe on the Discord model.**

### 7.2 Required ladder — **only** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`

**Results pack (manual verify):** [`docs/experiments/gibberish-isolation-2026-07-19/`](experiments/gibberish-isolation-2026-07-19/)
- **RESULTS.md** — full inlined model responses
- **ALL_RESPONSES.json** — all raw API payloads
- **raw/L*.json** — per-step files
- **INDEX.md** — summary table

| ID | Config | Result (2026-07-19 run) |
|----|--------|-------------------------|
| **L0-short/long** | pure OFF + SYNC + no MTP + no-think | **HTTP 200**, coherent Maxwell/wave (long hit `length`@800) |
| **L1-short/long** | pure OFF, no SYNC | **HTTP 200**, coherent |
| **L2-short/long** | pure default **ON** | **HTTP 200**, coherent (no mid-stream soup @800) |
| **L3-short/long** | thinking ON, pure OFF+SYNC | **HTTP 200**, thinking + content present |
| **L4-short/long** | `--use-mtp` | **HTTP 500** both — see `raw/L4-*-mtp.json` + `logs/L4-server.log` |


### 7.3 Ops still needed from human

| Ask | Why |
|-----|-----|
| Stop GGUF on `:8001` (consent) | Load **only** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| OWUI base URL | Confirm OWUI hits engine serving **this** model, not GGUF |

### 7.4 Recommended server env (when VRAM free) — **canonical model only**

```bash
export MLX_DECODE_GRAPH_PURE_OFF=1
export MLX_SYNC_DECODE=1
# do NOT pass --use-mtp until trunk is clean

./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 \
  --no-think \
  --max-tokens 2048
```

Every curl/CLI/OWUI body must use `"model": "LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit"` (or the id the server exposes for that load).  
Only after L0 stable on **this** model: re-enable thinking / pure-graph / MTP **one at a time** (full process restart each flip).

---

## 8. Experiment plan (supervisor)

### 8.0 Hard constraint

**Model id for every L\* / SM / CLI / HTTP / OWUI test:**  
`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
No 0.8B, no 4B, no 27B, no GGUF stand-ins.

### 8.1 NOW (no VRAM free / no kill of :8001)

1. **CR** pure-graph lifecycle, MTP fork, EOS loader, ChatSession — **done**.  
2. **Record** mlx SHA `0dadb703…`, model snapshot `5f638dff…` for **this** model.  
3. Ask OWUI port + consent to free GPU for **this** model only.  
4. Optional: engine P0 pure-graph opt-in design (no GPU).

### 8.2 After user frees GPU (consent required)

Do **not** kill lemonade `llama-server` without explicit approval.  
Load **only** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`.

Ladder (one variable per step; full restart; **always same model**):

| Step | Config | If symptom gone |
|------|--------|-----------------|
| **L0** | pure OFF + sync + no MTP + no-think + high max_tokens + curl non-stream | baseline “should work” |
| **L1** | drop SYNC only | async race (H8) |
| **L2** | pure ON (MTP still off) | **H1** → engine default/cleanup first; mlx if capture proven |
| **L3** | thinking ON | H4 / EOS length |
| **L4** | MTP ON (pure inactive) | **H2** engine |
| **L6** | stream ON | SSE / late desync |
| **L7** | OWUI clean (no Memory/tools) | OWUI assembly |
| **L8** | OWUI Memory/tools ON | H5 |
| **L10** | CLI multi-turn | H6 |

### 8.3 Evidence bar before PRs

| Target | Bar |
|--------|-----|
| **Engine-only PR** | Repro recipe + A/B one variable + path (CLI/HTTP) + scope proof. Pure-graph: OFF pass / ON fail **or** static proof of missing cleanup. ChatSession: multi-turn fail / HTTP OK. |
| **NripeshN/mlx PR** | Engine bar **plus** fail under pure-OFF **and** SYNC **or** isolated kernel wrongness; minimal repro preferred; gfx id + mlx SHA; not just CLI/OWUI/EOS cases. |
| **Do not open mlx PR if** | Only H6, only H5, only `finish_reason=length`, only hipBLASLt stderr, or **no fail on canonical `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`**. Proxy-model greens do not clear H1. |

### 8.4 Decision → repo

```
Token-level mid-gen gibberish on curl?
  no  → OWUI/client or CLI-only → E docs/ChatSession or C
  yes → L0 safe stack fail?
         yes → weights / deep mlx or hybrid-specific (B / M)
         no  → which step reintroduces?
                pure ON only → E first (+ M if capture)
                MTP ON only  → E
                thinking only → E
                CLI multi only → E H6
```

---

## 9. Likely fix directions (not all implemented)

| Priority | Fix | Owner |
|----------|-----|-------|
| **P0** | Make pure-graph **opt-in** (default OFF); positive env `MLX_DECODE_GRAPH_PURE=1` or CLI flag | **E** |
| **P0** | Clear `set_graph_external_pos(false)` + destroy capture on **every** generation end (RAII), not only pure disable | **E** |
| **P0** | Fix ChatSession: prefill **delta** only when reusing multi-turn KV (or drop reuse and full re-prefill) | **E** |
| **P1** | Honor request `stop`; preserve multi-id EOS from `generation_config` | **E** |
| **P1** | Document OpenWebUI: no multi-turn tools; disable Native Memory tools or expect 400 | **E** docs + **C** |
| **P1** | Pin `GIT_TAG` to mlx SHA for reproducible diagnosis | **E** CMake |
| **P2** | MTP: harden GDN rollback; warn if 0 quant groups / config fallback | **E** |
| **P2** | If pure-off+sync still fails on hybrid: mlx capture/arena/pos or gfx115x FA | **M** / **B** |

**Product note on pure-graph default:** performance experiment currently sits on the **correctness path** for ROCm chat. Opt-in (or auto-disable for hybrid GDN until soak-tested) is the recommended product polarity; mlx kernel fixes remain separate and do not gate “default safe.”

---

## 10. Supervisor summary (investigation gates)

| # | Focus | Result |
|---|--------|--------|
| 0 | Which backend is OWUI using? | **OPEN** — :8001 GGUF is live and may be the OWUI target |
| 1 | ChatSession multi-turn KV | **CONFIRMED double-prefill (CLI)**; **not** OWUI HTTP |
| 2 | Pure-graph vs MTP | **Mutually exclusive** decode forks — split isolation |
| 3 | Pure-graph ownership | **Both** (E policy/orchestration/model + M capture/arena/kernels) |
| 4 | OpenWebUI history/tools | **HIGH** for “memory” path (C + E protocol) |
| 5 | EOS / max_tokens | **HIGH** (E; multi-id EOS collapse is general loader) |
| 6 | OpenAI wrapper inventing tokens | **LOW** as sole root cause |
| 7 | Proxy-model (0.8B) pure A/B | **INVALID for conclusions** — wrong model |
| 8 | Empirical isolation on **canonical 35B MTP MLX** | **BLOCKED** on VRAM (GGUF :8001 @ 98%) |
| 9 | Pure-graph product default | **Should become opt-in** (E P0) |
| 10 | Canonical model mandate | **ALL tests = `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` only** |

---

## 11. Bottom line

1. **Not primarily “OpenAI API invents garbage.”**  
2. **Not primarily HTTP multi-turn KV reuse** (server rebuilds cache each request).  
3. **Token-level mid-gen soup on ROCm** is most likely **H1 pure-graph** (when MTP off) or **H2 MTP** (when MTP on) — **XOR**. Pure-graph ownership is **dual**: engine + [NripeshN/mlx rocm-support](https://github.com/NripeshN/mlx/tree/rocm-support).  
4. **All empirical proof must be on `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`.** Prior 0.8B / GGUF curls do **not** count.  
5. **CLI path** has a **confirmed** multi-turn double-prefill bug (**H6**, engine-only) — still verify on **this** model once VRAM free.  
6. **Next empirical step:** free GPU → load **only** this model → L0–L4 pure **XOR** MTP.  
7. **Code can still ship without GPU:** pure-graph opt-in + external-pos RAII + ChatSession delta-prefill + multi-id EOS — all **engine**.

---

## 12. Immediate asks for the human operator

1. **OK to stop lemonade GGUF on :8001** so we can load **`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`** and run the isolation ladder on **that model only**?  
2. **What base URL does OpenWebUI use**, and is it configured for **this** MLX model (not GGUF)?  
3. Prefer **engine P0** (pure-graph opt-in + external-pos clear ± ChatSession) in parallel while waiting on GPU?

---

*Update this file when isolation results land (fill §7 with canonical model only). Do not kill `:8001` without consent. Do not substitute another model.*

---

## 13. Thinking-ON dual proof (2026-07-19, late)

**Not clean:** feat still has real issues (MTP 500, thinking length, ChatSession, tools surface).

**Thinking ON** ladder + **main-equivalent** binary (same mlx):

- Pack: [`docs/experiments/gibberish-isolation-thinking-on-2026-07-19/`](experiments/gibberish-isolation-thinking-on-2026-07-19/)
- T0/T2 content **identical** feat vs main (only id/created differ)
- T4 MTP **same** HTTP 500 `Stream(cpu, 0)` on both
- Mid-stream token soup **not** reproduced on single-turn HTTP with thinking on for this model/prompt budget

