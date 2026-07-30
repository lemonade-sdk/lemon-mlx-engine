# Complete work report — ROCm GDN multi-turn collapse (PR #74)

**Document type:** End-to-end engineering + experiment report  
**Branch:** `fix/rocm-gdn-fused2-optin`  
**PR:** [#74](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/74) — **human merge only**  
**Date:** 2026-07-29 (local) / 2026-07-30 (UTC for some CI stamps)  
**Host:** Ubuntu, AMD Radeon 890M (`gfx1150`), 8 CUs, ROCm  
**Engine tip (code):** `52d64de`  
**Docs tip:** `a372276`  
**Binary under matrix:** `build/chat` mtime epoch `1785380863` (engine = `52d64de`; docs-only commits after do not change binary)

**Related docs (this directory):**

| File | Role |
|------|------|
| `REAL_ISSUE.md` | Charter — what counts as fixed |
| `RESULTS_LOCAL_gfx1150.md` | Early 0.8B ladder / residual notes |
| `MASTER_REPORT_FIELD_MATRIX_2026-07-29.md` | 35B 2×2 matrix snapshot |
| **This file** | **Complete consolidated report** |

---

## 1. Executive summary

### The problem

On ROCm, multi-turn chat with hybrid Qwen **GDN** (gated delta net) models can stay coherent for several turns, then **collapse** on a long generation (especially code) into endless token/phrase thrash — historically `synchronization synchronization…`, this matrix also `f_s_orig*f_s_orig*…`.

This is **not**:

- Cross-turn GDN cache reuse (each turn full re-prefill + **fresh** cache)
- Fixed by LoopBrake / maxphrase seatbelts
- Proven by CI “2+2” smoke alone

### What we shipped (real fixes)

1. **Float32 SSM lifetime** for prefill multi-T **and** decode T=1 (stop bf16/InT state RMW between tokens)
2. **Prefill keep SSM f32** — stop casting multi-T state back to bf16 (was undoing f32 accumulate)
3. **g → activation dtype**, softplus via **logaddexp**, act-dtype q/k norms
4. **`gdn_fused_decode` (fused2) opt-in** — was originally default ON; demoted after hard loops
5. **LoopBrake removed** — early-stop is not a GDN fix

### Headline experimental matrix (35B field SAR)

Same binary, same prompts, MTP off, no LoopBrake:

| Path | temp = 0 | temp = 0.7 (product-like) |
|------|----------|---------------------------|
| **Default** (no fused2) | **PASS** (D0) | **MIXED** — FAIL n=1 thrash / **PASS n=2** |
| **`MLX_GDN_FUSED2=1`** | **PASS** (F0) | **PASS** 2/2 (F7 n=1 + n=2) |

Do not over-claim statistical certainty. Product bar = default@0.7 without seatbelts.

### Bottom line

| Claim | Verdict |
|-------|---------|
| f32 SSM / prefill≡decode is the real correctness class | **Supported** |
| Field collapse “fully fixed” for product defaults | **No** — default@0.7 thrash still observed once; n=2 green is not enough alone |
| fused2 known poison forever | **No** — fused@0 and @0.7 PASS; F7 **2/2** on tip |
| fused2 as product **code** default tonight | **Not yet** — keep opt-in; operator may `export MLX_GDN_FUSED2=1` |
| ~63 t/s expected on 35B gfx1150 | **No** — measured ~25–27; ~63 is 0.8B-class |
| LoopBrake as product success | **Rejected** |
| Parity tools urgent tonight | **No** — charter: only if D7 stays red after n=2; D7 did not |

---

## 2. System under test

### Models

| Role | Model |
|------|--------|
| **Field oracle** | `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| **Fast bisect only** | `mlx-community/Qwen3.5-0.8B-4bit` (not field oracle) |

### Field acceptance sequence (PASS definition)

```
tell me about maxwell's equations
tell me about fourier analysis
how about doppler
taken together, what are these good for?
that sounds really interesting, can you give me a python implementation?
```

**PASS:** Coherent turns 1–4; turn 5 **usable Python** (or real plan+code); finishes or hits budget with real content; **no** endless single-token/phrase thrash. No seatbelt required.

**FAIL:** Starts code (or answer) then collapses into repeated tokens/phrases; or “works” only via early-stop.

### Canonical product-like chat flags

```bash
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 20480 \
  --repetition-penalty 1.0 \
  --ctx-size 32768
```

(`examples/chat.cpp` defaults temperature **0.7**, top_p **0.9**. MTP head skipped unless enabled.)

### What the engine does each turn

```
history messages
  → chat template (full re-prefill every turn)
  → Prefill T>1: attention + GDN (conv window, multi-T SSM)
  → Decode T=1 loop: GDN step
       default: external rms_norm + gated_delta_update
       opt-in:  gdn_fused_decode if MLX_GDN_FUSED2=1
  → logits → sample
  → append assistant text to history
  → drop KV/GDN cache (fresh next turn)
```

**Implication:** Collapse = long T=1 horizon and/or re-prefill of bad prior text + **prefill≡decode numerics**, not cross-turn SSM reuse.

---

## 3. Decode path taxonomy (do not mix names)

| Path | How selected | What runs |
|------|----------------|-----------|
| **Default decode** | (no env) | External `rms_norm` + `gated_delta_update` (HIP fused **step**) |
| **fused2** | `MLX_GDN_FUSED2=1` (exact `"1"`) | `gdn_fused_decode` mega-kernel (q/k RMSNorm + beta/g + recurrence) |
| **Force fused2 off** | `MLX_GDN_NO_FUSED2=1` | Even if FUSED2 set |
| **No HIP GDN fuse** | `MLX_GDN_NO_FUSED=1` | `mx::compile` inline recurrence (slowest / portable) |

**Originally fused2 ON?** **Yes.**

| Era | Commit | Polarity |
|-----|--------|----------|
| Intro | `8d7c95b` (2026-06-25) | **Default ON**; opt-out via `MLX_GDN_NO_FUSED2` |
| Flip | `ab1b518` (2026-07-29) | **Default OFF**; opt-in via `MLX_GDN_FUSED2=1` |
| Now | tip | Still opt-in |

Original gate:

```cpp
// Default: FlashQLA-style gdn_fused_decode
use_fused2 = use_fused_gdn && (getenv("MLX_GDN_NO_FUSED2") == nullptr);
```

Current gate:

```cpp
// Default: rms_norm + gated_delta_update
// MLX_GDN_FUSED2=1: gdn_fused_decode
use_fused2 = use_fused_gdn && fused2_opt_in && !fused2_force_off;
```

`export MLX_GDN_FUSED2=1` **restores original product polarity** without a code flip.

---

## 4. Engineering work on this branch

### 4.1 Commit stack (engine-relevant, oldest → newest among key fixes)

| Commit | Summary |
|--------|---------|
| `ab1b518` | fused2 **opt-in**; cast recurrence dtype on MoE path |
| `d218c7c` | **P0 g-dtype**: cast decay `g` to activation dtype; match prefill `a_log` |
| `64387f8` / `0564ffc` / `b170395` / `2a8b536` | LoopBrake (later **removed**) |
| `7b8f17d` | fused2 g/beta InT round-trip + Dk%32 guard |
| `43c8ae6` | stable softplus via **logaddexp** on default paths |
| `36f83f5` | **Remove LoopBrake** entirely |
| `c7685d8` | **float32 SSM lifetime** + act-dtype q/k norm (decode) |
| `52d64de` | **Prefill keep SSM float32**; f32 softplus on NO_FUSED path |
| `a372276` | Docs: field PASS stamp (prefill f32) |

Full branch vs `origin/main`: ~21 commits (includes docs/logs).

### 4.2 Core code changes (by theme)

#### A. Float32 SSM lifetime (`c7685d8`, `52d64de`)

**Problem:** Prefill multi-T kept SSM in f32 during the sequence, then **cast state back to bf16** into cache; decode T=1 did InT/bf16 RMW every step → error grows with horizon; prefill≢decode.

**Fix:**

- HIP `gated_delta_step` / seq / fused2: `state_in` / `state_out` **float32**; only `y` quantized to activation dtype
- Cache zeros / promote: f32 SSM
- Prefill: **do not** cast `new_state` to act/bf16 after multi-T
- Decode: keep `ns` f32 between tokens

#### B. Prefill ≡ decode dtypes (`d218c7c`, norms, softplus)

- Decay `g` in **activation** dtype (not forced f32 from `a_log_f32` alone on wrong path)
- q/k RMSNorm weights materialized in **activation** dtype
- softplus: cast `a` / `dt_bias` to f32, use **logaddexp** (torch-aligned)

#### C. fused2 policy (`ab1b518` + parity patches)

- Default path safer after H1 hard loops on gfx1150 multi-turn thinking
- fused2 remains available for perf/path A/B
- Parity work: RMSNorm/softplus/InT/Dk%32 while still opt-in

#### D. LoopBrake (`36f83f5` final)

- Temporarily added as residual CoT mitigation
- **Removed:** masks field bugs; not GDN numerics; charter forbids it as success

### 4.3 Key files

| Area | Path |
|------|------|
| T=1 GDN select / prefill cache | `src/llm/models/qwen35_moe.cpp` |
| Header / materialize | `include/mlx-lm/llm/models/qwen35_moe.h` |
| HIP GDN + fused2 kernel | `src/common/gated_delta.cpp` |
| Softplus on other GDN models | `src/llm/models/qwen35.cpp`, `qwen3_next.cpp` |
| Multi-turn session | `src/common/chat_session.cpp` |
| Decode async / sample | `src/common/generate.cpp` |

---

## 5. Hypotheses and outcomes

| ID | Claim | Status |
|----|--------|--------|
| **H-gdn** | Accumulated GDN T=1 numeric/state error | **Primary** — addressed by f32 SSM + dtypes |
| **H-prefill-decode** | Prefill and decode disagree on math/dtypes | **Supported then fixed** (g-dtype, softplus, f32 cast hole) |
| **H1 fused2** | Default-on mega-kernel hard-loops multi-turn | **Mitigated** by opt-in; re-test on tip: not classic H1 on 0.8B; 35B fused PASS |
| **H-async** | One-behind async corrupts state | **Refuted** on short ladder (SYNC_DECODE did not clear) |
| **H-model** | Pure CoT self-reinforcement | **Amplifier** especially 0.8B thinking / temp 0.7 |
| **H-seatbelt** | LoopBrake / low max_tokens “fixes” it | **Rejected** |
| **H-cross-turn-cache** | Stale GDN across turns | **Rejected** (design: fresh cache) |

---

## 6. Experiments

### 6.1 35B field SAR matrix (primary)

**Binary:** `1785380863` · **Engine:** `52d64de` · **MTP off** · **No pure-graph** · **No LoopBrake**

| Cell ID | Path | Temp | top_p | max_tokens | Log | Verdict |
|---------|------|------|-------|------------|-----|---------|
| **D0** | default | 0 | — | 8192 | `logs/FIELD_SAR_35B_prefill_f32.txt` | **PASS** |
| **D7 n=1** | default | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07.txt` | **FAIL** thrash |
| **D7 n=2** | default | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07_n2.txt` | **PASS** |
| **F7 n=1** | `MLX_GDN_FUSED2=1` | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07_FUSED2.txt` | **PASS** |
| **F7 n=2** | `MLX_GDN_FUSED2=1` | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07_FUSED2_n2.txt` | **PASS** |
| **F0** | `MLX_GDN_FUSED2=1` | 0 | — | 20480 | `logs/FIELD_SAR_35B_temp00_FUSED2.txt` | **PASS** |

#### D0 — default temp=0 PASS

| Metric | Value |
|--------|--------|
| Wall | ~12.6 min (20:12:15 → 20:24:50) |
| EXIT | **0** |
| Prompts | 17 → 1139 → 2352 → 3584 → 4692 (**HISTORY_OK**) |
| Gens | 2718 / 2371 / 2890 / 2874 / **5633** |
| Avg gen t/s | ~25.3 |
| Python | 4× fences, `import numpy`, usable Doppler/FFT-style code |
| Thrash | None |

#### D7 — default temp=0.7 (mixed n=1 FAIL / n=2 PASS)

**n=1 FAIL** (`FIELD_SAR_35B_temp07.txt`):

| Metric | Value |
|--------|--------|
| Wall | ~19 min then killed |
| EXIT | **143** (killed mid thrash) |
| Prompts | 17 → 1213 → 2773 → 4056 (4 turns with stats; T5 incomplete) |
| Gens | 2727 / 3349 / 2662 / 2828 (T1–4 OK) |
| Avg gen t/s | ~25.6 |
| Python | Started real code/matplotlib, then thrash |
| Thrash | **`f_s_orig*f_s_orig*…`** — max same-word run **~3101** |

**n=2 PASS** (`FIELD_SAR_35B_temp07_n2.txt`):

| Metric | Value |
|--------|--------|
| Wall | ~12 min (21:39:16 → 21:51:03) |
| EXIT | **0** |
| START | binary=`1785380863` tip=`a372276` path=default_no_fused2 MLX_GDN_FUSED2=unset |
| Prompts | 17 → 1062 → 2532 → 3706 → 5020 (**HISTORY_OK**) |
| Gens | 2360 / 2626 / 2315 / 3117 / **5222** |
| Avg gen t/s | ~26.2 |
| Python | 3× fences, `import numpy`, Doppler/FFT scaffolding |
| Thrash | **None** (`f_s_orig=0`, max same-word run **2**) |

**Interpretation:** Product-like sampling **can** hit field collapse (n=1) but **can also clear** (n=2) on the same binary. Default@0.7 is **mixed**, not solid red after n=2. Prefer more n or parity isolation only if thrash reappears; do not claim product bar fully closed.

#### F7 — fused2 temp=0.7 PASS (n=1 and n=2)

**n=1** (`FIELD_SAR_35B_temp07_FUSED2.txt`):

| Metric | Value |
|--------|--------|
| Wall | ~12 min (20:49:48 → 21:01:59) |
| EXIT | **0** |
| Env | `MLX_GDN_FUSED2=1` verified on process |
| Prompts | 17 → 1228 → 2567 → 3822 → 5343 |
| Gens | 2405 / 2723 / 2235 / 3281 / **5638** |
| Avg gen t/s | ~26.3 |
| Python | 3× fences, usable numpy Doppler/FFT demo |
| Thrash | None |

**n=2** (`FIELD_SAR_35B_temp07_FUSED2_n2.txt`):

| Metric | Value |
|--------|--------|
| Wall | ~14 min (21:51:38 → 22:05:43) |
| EXIT | **0** |
| START | binary=`1785380863` tip=`a372276` path=MLX_GDN_FUSED2=1 |
| Prompts | 17 → 1027 → 2361 → 3635 → 4892 (**HISTORY_OK**) |
| Gens | 2351 / 2848 / 3226 / 3301 / **6834** |
| Avg gen t/s | ~25.3 |
| Python | 4× fences, `import numpy`, usable code |
| Thrash | **None** (max same-word run **2**) |

#### F0 — fused2 temp=0 PASS

| Metric | Value |
|--------|--------|
| Wall | ~11 min (21:02:16 → 21:13:01) |
| EXIT | **0** |
| Prompts | 17 → 1131 → 2481 → 3659 → 4748 |
| Gens | 2502 / 2927 / 2506 / 2652 / **3785** |
| Avg gen t/s | ~26.8 |
| Python | 3× fences, usable numpy |
| Thrash | None |

#### Supporting earlier 35B runs (context, not matrix cells)

| Log | Notes |
|-----|--------|
| `FIELD_SAR_35B_default_no_brake.txt` | PASS quality n=1 on `c7685d8` (f32 decode; prefill still bf16 cast) — gens …/5169 Python |
| `FIELD_SAR_35B_f32ssm.txt` | Incomplete EXIT:143 mid T5 (user kill) — not scored as FAIL |
| `FIELD_SAR_35B_f32ssm_v2.txt` | Segfault EXIT:139 on quick restart |
| `FIELD_SAR_35B_LemonMLXE.txt` | Earlier quality multi-turn (pre full charter stamp) |

#### Throughput note

| Workload | Gen t/s |
|----------|---------|
| 35B all matrix cells | **~24–28** |
| 0.8B radar ladder (tip) | **~111–115** |
| Historical 0.8B “~63” cells | Long-context / mid ladder rates |

**~63 t/s is not the 35B expectation on this iGPU.**

---

### 6.2 0.8B radar ladder re-run on tip (proper bisect)

**Why:** Old 0.8B logs (B0, P0, FIX*, LB*, etc.) are **pre-f32 / pre-prefill-fix / LoopBrake-era** and **must not** rank current tip.

**Protocol (locked from RESULTS):**

```bash
printf '%s\n' \
  'Explain how a phased array radar steers a beam without moving antennas.' \
  'Now simplify for a non-expert.' \
  'What fails if phase synchronization drifts?' \
  'who are you?' \
  'quit' | ./build/chat mlx-community/Qwen3.5-0.8B-4bit \
  --temperature 0 --max-tokens 400
# A/B: default vs MLX_GDN_FUSED2=1
```

**Tip re-run (2026-07-29 ~21:28):**

| Cell | Log | HISTORY | EXIT | Content quality |
|------|-----|---------|------|-----------------|
| default | `logs/TIP_0.8B_radar_default.txt` | OK 25→442→860→1274 | 0 | **Residual hard CoT thrash** — line ×34: *“Wait, actually, standard phased array radar uses moving antennas…”* (classic **wrong-fact** loop). All turns hit **400**. |
| fused2 | `logs/TIP_0.8B_radar_FUSED2.txt` | OK 25→443→861→1275 | 0 | **Not classic H1 wrong-radar lock**; residual **“Wait, no”** self-correction fills budget. All turns hit **400**. |

**0.8B conclusions:**

- **Not field oracle** — confounds model CoT with engine bugs
- fused2 on tip is **better than historical H1** (no instant wrong-fact hard lock)
- **Neither path is a clean thinking multi-turn PASS** (budget filled every turn)
- Use for **A/B bisect only**

---

### 6.3 What we did **not** complete

| Item | Status |
|------|--------|
| Teacher-force multi-T ≡ T=1 ≡ fused2 | Not run |
| Formal SSM state checksum suite | Only debug env `MLX_STATE_CKSUM` |
| 35B D7 / F7 **n≥2** | Only n=1 |
| fused2 product default flip | Not done (correct) |
| MTP / pure-graph field | Out of scope |

---

## 7. Issues register (found → status)

| ID | Issue | Status |
|----|--------|--------|
| I1 | fused2 default-on multi-turn hard loops (H1) | **Mitigated** — opt-in; tip 0.8B/35B better |
| I2 | Decode `g` f32 vs prefill bf16 HIP type-pun | **Fixed** `d218c7c` |
| I3 | Unstable softplus | **Fixed** logaddexp |
| I4 | Decode SSM InT/bf16 RMW each step | **Fixed** f32 lifetime |
| I5 | Prefill cast multi-T SSM → bf16 | **Fixed** `52d64de` |
| I6 | q/k norm weight dtype mismatch | **Fixed** act-dtype materialize |
| I7 | LoopBrake as “fix” / acceptance | **Rejected + removed** |
| I8 | Suspected cross-turn GDN reuse | **Rejected** (fresh cache) |
| I9 | Async/KV as primary residual | **Refuted** on short ladder |
| I10 | **35B default @ product temp 0.7 thrash** | **OPEN** (D7 FAIL n=1) |
| I11 | 35B default @ temp 0 field | **PASS n=1** |
| I12 | 35B fused2 @ 0 and 0.7 | **PASS n=1** |
| I13 | 0.8B thinking multi-turn residual CoT | **Open residual** (model + horizon) |
| I14 | Numeric parity proof fused2 ≡ default | **OPEN** |
| I15 | Expectation ~63 t/s on 35B | **Rejected** (category error) |

---

## 8. Supervisor / consensus scores

Independent quality-reviewer + explore supervisors + Clear Thought (2026-07-29 session):

| Claim | Confidence (0–100) | Notes |
|-------|-------------------:|-------|
| Field collapse fully fixed | **~58** | Greedy + fused green; product-temp default still red once |
| Default path product-ready (temp 0.7) | **~38** | D7 FAIL n=1 |
| fused2 product-default-ready | **~28** | n=1, historical H1, no parity suite |

### Consensus paragraph

> The real engineering win on this branch is **prefill≡decode float32 SSM lifetime** plus **g-dtype / softplus / fused2 opt-in** — **not** LoopBrake. On tip `52d64de`, the field SAR bar is **achievable** under greedy default and under fused2 at both 0 and 0.7 (all n=1), but **default@0.7 still exhibited classic turn-5 code thrash**, so the product default path is **not proven ready**. fused2 is **no longer “known poison forever”** on this stack but must stay **opt-in** until n≥2 and parity evidence. Operators may run `MLX_GDN_FUSED2=1` (original polarity). Do not reintroduce LoopBrake; do not claim 63 t/s on 35B gfx1150; human merge only on PR #74.

---

## 9. Recommendations

### Product / merge

1. **Land** f32 SSM + prefill keep-f32 + g-dtype/softplus + fused2 opt-in as the correctness stack.
2. **Keep** `MLX_GDN_FUSED2` **opt-in** in code for this PR.
3. **Do not** reintroduce LoopBrake.
4. **Document** product temp 0.7; field bar should include at least one 0.7 cell going forward.
5. **Human merge only** on PR #74.

### Operator (local gfx1150 / field)

```bash
# Recommended for daily 35B while default@0.7 remains unproven at n≥2:
export MLX_GDN_FUSED2=1
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --temperature 0.7 --top-p 0.9 \
  --max-tokens 20480 --repetition-penalty 1.0 --ctx-size 32768
```

This restores **original fused2 polarity** via env without flipping the repo default.

### Next experiments (priority)

| Pri | Experiment | Why |
|-----|------------|-----|
| **P0** | 35B **default@0.7 n≥2** (and F7 n≥2) | Product hole |
| **P1** | 0.8B radar ladder (done once on tip) | fused2 health vs H1 |
| **P2** | Teacher-force / SSM checksum multi-T ≡ T=1 ≡ fused2 | Parity before default-on |
| **P3** | Only if P0–P2 green: consider code default-on fused2 with `NO_FUSED2` kill-switch | Product decision |

### Do-not-do list

1. Do not reintroduce LoopBrake as the fix or acceptance metric  
2. Do not flip fused2 to product default without n≥2 + parity  
3. Do not claim “field collapse fixed” while D7 remains FAIL  
4. Do not use 0.8B CoT as field oracle  
5. Do not use maxphrase≤5 / early-stop as PASS  
6. Do not raise max_tokens alone to “fix” thrash  
7. Do not force-push branch  
8. Do not mid-run rebuild during a matrix cell  
9. Do not expect ~63 t/s on 35B iGPU  
10. Do not treat one F7 PASS as proof fused2 always fixes sampling  

---

## 10. Artifact index

### Field / matrix logs

| Artifact | Role |
|----------|------|
| `logs/FIELD_SAR_35B_prefill_f32.txt` | D0 PASS |
| `logs/FIELD_SAR_35B_temp07.txt` | D7 FAIL thrash |
| `logs/FIELD_SAR_35B_temp07_FUSED2.txt` | F7 PASS |
| `logs/FIELD_SAR_35B_temp00_FUSED2.txt` | F0 PASS |
| `logs/CHAIN_temp07_then_fused2.txt` | D7→F7 automation chain |
| `logs/CHAIN_fused07_then_fused00.txt` | F7→F0 automation chain |
| `logs/FIELD_SAR_35B_default_no_brake.txt` | Earlier PASS (pre-prefill-fix) |
| `logs/FIELD_SAR_35B_f32ssm.txt` | Partial (killed) |
| `logs/TIP_0.8B_radar_default.txt` | Tip 0.8B default residual CoT |
| `logs/TIP_0.8B_radar_FUSED2.txt` | Tip 0.8B fused2 residual CoT |
| `logs/F32SSM_0.8B_smoke.txt` | Short smoke post f32 decode |

### Docs

| Artifact | Role |
|----------|------|
| `REAL_ISSUE.md` | Charter |
| `RESULTS_LOCAL_gfx1150.md` | Ladder history (some superseded notes) |
| `MASTER_REPORT_FIELD_MATRIX_2026-07-29.md` | Matrix-focused report |
| **`COMPLETE_WORK_REPORT_PR74.md`** | **This complete document** |

---

## 11. Reproduce commands

### 35B field (default)

```bash
{
  echo "START $(date -Iseconds) tip=$(git rev-parse --short HEAD) path=default"
  printf '%s\n' \
    "tell me about maxwell's equations" \
    "tell me about fourier analysis" \
    "how about doppler" \
    "taken together, what are these good for?" \
    "that sounds really interesting, can you give me a python implementation?" \
    "quit" \
  | ./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
      --temperature 0.7 --top-p 0.9 \
      --max-tokens 20480 --repetition-penalty 1.0 --ctx-size 32768
  echo "EXIT:$? $(date -Iseconds)"
} > docs/experiments/rocm-decode-degeneration/logs/FIELD_SAR_35B_manual.txt 2>&1
```

### 35B field (fused2)

```bash
# same as above but:
#   env MLX_GDN_FUSED2=1 ./build/chat ...
```

### 0.8B radar ladder

```bash
printf '%s\n' \
  'Explain how a phased array radar steers a beam without moving antennas.' \
  'Now simplify for a non-expert.' \
  'What fails if phase synchronization drifts?' \
  'who are you?' \
  'quit' | ./build/chat mlx-community/Qwen3.5-0.8B-4bit \
  --temperature 0 --max-tokens 400
```

### Scoring checklist

- [ ] START line has tip, binary mtime, env, temp, max_tokens  
- [ ] HISTORY_OK (prompt tokens strictly increase each turn)  
- [ ] EXIT code and wall time  
- [ ] Gen lengths and t/s  
- [ ] Turn 5: ` ```python ` / `import numpy` (35B) or coherent answers (0.8B)  
- [ ] Thrash: max same-token run, repeated n-grams, `f_s_orig*`, `synchronization synchronization`  
- [ ] Natural EOS vs max-token fill  

---

## 12. One-line conclusion

**f32 SSM prefill≡decode is the real fix class; field bar is green on default@0 and fused2@{0,0.7}; default@0.7 still showed one classic turn-5 thrash; keep fused2 opt-in in code, operators may enable it; LoopBrake is not a fix; 0.8B is bisect-only and still has residual CoT; human merge only on PR #74.**

---

*End of complete work report.*
