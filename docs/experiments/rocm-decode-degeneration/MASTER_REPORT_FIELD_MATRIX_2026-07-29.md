# Master report — ROCm GDN field matrix (35B SAR)

**Date:** 2026-07-29 (local gfx1150)  
**Host:** AMD Radeon 890M (`gfx1150`), 8 CUs, ROCm  
**Branch:** `fix/rocm-gdn-fused2-optin`  
**Engine tip under test:** `52d64de` (prefill+decode f32 SSM); docs tip `a372276`  
**Binary:** `build/chat` mtime epoch `1785380863` (same binary for all matrix cells)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**MTP:** off (head skipped)  
**Pure-graph:** off  
**LoopBrake:** removed (not used)  
**PR:** [#74](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/74) (human-merge only)

---

## 1. Executive summary

| Question | Answer |
|----------|--------|
| Is multi-turn GDN field collapse still real? | **Yes, residual risk** — default@0.7 n=1 thrash (`f_s_orig*…`); **n=2 PASS** same path. |
| Does f32 SSM + prefill keep-f32 help? | **Yes** — default @ temp=0 full SAR **PASS**; default@0.7 can PASS too (n=2). |
| Is fused2 (`MLX_GDN_FUSED2=1`) still poison? | **Not on this matrix** — fused2 **PASS** at temp=0 and 0.7 (**F7 2/2**). |
| Should fused2 become product default tonight? | **Not yet** — keep opt-in; product bar prioritizes **default** path; need parity + more green before flip. |
| Is ~63 t/s expected on 35B here? | **No** — measured ~25–27 gen t/s; ~63 is 0.8B-class on this host. |
| Did temp alone fix collapse? | **No** — temp=0 default PASS; temp=0.7 **mixed** (FAIL then PASS). Sampling interacts with residual fragility. |

**Headline:** On tip `52d64de`, field SAR is achievable on default and fused. Default@0.7 is **mixed 1/2** (not closed, not solid red). Fused@0.7 is **2/2 PASS**. Keep fused2 **opt-in**. No LoopBrake.

---

## 2. Problem statement (REAL_ISSUE)

**Not the issue:** LoopBrake early-stop, CI “2+2”, maxphrase≤5 after truncation.

**The issue:** On ROCm, long multi-turn hybrid Qwen GDN chat can stay coherent for several turns then **collapse into token/phrase thrash** (field class: endless `synchronization…` or, this matrix, `f_s_orig*f_s_orig*…`) especially on a large code-generation turn.

**Session design:** Each turn full re-prefill + **fresh** KV/GDN cache. Collapse is **not** cross-turn SSM reuse; it is accumulated T=1 decode / prefill≡decode numerics + long generation horizon (and sampling).

**PASS definition:** Coherent turns 1–4; turn 5 produces **usable Python** (or real plan+code), finishes or hits budget with real content — **no** endless single-token/phrase thrash. No seatbelt required.

**Field prompts:**

1. tell me about maxwell's equations  
2. tell me about fourier analysis  
3. how about doppler  
4. taken together, what are these good for?  
5. that sounds really interesting, can you give me a python implementation?

---

## 3. Code under test (what landed)

| Commit | Change |
|--------|--------|
| `c7685d8` | float32 SSM lifetime (decode); act-dtype q/k norm; softplus f32 |
| `52d64de` | **Prefill keep SSM float32** (stop casting multi-T state back to bf16); f32 softplus on NO_FUSED path |
| `a372276` | Docs: field PASS stamp for prefill f32 |
| `ab1b518` (earlier) | `gdn_fused_decode` **opt-in** via `MLX_GDN_FUSED2=1` (was default ON; demoted after hard loops) |
| `36f83f5` | LoopBrake **removed** |

### Decode path selection (`qwen35_moe.cpp`)

| Path | How to select | What runs |
|------|----------------|-----------|
| **Default** | (no env) | External `rms_norm` + `gated_delta_update` (HIP step) |
| **fused2** | `MLX_GDN_FUSED2=1` | `gdn_fused_decode` (norm + beta/g + recurrence mega-kernel) |
| **mxops fallback** | `MLX_GDN_NO_FUSED=1` | compile inline recurrence (slow/portable) |

**Originally fused?** Yes — fused2 was introduced as default decode (`8d7c95b` era), then flipped to opt-in after local multi-turn hard loops. “Fused” is also overloaded: prefill multi-T and default `gated_delta_update` are already fused kernels; **fused2** means the FlashQLA-style mega-kernel only.

---

## 4. Experimental matrix (locked)

Same binary, same model, same 5 prompts, MTP off, no pure-graph, no LoopBrake, no mid-run code changes.

| Cell | Path | Temp | top_p | max_tokens | Log |
|------|------|------|-------|------------|-----|
| D0 | default | 0 | — | 8192 | `logs/FIELD_SAR_35B_prefill_f32.txt` |
| D7 | default | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07.txt` |
| F7 | `MLX_GDN_FUSED2=1` | 0.7 | 0.9 | 20480 | `logs/FIELD_SAR_35B_temp07_FUSED2.txt` |
| F0 | `MLX_GDN_FUSED2=1` | 0 | — | 20480 | `logs/FIELD_SAR_35B_temp00_FUSED2.txt` |

**Note:** D0 used max_tokens 8192; others 20480. D0 finished naturally at ~5.6k so budget difference did not create the PASS. D7 thrash would also have failed at 8192.

---

## 5. Results

### 5.1 Scoreboard

| Cell | Turns | HISTORY | Turn-5 Python | Thrash | EXIT | Avg gen t/s | Verdict |
|------|-------|---------|---------------|--------|------|-------------|---------|
| **D0** default@0 | 5 | OK 17→…→4692 | Yes (4 fences, numpy) | None | 0 | 25.3 | **PASS** |
| **D7 n=1** default@0.7 | 4 (+T5 mid) | OK 17→…→4056 | Started then thrash | **`f_s_orig` ×3101 run** | 143 (killed) | 25.6 | **FAIL** |
| **D7 n=2** default@0.7 | 5 | OK 17→…→5020 | Yes (3 fences, numpy) | None | 0 | 26.2 | **PASS** |
| **F7 n=1** fused2@0.7 | 5 | OK 17→…→5343 | Yes (3 fences, numpy) | None | 0 | 26.3 | **PASS** |
| **F7 n=2** fused2@0.7 | 5 | OK 17→…→4892 | Yes (4 fences, numpy) | None | 0 | 25.3 | **PASS** |
| **F0** fused2@0 | 5 | OK 17→…→4748 | Yes (3 fences, numpy) | None | 0 | 26.8 | **PASS** |

**D7 product path aggregate:** **1 FAIL / 1 PASS** (mixed). Not deterministic red after n=2; not product-closed.  
**F7 fused2@0.7 aggregate:** **2/2 PASS**.

### 5.2 Detail — D0 default temp=0 (PASS)

- START 20:12:15 → EXIT:0 20:24:50 (~12.6 min)  
- Gens: 2718 / 2371 / 2890 / 2874 / **5633** @ ~24.3–25.8 t/s  
- Natural EOS ≪ 8192; Maxwell→…→Python with Doppler/FFT-style content  
- Establishes: **f32 SSM + prefill keep-f32** can clear field bar under greedy sampling  

### 5.3 Detail — D7 default temp=0.7

#### n=1 FAIL (`logs/FIELD_SAR_35B_temp07.txt`)

- START 20:30:40; turns 1–4 coherent (2727 / 3349 / 2662 / 2828)  
- Turn 5: real Python/matplotlib structure began, then locked into  
  `f_s_orig*f_s_orig*f_s_orig*…` (1550 pair hits; max same-word run **3101**)  
- Killed EXIT:143 at 20:49:38 (no point finishing 20480 thrash budget)  

#### n=2 PASS (`logs/FIELD_SAR_35B_temp07_n2.txt`)

- START 21:39:16 → EXIT:0 21:51:03 (~12 min); binary=`1785380863` tip=`a372276` path=default_no_fused2  
- Prompts **17 → 1062 → 2532 → 3706 → 5020** HISTORY_OK  
- Gens **2360 / 2626 / 2315 / 3117 / 5222** @ ~25.4–26.7 t/s  
- Thrash: **none** (`f_s_orig=0`, `synchronization=0`, max same-word run **2**)  
- Turn 5: usable Python (`import numpy`, matplotlib, Doppler/FFT scaffolding; 3× ` ```python ` fences); natural EOS ≪ 20480  
- Establishes: D7 n=1 thrash is **not** a deterministic always-fail under product sampling; default@0.7 is **mixed**  

### 5.4 Detail — F7 fused2 temp=0.7 (PASS n=1 and n=2)

#### n=1 (`logs/FIELD_SAR_35B_temp07_FUSED2.txt`)

- START 20:49:48 → EXIT:0 21:01:59 (~12 min)  
- Env verified: `MLX_GDN_FUSED2=1` on process  
- Gens: 2405 / 2723 / 2235 / 3281 / **5638** @ ~25.7–27.2 t/s  
- No thrash; usable numpy Doppler/FFT demo; natural finish  

#### n=2 (`logs/FIELD_SAR_35B_temp07_FUSED2_n2.txt`)

- START 21:51:38 → EXIT:0 22:05:43 (~14 min); binary=`1785380863` tip=`a372276` path=MLX_GDN_FUSED2=1  
- Prompts **17 → 1027 → 2361 → 3635 → 4892** HISTORY_OK  
- Gens **2351 / 2848 / 3226 / 3301 / 6834** @ ~22.3–27.2 t/s  
- Thrash: **none**; max same-word run **2**; usable Python (4× fences, `import numpy`); natural EOS  
- **Same temp/top_p/max as D7** — only decode path differs  

### 5.5 Detail — F0 fused2 temp=0 (PASS)

- START 21:02:16 → EXIT:0 21:13:01 (~11 min)  
- Gens: 2502 / 2927 / 2506 / 2652 / **3785** @ ~26.2–27.5 t/s  
- No thrash; usable numpy; natural finish  
- Matches D0 quality class under greedy + fused2  

### 5.6 Throughput

| Workload | Observed gen t/s |
|----------|------------------|
| 35B LemonMLXE all cells | **~24–28** (stable) |
| 0.8B ladder (prior logs) | **~60–120** (includes ~63) |
| fused2 vs default (this matrix) | fused ~**+0.5–1.5 t/s** avg (noise / n=1; not 2×) |

**Conclusion:** Expectation of ~63 t/s on 35B on this iGPU was a **category error** (0.8B rates). Temperature does not drive the gap. fused2 is not a throughput miracle here; quality is the question.

---

## 6. Interpretation

### 6.1 What is fixed / improved

1. **Prefill≡decode SSM dtype lifetime (f32)** — primary engineering fix this cycle; D0 PASS supports it.  
2. **fused2 not categorically broken** on current tip for this field sequence (F0+F7 PASS).  
3. **LoopBrake correctly rejected** as product success metric.  

### 6.2 What is not fully closed

1. **D7 mixed (1 FAIL / 1 PASS)** — product temperature path still thrash-capable (n=1) but can also clear (n=2). Could be:  
   - residual GDN numeric fragility under longer/stochastic trajectories, and/or  
   - pure sampling/model self-reinforcement amplified at 0.7  
2. **F7 n=2 PASS** — fused@0.7 is 2/2 green on this tip (still not a code-default flip without parity).  
3. **No formal teacher-force / state checksum** multi-T ≡ T=1 ≡ fused2 (deferred while D7 is not solid red after n=2).  
4. **fused2 still opt-in in code** — correct until repeated green + parity proof.  
5. **Historical fused2 hard loops** (0.8B radar thinking ladder) re-run on tip separately (see TIP_0.8B logs).  

### 6.3 Path vs sampling (working model)

```
                temp=0              temp=0.7
default         PASS (D0)           MIXED (D7: FAIL n=1, PASS n=2)
fused2          PASS (F0)           PASS 2/2 (F7)
```

- Greedy + default works on this tip.  
- Product sampling + default: thrash once, clean once — **not deterministic FAIL**.  
- fused2@0.7: **2/2 PASS**.  

**Do not overclaim** “fused2 fixes sampling” or “default is broken forever.” Claim: “after f32 SSM work, field bar is achievable on default and fused; product@0.7 still has residual thrash risk (seen once on default).”

### 6.4 Why thrash phrase changed

Older field notes emphasized `synchronization synchronization…`. This matrix’s default@0.7 thrash was **`f_s_orig` multiplication spam** mid-print. Same **class** (decode lock into high-probability token loop), different surface string — detection should be general (max same-token run / n-gram), not one phrase.

---

## 7. Recommendations

### Product / merge (PR #74)

1. **Ship f32 SSM + prefill keep-f32** as the real correctness work (already on branch).  
2. **Keep `MLX_GDN_FUSED2` opt-in** for merge unless follow-up n≥2 and 0.8B ladder stay green.  
3. **Do not reintroduce LoopBrake** as the fix.  
4. **Document** product default temp 0.7; field acceptance should include at least one 0.7 cell.  

### Next experiments (no rush, no agent thrash)

| Priority | Experiment |
|----------|------------|
| P0 | ~~D7 n=2~~ **PASS**; ~~F7 n=2~~ **PASS**; optional D7 n=3 only if policy wants ≥2 green default@0.7 |
| P1 | 0.8B multi-turn thinking ladder default vs fused2 on this tip (TIP_* logs exist) |
| P2 | Teacher-force / SSM checksum: prefill multi-T vs T=1 default vs T=1 fused2 — **if thrash reappears** or before fused2 code default-on |
| P3 | Only if default@0.7 stable green + parity: consider default-on fused2 with kill-switch env |

### Performance (separate track)

- 35B on gfx1150 ≈ **25–27 t/s** is the measured band.  
- Optimizations: MTP (off today), pure-graph, fused2 launch reduction — after correctness.  

---

## 8. Process notes (this session)

- Continuous 5-minute agent loop **cancelled** after field PASS stamp (avoid mid-run binary thrash).  
- Default@0.7 killed mid-thrash deliberately (EXIT 143).  
- Fused@0.7 then fused@0 chained automatically; env `MLX_GDN_FUSED2=1` verified on PIDs.  
- No engine source edits during the matrix.  

---

## 9. Artifact index

| Artifact | Role |
|----------|------|
| `logs/FIELD_SAR_35B_prefill_f32.txt` | D0 PASS |
| `logs/FIELD_SAR_35B_temp07.txt` | D7 n=1 FAIL thrash |
| `logs/FIELD_SAR_35B_temp07_n2.txt` | D7 n=2 PASS |
| `logs/FIELD_SAR_35B_temp07_FUSED2.txt` | F7 n=1 PASS |
| `logs/FIELD_SAR_35B_temp07_FUSED2_n2.txt` | F7 n=2 PASS |
| `logs/FIELD_SAR_35B_temp00_FUSED2.txt` | F0 PASS |
| `logs/CHAIN_temp07_then_fused2.txt` | D7→F7 chain |
| `logs/CHAIN_fused07_then_fused00.txt` | F7→F0 chain |
| `REAL_ISSUE.md` | Charter |
| `RESULTS_LOCAL_gfx1150.md` | Prior ladder / residual notes |
| This file | Master matrix report |

---

## 10. One-line conclusion

**f32 SSM prefill≡decode is the real fix class; field bar is green on default@0 and fused2@{0,0.7}; default@0.7 still showed one classic turn-5 thrash — keep fused2 opt-in, re-test product temp, don’t claim 63 t/s on 35B iGPU.**
