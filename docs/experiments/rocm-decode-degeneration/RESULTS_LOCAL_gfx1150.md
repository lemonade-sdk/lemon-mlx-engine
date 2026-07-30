# Local ROCm decode degeneration — results (gfx1150)

**Host:** Ubuntu, AMD Radeon 890M (`gfx1150`), ROCm present  
**Binary:** `build/chat` (ROCm-linked)  
**Model:** `mlx-community/Qwen3.5-0.8B-4bit` (hits `qwen35_moe` fused GDN path)  
**Date:** 2026-07-30  
**Constraints:** MTP off, pure-graph unset, eager decode, temp=0, single process  

## Pass definition used

| Term | Rule |
|------|------|
| **Hard loop** | Same 12-word phrase repeated ≥6 times in one turn, or same long line ≥6 |
| **Budget** | `--max-tokens 400` multi-turn; `128` short smoke |

## Repro (locked)

**Command (thinking on, multi-turn):**

```bash
printf '%s\n' \
  'Explain how a phased array radar steers a beam without moving antennas.' \
  'Now simplify for a non-expert.' \
  'What fails if phase synchronization drifts?' \
  'who are you?' \
  'quit' | ./build/chat mlx-community/Qwen3.5-0.8B-4bit \
  --temperature 0 --max-tokens 400
```

**Baseline (pre-fix, fused2 default ON):** hard loop from turn 1 on  
`standard phased array radar uses moving antennas to steer the beam (or…)`  
Logs: `logs/B0c_0.8B_multiturn_r1.txt`, `logs/B0_reconfirm.txt`.

**Controls:**

| Cell | Result |
|------|--------|
| Short `who are you?` + `--no-think` | **PASS** (no loop) |
| Multi-turn + `--no-think` | **PASS** (no loop) |
| Multi-turn + thinking | **FAIL** hard loop (field-like) |

## Env ladder (old binary, fused2 still default ON)

| Cell | Env | Hard loop? | Notes |
|------|-----|------------|-------|
| B0 | (none) | **Y** turns 1–4 | radar phrase |
| G1 | `MLX_GDN_NO_FUSED2=1` | turn1–3 **N**, turn4 **Y** | **H1 partial clear** |
| G2 | `MLX_GDN_NO_FUSED=1` | near-clear (maxphrase≤5) | strongest |
| C1 | `MLX_GDN_CONV_MXOPS=1` | **Y** like B0 | conv alone not root |
| P1 | `MLX_SYNC_DECODE=1` | **Y** like B0 | **H2 refuted** |
| P2 | `MLX_KV_INPLACE_OFF=1` | **Y** like B0 | **H2 refuted** |

**Outcome lock: S1 (H1)** — GDN fused recurrence path, not async/KV pipeline.

## Code fix applied (local)

Files: `src/llm/models/qwen35_moe.cpp`, `src/common/gated_delta.cpp`

1. **`gdn_fused_decode` is opt-in** via `MLX_GDN_FUSED2=1` (exact `"1"`).  
   Default path = external `rms_norm` + `gated_delta_update` (old `MLX_GDN_NO_FUSED2` behavior).  
   `MLX_GDN_NO_FUSED2` still force-disables fused2 if set with opt-in.
2. **Cast recurrence outputs** (`o` / `ns`, and compile-path `out`/`ns`) back to model `dtype` before `norm_` / cache write. Without this, non-fused2 hit monomorphic HIP `gated_rms_norm` / residual dtype mismatches or garbage text after rebuild.
3. **P0 `g` dtype (`d218c7c`)**: `compiled_beta_and_g` cast `g` to **activation (`b`) dtype**, not `a_log.dtype()`. Decode used `a_log_f32` → `g` became f32 while prefill JIT of `gated_delta_step` bound bf16 `g` (ROCm custom-kernel cache keys ignore input dtypes → type-pun decay). T=1 `gated_delta_update` now passes model `a_log_`/`dt_bias_` like prefill; fused2 still uses f32 constants for on-chip math.

## Post-fix verification

| Cell | Result |
|------|--------|
| Default (no env), short no-think | **PASS** coherent (log `FIX2_default_nothink.txt`) |
| Default, multi-turn thinking | turn1 improved; turns 2–4 still soft/hard loop on “Wait, I need to check…” (`FIX2_default_mt.txt`) — **better than baseline turn1 radar lock**, not fully clean |
| `MLX_GDN_FUSED2=1`, multi-turn | still hard loops turns 2–4 (`FIX2_FUSED2_on_mt.txt`) — opt-in keeps old kernel for kernel debugging |

## History integrity gate (required)

Prompt-token growth must prove multi-turn re-prefill (ChatSession history).

| Run | Prompt tokens | Verdict |
|-----|---------------|---------|
| Pre-fix B0c / G1 | 25→442→860→1274 | HISTORY_OK |
| Early FIX2_default_mt | 25→18→18→14 | **HISTORY_BROKEN** — do not use for residual ranking |
| **HIST_gate_default_mt** (post ab1b518) | **25→443→861→1275** | **HISTORY_OK** |
| R1_nonradar / R1_budget2048 | 17→… growing | HISTORY_OK |

**ChatSession design:** fresh GDN/KV cache each turn; residual is **not** cross-turn SSM state.

## Residual multi-turn (post fused2 opt-in, HISTORY_OK only)

| Protocol | Result |
|----------|--------|
| Radar multi-turn, default (fused2 off), max=400 | **PASS** hard-loop rule (maxphrase≤5); soft “Wait…” still in CoT (`HIST_gate_default_mt.txt`) |
| Radar + `MLX_GDN_FUSED2=1`, max=400 | HISTORY_OK; this re-run maxphrase≤4 (`HIST_gate_FUSED2_on_mt.txt`) — **flaky vs earlier B0**; do not un-lock H1 from one green |
| **P0 after g-dtype** radar default (`P0_gdtype_radar_mt.txt`) | HISTORY_OK 25→1275; **turn1 hard=N**, correct “phased array steers without moving antennas” fact; soft Wait near budget; t2–4 hard Wait reseeds |
| **P0 radar `--no-think`** (`P0_gdtype_radar_nothink.txt`) | **PASS** coherent answers |
| R1 France multi-turn, max=400 | **FAIL** hard “Fact Check: capital of France is Paris” loops (`R1_nonradar_mt_think.txt` / `P0_gdtype_nonradar_mt.txt`) |
| R1 + max=2048 | **WORSE** — fills entire budget with Fact Check loops (gens all 2048) (`R1_budget2048_mt.txt`) |
| R1 + `--repetition-penalty 1.15`, max=400 | **MUCH BETTER** — turn1 finishes `</think>` + answer; later turns softer/partial (`R1_reppenalty_mt.txt`) |
| R2 math multi-turn | **PASS** (`R2_math_mt_think.txt`) |
| Multi-turn `--no-think` | **PASS** (control) |

### Residual root-cause rank (updated)

| Rank | Cause | Confidence |
|------|-------|------------|
| **1** | Thinking CoT **self-reinforcement** on 0.8B (temp=0); incomplete mid-CoT re-seeded via history | **High** |
| **2** | Tight/mid max_tokens amplifies unfinished CoT (400 truncates; 2048 extends the loop) | **High** |
| **3** | Remaining GDN numeric (fused2 RMSNorm parity; optional further HIP) | **Medium-low** after P0 g-dtype; H1 still for fused2 catastrophic class |
| **Refuted** | Cross-turn GDN cache reuse | Design + history gate |
| **Refuted** | H2 async/KV as primary residual | P1/P2 ladder |
| **Fixed** | Decode `g` f32 vs prefill bf16 HIP type-pun | `d218c7c` |

### Residual mitigations (product)

1. **`LoopBrake` in ChatSession** (`64387f8`) + **`generate_text`** (`0564ffc`) — covers server chat/completions.  
   Defaults after `0564ffc`: phrase ≥3× (32–120 chars), same-line ≥5, token n-gram ≥3× (8–16), word n-gram freq ≥5 in trailing window.  
2. **`--repetition-penalty ~1.1–1.2`** still helps as sampling-side option (measured on R1).  
3. Do **not** “fix” residual by raising max_tokens alone (worsens R1 without brake).  
4. Keep **fused2 opt-in** for H1 catastrophic class until more fused2 green cells.  
5. Optional: strip unfinished think blocks before history append (experiment).

### Loop brake local verify (HISTORY_OK)

| Cell | Result |
|------|--------|
| France multi-turn (`64387f8`) | t2–4 **hard=N**; gen cut ~400→152/118/113 (`LB_france_mt.txt`) |
| France multi-turn tighter (`0564ffc`) | **all turns hard=N** max12≤4; gens 312/138/104/100 (`LB2_france_mt.txt`) |
| Radar multi-turn default | all turns **hard=N** max12≤4 (`LB_radar_mt.txt`) |
| Radar + `MLX_GDN_FUSED2=1` | hard=N max12≤5 but fills 400 soft Wait (`LB_FUSED2_radar_mt.txt`, `F3_FUSED2_radar_mt.txt`) — **do not re-default fused2** |
| Fused2 France ST after g/beta InT cast | coherent; gen hits 200 cap; hard=N (`F4_FUSED2_france_st.txt`) |
| **F5 multi-cell (post full stack)** | see table below — **hard=N**; France gens ~match default; **still hold opt-in for merge** |

### F5 multi-cell (post softplus + InT + LoopBrake, HISTORY_OK)

| Cell | Env | hard-loop | gens | Log |
|------|-----|-----------|------|-----|
| France MT | default | **N** | 184/328/152 | `F5_default_france_mt.txt` |
| France MT | `MLX_GDN_FUSED2=1` | **N** | 184/327/162 | `F5_FUSED2_france_mt.txt` |
| Radar 2-turn | `MLX_GDN_FUSED2=1` | **N** | 303/333 | `F5_FUSED2_radar_2t.txt` |

**Decision:** multi-cell hard-loop rule green for fused2 on 0.8B **after** full numeric stack, but **do not flip product default in this PR** (H1 was catastrophic historically; re-default = follow-up human decision).

### Field-size (35B A3B) smoke — gfx1150

**Model:** `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` (default GDN path; fused2 off; MTP off; ~19.2 GB active via GTT)

| Cell | Result |
|------|--------|
| no-think multi-turn who/2+2 | **PASS** coherent (`S35_nothink_smoke.txt`) |
| thinking capital of France max=128 | **PASS** finishes `</think>` + answer Paris (`S35_think_smoke.txt`) |

0.8B residual CoT severity does **not** reproduce on this short 35B smoke. Full radar multi-turn 35B not required to keep fused2 opt-in.

## Remaining work (not closed)

1. ~~History confound on FIX logs~~ — gate defined; HIST_gate PASS.  
2. ~~P0 `g` activation-dtype / prefill–decode HIP type-pun~~ — `d218c7c`.  
3. ~~Residual CoT loop brake (chat + server)~~ — `64387f8` / `0564ffc`.  
4. ~~Fused2 RMSNorm/softplus + g/beta InT round-trip + Dk%32 guard~~ — still **opt-in**.  
5. ~~Field-size short smoke~~ — 35B A3B PASS (`S35_*`).  
6. ~~Multi-cell fused2 hard-loop re-check~~ — F5 France/radar hard=N, gens ~parity default; **opt-in held for this PR**.  
7. ~~Stable softplus on default GDN path~~ — `logaddexp`; smoke `SP_default_*` hard=N.  
8. Optional follow-up: re-default fused2 ON after human review of F5 + field load.  
9. ~~LoopBrake kill-switch~~ — `MLX_LOOP_BRAKE_OFF=1`.  
10. **q-norm `1/D` vs `1/√D`** A/B only if product asks (paths already consistent).

## Supervisor consensus (quintuple)

| Lens | Verdict |
|------|---------|
| Code audit (explore) | Fused2 hold opt-in; gaps softplus vs compile + residual soft Wait |
| QA protocol | HISTORY gate + hard-loop defs; 35B short PASS |
| Senior ROCm | g/beta InT + Dk%32 smallest next fused deltas; no default flip |
| Planning | Product residual closed via LoopBrake; fused2 perf path opt-in |
| Local science | H1 mitigated; 0.8B CoT ≠ 35B short smoke |

## Bottom line

**H1 (default fused2 hard-loop): mitigated** — opt-in + dtype cast (`ab1b518`).  
**P0 decode `g` type-pun: fixed** (`d218c7c`).  
**Residual CoT: braked** (chat + `generate_text`/server).  
**Fused2 numerics: improved, still opt-in** — RMSNorm/softplus + g/beta InT + Dk%32; soft Wait multi-turn still weaker than default.  
**Default-path softplus: stable** — `logaddexp` (torch-like); `SP_default_*` France/radar hard=N.  
**35B A3B short smoke: PASS** on default path (gfx1150).  
**Branch:** `fix/rocm-gdn-fused2-optin` off `origin/main` — **human merge only**.

## Standard models for local tests (locked)

| Role | Model id | Notes |
|------|----------|--------|
| Fast / ladder | `mlx-community/Qwen3.5-0.8B-4bit` | Load smoke + residual A/B/C |
| Field 35B | **`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`** | Not unsloth; MTP weights present, head **skipped** unless `MLX_LOAD_MTP_HEAD=1` |

### Canonical 35B chat invocation (product defaults: MTP off, pure-graph unset)

```bash
./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --max-tokens 20480 \
  --repetition-penalty 1.0 \
  --ctx-size 32768
```

HF cache: `~/.cache/huggingface/hub/models--LemonMLXE--Qwen3.6-35B-A3B-MTP-mlx-4bit/` (~20G weights). Refresh: `hf download LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`.

### Smoke 2026-07-30 (gfx1150, branch fix/rocm-gdn-fused2-optin)

| Cell | Result |
|------|--------|
| 0.8B load + who are you --no-think | PASS (`SMOKE_0.8B_load.txt`) active~889MB |
| 35B LemonMLXE + user flags --no-think who are you / 2+2 | PASS (`SMOKE_35B_LemonMLXE_nothink.txt`) active~18.2GB, MTP head skipped, answers coherent |

### Field SAR multi-turn (default path, MTP off)

| Cell | Result |
|------|--------|
| 0.8B multi-turn science (Maxwell→Fourier…) | HISTORY_OK; **hard=N** (max12≤5); soft self-reinforcement still present (`FIELD_SAR_0.8B_default.txt`) |
| 35B LemonMLXE Maxwell open question | **PASS quality** — finishes `</think>` + structured Maxwell answer (`FIELD_SAR_35B_LemonMLXE.txt`); MTP skipped |

**LoopBrake kill-switch:** `MLX_LOOP_BRAKE_OFF=1` disables product brake (ChatSession + `generate_text`).

