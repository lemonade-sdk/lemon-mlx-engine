# MTP Optimality Plan (fix/mtp-stream-p0)

**Date:** 2026-08-01  
**Model bar:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` on gfx1150  
**Branch tip at plan write:** see git log on this file  
**Goal:** make MTP either beat or auto-fallback vs full-fuse eager (~26 t/s), without reintroducing LoopBrake or dual-loading two 35B processes.

---

## 1. Online reference findings (MLX / mlx-lm / ecosystem)

### 1.1 mlx-lm native MTP (primary reference)

| Item | Detail |
|------|--------|
| PR | [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990) — “Native MTP speculative decoding (Qwen3.5/3.6)” |
| CLI | `mlx_lm.generate --mtp` / `mlx_lm.server --mtp` (opt-in; mirrors separate-drafter style flags) |
| Core API | `mtp_generate_step()` in `generate.py`: draft via MTP head, verify with backbone `[confirmed, draft]`, GDN `n_confirmed` snapshot + rollback on reject |
| Head shape | One extra transformer layer: fuse `pre_fc_norm(h_t)` + `pre_fc_norm(embed(t+1))` → MTP layer → **shared `lm_head`** |
| Checkpoint | Must **preserve** `mtp.*` weights; historical sanitize stripped them (`"mtp." not in k`) |
| Dense result | Qwen3.5/3.6-27B 4-bit M4 Pro: ~15.3 → ~23–24.6 t/s (**~1.5–1.57×**), accept ~80–88% (temp=0) |
| MoE result | Qwen3.5-35B-A3B 4-bit M4 Pro: 85.3 → 87.9 t/s (**~1.04×**); M2 Ultra 8-bit ~1.11× — **marginal** |
| Server | Dynamic MTP when solo request; batch path falls back / switches |
| Follow-ups | Probabilistic accept `min(1, p_t/p_d)` for temp>0; residual sampling on reject; exclude `mtp.fc` (sometimes all `mtp.*`) from quant debates |

Community notes on MoE:

- Bandwidth model (PR discussion): `speedup ≈ (1+p) / (β+δ)` with `β = T_verify / T_eager`, `δ = T_mtp_head / T_eager`.
- On MoE, multi-token **verify tends to touch different experts** → near-linear memory traffic (unlike dense, where K-token verify reuses the same weights).
- Debate on “physically impossible for single-stream MoE MTP” vs measured small wins (~1.05–1.18× with good accept). Consensus: **dense wins big; MoE wins only when verify is cheap relative to baseline and accept is high**.

Related:

- [mlx-vlm#981](https://github.com/Blaizzy/mlx-vlm/issues/981): mlx-lm server also has **classic** speculative decode: `--draft-model` + `--num-draft-tokens`.
- [EAGLE-3 prototype discussion](https://github.com/ml-explore/mlx-lm/discussions/890): hidden-state draft without separate full model (different path).
- [mlx-serve](https://github.com/ml-explore/mlx/discussions/654) (community): claims native MTP heads for Qwen 3.6 ~1.1–2.1× depending on workload.

### 1.2 OptiQ / Apple Silicon practice

Source: [mlx-optiq MTP docs](https://mlx-optiq.com/docs/mtp)

| Finding | Implication for us |
|---------|-------------------|
| Enable with `--mtp` for bundled Qwen MTP head | Same product shape as our `--use-mtp` |
| **γ (depth) = 1 default and optimal on Metal** | K-token verify scales ~linearly with K; γ≥2 often loses |
| Adaptive depth (raise/lower K) **lost 4–17%** | Do not invest in fancy adaptive until base step is faster than eager |
| Skip MTP when base already fast (0.8B/2B regress) | **Auto-disable / fallback** is first-class product behavior |
| Qwen3.6-27B dense ~1.40× greedy | Ceiling for *dense*; our target is MoE 35B-A3B on ROCm |
| MTP head in quants: 4-bit proj + BF16 final where needed | Align with quant policy; avoid accidental full dequant thrash |

### 1.3 Broader MTP (context only)

- Training-time multi-token targets vs inference drafter: [Raschka gallery](https://sebastianraschka.com/llm-architecture-gallery/mtp/), Nemotron / Gemma-4 assistant drafters (external small model).
- Gemma-4 uses **separate `-assistant` drafter**, not bundled MTP head — different codepath (ignore for Qwen3.6 MoE bar).

---

## 2. Our stack vs reference (gap list)

### 2.1 How we enable MTP

| Ours | mlx-lm / OptiQ |
|------|----------------|
| `MLX_LOAD_MTP_HEAD=1` (load/build head) | Weights present + `--mtp` |
| `--use-mtp` (chat/server) | `--mtp` |
| `--n-draft N` (chat, default **1**); server `--n-draft-tokens` default **3** | Implicit depth often **1** draft (+ confirm); OptiQ γ=1 |
| `MTP_DEBUG=1`, `MTP_TIMING=1` | Engine counters `drafts_accepted/attempted` |
| `StreamGuard` on `mtp_speculative_step` | N/A (Metal default stream) |
| Quant fuse: `MLX_ENABLE_QUANT_FUSE` (+ `_GDN`) orthogonal | Weight quant fine with MTP |

Semantics (our engine, from README): `n_draft_tokens` = block size = **d0** (already sampled trunk token, trusted) + **N−1** drafted tokens.  
So `--n-draft 2` ⇒ **1** true draft slot (closest to OptiQ γ=1).  
`--n-draft 4/6` ⇒ 3/5 draft slots (OptiQ-discouraged depth).

### 2.2 Call chain (code)

Primary implementation: `/home/antmi/lemon-mlx-engine/src/common/generate.cpp` → `TokenIterator::mtp_speculative_step()`

1. **Draft (serial):** for `i = 1 .. n_draft-1`:  
   `embed(prev)` → `MTPHead(hidden, embed, mtp_cache)` → `apply_output_norm` → **`apply_lm_head_fn` (full vocab)** → argmax → next prev.  
   One host sync after the chain (`mx::eval` on draft token array).
2. **Verify:** trunk `call_fn` on full draft sequence `[d0..d_{K-1}]` with **`capture_spec=true`** on all Mamba/GDN caches when present.
3. **Accept:** host scan of trunk argmax vs drafts; stop at first mismatch; set next `y_`.
4. **Commit:** prefer `rollback_spec` / position trim; else restore snapshot + re-run prefix if intermediates missing; clear `capture_spec`.
5. **Adaptive:** `record_acceptance()` writes history; **`current_draft_count()` returns fixed `n_draft_tokens_` only** (stub).

Supporting:

- Head: `include/mlx-lm/llm/models/mtp_head.h`, `src/llm/models/mtp_head.cpp`, MoE layer `mtp_moe.cpp`
- Load: `qwen35_moe.cpp` / `qwen35.cpp` — **dequantizes MTP weights into dense-ish maps** (`[MTP] Dequantized N weights total`)
- Fallback if no head fn / null head: plain `step()` (not TPS-based auto-disable)

### 2.3 Field baseline (this branch experiments)

From `docs/experiments/mtp-stream-p0/TPS_probe_*.txt` (256 gen tokens, Maxwell-style prompt, full fuse + MTP load where applicable):

| Config | Generation t/s | Notes |
|--------|----------------|-------|
| no MTP (full fuse) | **26.13** | Target bar |
| MTP `--n-draft 2` | **6.05** | ~4.3× slower |
| MTP `--n-draft 4` | **6.76** | slightly better than 2; still ~3.9× slower |
| MTP `--n-draft 6` | **5.38** | worse; deeper draft thrash |

Per-step timing (`MTP_TIMING` / `[mtp-t]`), typical n_draft=2 steady state:

- **draft ≈ 150–165 ms** (serial MoE MTP + full lm_head)
- **verify ≈ 75–230 ms** (K-token GDN trunk + capture_spec; high variance)
- **commit ≪ 1 ms**
- **total ≈ 230–390 ms / speculative step** while often emitting only **1–2 tokens** → **~3–8 t/s** order-of-magnitude matches measured gen t/s

Acceptance: many steps `accepted=0` or `1` of 1 draft slot; longer n_draft frequently wastes verify on rejected suffixes.

### 2.4 Gap list (vs healthy mlx-lm / OptiQ path)

| # | Gap | Severity |
|---|-----|----------|
| G1 | **No auto-disable** when MTP gen t/s ≪ eager | Product / P0 |
| G2 | **Verify cost** multi-token hybrid GDN + `capture_spec` intermediates >> single-token eager step | P0 dominate |
| G3 | **Draft cost**: serial MoE MTP steps + **full lm_head** each draft token | P0 dominate |
| G4 | **MTP dequant at load** (dense head mats / dequant path) — bandwidth + memory; diverges from mlx-lm quant_predicate debates | P1 |
| G5 | **Hard host `mx::eval` / `.item` barriers** in draft+verify | P1 |
| G6 | **`current_draft_count` adaptive stubbed** (history recorded but unused) | P1 (after G2/G3) |
| G7 | Default server `n_draft_tokens=3` may exceed OptiQ γ=1 sweet spot | P1 policy |
| G8 | MoE structural bandwidth: multi-token verify ≠ free reuse of weights | P2 / physics ceiling |
| G9 | No microbench table isolating verify(K) vs K×eager T=1 in `MICROBENCH.md` | Process (step B) |
| G10 | Stream(cpu) class fixed via StreamGuard — **do not re-litigate** without new evidence | Done |

---

## 3. Ranked fixes

### P0 (do first; smallest changes that attack dominate cost or prevent user footgun)

| ID | Fix | Rationale | Done? |
|----|-----|-----------|-------|
| **P0-1** | **Auto-disable / fallback to eager** when MTP is slower (runtime window or documented policy + flag): e.g. after N steps if `(tokens/time) < eager_baseline * 0.9`, set `use_mtp_=false` and finish with plain `step()` | OptiQ “skip when base already fast”; our field is **4× regression** — must not ship as default win | **no** |
| **P0-2** | **Document + enforce n_draft policy**: prefer `--n-draft 2` (γ≈1); warn or cap higher N until verify is sub-linear | OptiQ γ=1; our n_draft 6 is slowest | **no** |
| **P0-3** | **Reduce verify / `capture_spec` cost**: avoid capture when n_draft==2 and accept path always uses simple trim; skip intermediate capture if rollback rare; measure `MTP_NO_INTERMEDIATES` vs default | capture_spec + multi-token GDN is named dominate | **no** |
| **P0-4** | **Draft path: avoid unnecessary full lm_head work / sync** (shared head kernel, keep argmax on device longer, fuse draft steps if safe) | draft ~160 ms/step at n_draft=2 | **no** |

### P1

| ID | Fix | Done? |
|----|-----|-------|
| **P1-1** | Wire **`current_draft_count()`** from `accept_history_` (raise after full accept, lower on reject) **only after** P0 proves step cost can beat eager at γ=1 | **no** |
| **P1-2** | **Keep MTP quantized** (match backbone) instead of full dequant at load if accept rate holds; or BF16-only `mtp.fc` like mlx-lm quant_predicate | **no** |
| **P1-3** | Collapse host barriers: fewer `eval`/`item` per step; align residual-sampling / accept with mlx-lm for temp>0 later | **no** |
| **P1-4** | Align server default `--n-draft-tokens` with chat γ≈1 policy | **no** |

### P2

| ID | Fix | Done? |
|----|-----|-------|
| **P2-1** | Accept MoE **structural ceiling** study: if β≈K and p≪1, MTP cannot win single-stream; publish “eager recommended on gfx1150 MoE” | **no** |
| **P2-2** | Separate small dense draft model (classic `--draft-model` analog) if MTP head never wins | **no** |
| **P2-3** | Batch / multi-request MTP skip (mlx-lm server pattern) | **no** |

**Hard constraints (all fires):** no LoopBrake; no dual 35B processes; no unrelated fuse/GDN thrash merges into MTP-only commits.

---

## 4. Microbench plan (step B — next fire)

Create `docs/experiments/mtp-stream-p0/MICROBENCH.md` with at least:

| Row | Metric | How |
|-----|--------|-----|
| **VERIFY_COST** | verify wall for K=2,4,6 vs K× eager T=1 | Extract `[mtp-t]` verify= from TPS probes + optional single-process micro; no full Maxwell SAR |
| **DRAFT_COST** | draft µs / step and / draft token | same |
| **ACCEPT_RATE** | accepted / proposed | count from `[mtp]` lines |
| **GEN_TPS** | 256-token probe | already in TPS_probe_*.txt |

Stop criterion for B: VERIFY_COST row present with numbers.

---

## 5. Stop criteria (loop complete when all hold)

1. **[met this fire]** `MTP_OPTIMALITY_PLAN.md` exists with online MLX MTP findings + ranked fixes.
2. **[met 2026-08-01]** `MICROBENCH.md` has **VERIFY_COST** verify-vs-eager numbers (β_K≈1.5–1.7; draft δ≈4.1 at K=2).  
3. **[unmet]** ≥1 code **or** measured improvement that either  
   (a) improves gen t/s on 256-token probe vs prior ~6–7 MTP baseline, **or**  
   (b) **auto-disables / falls back to eager** when MTP is slower, with passing smoke (`MLX_LOAD_MTP_HEAD=1`, short `--use-mtp --n-draft 2`, max-tokens 32, no Stream(cpu)).

When 1–3 all met: report DONE + human next step; `scheduler_delete` this scheduled task.

---

## 6. Next fire recommendation

1. ~~**Step B:** build `MICROBENCH.md`~~ **done** — see `MICROBENCH.md`.  
2. **Step C:** implement **P0-1** (auto-fallback to eager when MTP slower) as smallest product-safe code change; microbench shows 0.23–0.26× with no plausible win until draft≪T₁.

---

## 7. Citation / link dump

- https://github.com/ml-explore/mlx-lm/pull/990  
- https://github.com/Blaizzy/mlx-vlm/issues/981  
- https://mlx-optiq.com/docs/mtp  
- https://www.reddit.com/r/LocalLLaMA/comments/1rzntv5/multitoken_prediction_mtp_for_qwen35_is_coming_to/  
- https://github.com/ml-explore/mlx-lm/discussions/890  
- https://sebastianraschka.com/llm-architecture-gallery/mtp/  
- Local: `docs/experiments/mtp-stream-p0/README.md`, `TPS_probe_*.txt`, `src/common/generate.cpp` (`mtp_speculative_step`)
