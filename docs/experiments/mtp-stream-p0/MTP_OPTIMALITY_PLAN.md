# MTP Optimality Plan (fix/mtp-stream-p0)

**Date:** 2026-08-01 (D1 path-to-100 revision)  
**Model bar:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` on **gfx1150** (AMD Radeon 890M, **cus=8**, low_cu iGPU)  
**Branch tip at this revision:** see git log on this file  
**User stop bar (scheduler):** **measured MTP Generation t/s ≥ 100** on the Generation: line (real probe log under this dir), smoke green (no Stream(cpu)), MICROBENCH/CRITICAL_ANALYSIS updated, quintuple supervisors PASS on stop claim.

**HARD BAN (never implement as “win”):** LoopBrake / phrase-brake / early-stop seatbelts; auto-disable MTP when slow; silent eager fallback as fix; max_tokens tricks; inventing ≥100 t/s without a probe log.

---

## 0. Executive: path to 100 t/s (this fire’s D1)

### 0.1 Measured single-stream ladder (256-tok, no-think, full quant fuse, gfx1150)

| Config | gen t/s | Notes |
|--------|---------|-------|
| Eager (no MTP path) | **26.13** | Single-seq product baseline |
| MTP pre-C1 (dense BF16 draft) | 6.05 | Draft dominated (~157 ms) |
| MTP C1 (runtime quant head) | 15.87 | draft 157→24 ms |
| MTP C2 sequential T=1 verify | 19.72 | verify 86→66 ms |
| MTP C3 adaptive + barrier defer | 19.64 | no regression; control for deep n_draft |
| **MTP C4 parallel draft ‖ first verify** | **20.64** | best MTP so far (`C4_TPS_probe_ndraft2_parallel.txt`) |
| C4 + inter-step prefetch (`MLX_MTP_PREFETCH=1`) | 19.42 | **regression** vs C4 default; default remains off (`C4_TPS_probe_ndraft2_prefetch.txt`) |

**Gap to stop bar:** 100 / 20.64 ≈ **4.8×** above best MTP; 100 / 26.13 ≈ **3.8×** above eager T=1.

### 0.2 Bandwidth / speculative theory (why 100 is not a micro-opt)

mlx-lm MTP form ([PR #990](https://github.com/ml-explore/mlx-lm/pull/990)):

\[
\text{speedup} \approx \frac{1+p}{\beta + \delta}
\]

with \(p\) = mean accepted draft tokens per step (per true draft slot rate × slots), \(\beta = T_{\text{verify}}/(K\cdot T_1)\), \(\delta = T_{\text{draft}}/T_1\), \(T_1 \approx 38.3\,\text{ms}\) at 26.13 t/s.

**Field (post-C2/C4, K=2 ≈ one draft slot):**

| Quantity | Ballpark | Source |
|----------|----------|--------|
| \(T_1\) | ~38.3 ms | eager 26.13 t/s |
| draft residual after C4 hide | still ~20–55 ms joint window | C4 timers |
| T=1 trunk verify | ~35–40 ms | ≈ one eager step |
| \(p\) (n_draft=2) | ~0.6–0.7 | accept histograms |
| \(\beta\) multi-token batch verify (pre-C2) | ~1.5–1.7 | MICROBENCH VERIFY_COST |

**Implications:**

1. **Even if draft were free** (\(\delta=0\)) and every draft accepted (\(p=1\)) with \(\beta=1\): speedup ≤ **2×** → ceiling **~52 t/s** on this device — **still &lt; 100**.
2. **Even if draft free, p→∞ theoretically impossible** on single-stream: each accepted token still needs trunk-equivalent work somewhere; MoE multi-token verify does **not** reuse weights the way dense does (community consensus: MoE verify pulls different experts → near-linear traffic).
3. mlx-lm MoE field: Qwen3.5-35B-A3B 4-bit M4 Pro **~1.03–1.11×** only when overhead is tiny — dense 27B sees **~1.5×**. OptiQ: γ=1 optimal; adaptive depth often **loses**. Our C4 **0.79× eager** is still a code-tax problem, but closing to 1.0–1.2× eager does **not** reach 100.
4. **100 t/s single-seq on this 35B@gfx1150 is not a realistic MTP optimality target** given measured \(T_1\) and MoE physics. Claiming it without a probe log is banned; inventing it is banned.

### 0.3 Hardware ceiling (honest)

| Layer | Estimate | Basis |
|-------|----------|--------|
| **Measured eager single-seq** | **~26 t/s** | `TPS_probe_no_mtp.txt` |
| **Measured best MTP single-seq** | **~20.6 t/s** | C4 parallel probe |
| **Optimistic MTP single-seq (software)** | **~28–40 t/s** | If draft ≪ T₁ and p≳0.8 and verify ≈ T₁ per emitted token → approach or slightly beat eager; still ≪100 |
| **Hard single-seq AR ceiling (this iGPU)** | **≪ 100 t/s** for 35B-class MoE | 8 CU, low_cu, ~22 GB weights resident; bandwidth-bound decode |
| **Aggregate 100 t/s (product)** | Possible only as **multi-seq / continuous-batch server throughput**, smaller model, or different GPU | See §0.4 |

Device stamp from probes: `gfx1150 (AMD Radeon 890M Graphics) cus=8 warp=32 lds=64KB` · model mem ~21.9 GB with quant MTP head.

### 0.4 What *would* make Generation or product ≥100 t/s real

Ranked **required** changes (any one class may suffice; micro-opts alone will not):

| Path | What changes | Fits stop bar wording? |
|------|----------------|------------------------|
| **H1 — Faster T=1 device** | Discrete GPU / higher-bandwidth ROCm target so eager alone is ≥100 (then MTP optional) | Yes if same model measured on documented device |
| **H2 — Smaller model** | e.g. dense 4B–9B or much smaller MoE where eager already 50–150 t/s on this iGPU | Yes if plan documents measured target model |
| **H3 — Aggregate / continuous batch** | Server multi-request token throughput (sum gen tokens / wall) ≥100 while single-seq still ~20–40 | **Only if** stop bar is redefined; current bar is single-process Generation: line |
| **H4 — Speculative miracle (not expected)** | Draft free + multi-token verify free (dense-like) + p≈K−1 with large K | Contradicted by MoE β≈1.5–1.7 and δ history; do not plan on this |

**Documented decision for this branch:** keep measuring **single-seq Generation t/s** on LemonMLXE 35B gfx1150; **do not fake 100**. Continue real throughput cuts toward **max single-seq MTP** (close gap to eager, then beat eager if physics allows). When software plateau is clear, stop claim for “100 on this stack” is **FAIL with hardware ceiling** — escalate to H1/H2 or renegotiate bar (not seatbelt).

### 0.5 Still-valuable single-seq code cuts (toward max MTP, not fake 100)

Priority order for **D2** fires (never disable MTP):

| ID | Cut | Why |
|----|-----|-----|
| **C5** | **Cheaper draft `lm_head`** — top-k / vocab slice / shared fused head path; keep argmax on device | Residual draft still tens of ms; full vocab each draft token |
| **C6** | **Fewer host barriers** — collapse `eval`/`item` in draft+accept; keep one-behind async where safe | Hard barriers kill overlap on ROCm |
| **C7** | **Draft MoE kernel path** — ensure quant gather_qmm / fused experts on all draft linears; avoid accidental dense | C1 partial; audit remaining dense_kept=7 |
| **C8** | **Verify path polish** — keep sequential T=1 default; only re-open batch verify if β→1 measured | C2 already won; don’t regress |
| **C9** | **Quality / Maxwell SAR** under C1+C2+C4 at temp 0 and 0.7 | Throughput without correctness is not product |

**Rejected as “wins”:** auto-disable MTP (HARD BAN); LoopBrake; n_draft gaming; claiming prefetch win (measured 19.42 &lt; 20.64).

### 0.6 Stop checklist (scheduler)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Measured MTP Generation t/s **≥ 100** on LemonMLXE 35B (or documented different measured target) with `--use-mtp` + head loaded; probe under this dir | **UNMET** (best **20.64**) |
| 2 | Smoke green (no Stream(cpu)) | met on C1–C4 probes |
| 3 | MICROBENCH / CRITICAL_ANALYSIS include the ≥100 result | **UNMET** |
| 4 | Quintuple supervisors PASS on stop claim | N/A until (1) |

If (1) is impossible on gfx1150 35B: **report ceiling**, continue real C5+ work or H1/H2 — **never invent numbers**.

---

## 1. Online reference findings (MLX / mlx-lm / ecosystem)

### 1.1 mlx-lm native MTP (primary reference)

| Item | Detail |
|------|--------|
| PR | [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990) — “Native MTP speculative decoding (Qwen3.5/3.6)” |
| CLI | `mlx_lm.generate --mtp` / `mlx_lm.server --mtp` (opt-in) |
| Core API | `mtp_generate_step()`: draft via MTP head, verify backbone `[confirmed, draft]`, GDN `n_confirmed` snapshot + rollback |
| Head shape | Fuse norms + embed → MTP layer → **shared `lm_head`** |
| Dense result | Qwen3.5/3.6-27B 4-bit M4 Pro: ~15.3 → ~23–24.6 t/s (**~1.5–1.57×**), accept ~80–88% (temp=0) |
| MoE result | Qwen3.5-35B-A3B 4-bit M4 Pro: ~85.3 → ~87.9 t/s (**~1.04×**); M2 Ultra 8-bit ~1.11× — **marginal** |
| Bandwidth model | `speedup ≈ (1+p)/(β+δ)`; MoE multi-token verify ≈ linear expert traffic |

Related:

- [mlx-vlm#981](https://github.com/Blaizzy/mlx-vlm/issues/981): classic `--draft-model` speculative decode on server
- OptiQ MTP: γ=1 default/optimal; adaptive depth often −4–17%; skip MTP when base already fast (product policy — **not** our stop-bar “fix”)
- Community: dense wins big; MoE wins only when verify cheap and accept high ([LocalLLaMA / PR discussion](https://www.reddit.com/r/LocalLLaMA/comments/1rzntv5/multitoken_prediction_mtp_for_qwen35_is_coming_to/))

### 1.2 Implication for 100 t/s

Reference stacks that already run **~85–90 t/s eager** on Apple Silicon MoE still only add **~1.05×** with MTP. Hitting **100** there is “slightly better metal + small MTP win,” not a 4× software miracle. Our gfx1150 eager is **26**, so the same relative MTP gains yield **~27–30 t/s**, not 100.

---

## 2. Our stack vs reference (gap list)

### 2.1 How we enable MTP

| Ours | mlx-lm / OptiQ |
|------|----------------|
| `MLX_LOAD_MTP_HEAD=1` | Weights present + `--mtp` |
| `--use-mtp` | `--mtp` |
| `--n-draft N` (chat default **1**); server `--n-draft-tokens` default **3** | Often γ≈1 |
| `MTP_DEBUG=1`, `MTP_TIMING=1` | Engine accept counters |
| `StreamGuard` on `mtp_speculative_step` | Metal default stream |
| Quant: runtime quant MTP head (C1); `MLX_MTP_KEEP_BF16` / `MLX_MTP_DEQUANT` escapes | quant_predicate debates |

Semantics: `n_draft_tokens` = block size = **d0** + **N−1** drafted tokens. `--n-draft 2` ⇒ **1** true draft slot (γ≈1).

### 2.2 Call chain (code)

Primary: `src/common/generate.cpp` → `TokenIterator::mtp_speculative_step()`

1. **Draft:** serial (or C4 side-stream) MTP MoE steps + full `lm_head` + device argmax  
2. **Verify:** default **sequential T=1** (C2); optional batch via `MLX_MTP_BATCH_VERIFY=1`  
3. **C4:** draft ‖ first d0 verify on side stream; join; continue sequential on accept  
4. **Adaptive depth (C3):** `current_draft_count()` from accept history (min 2); `MLX_MTP_FIXED_DRAFT=1` disables  
5. **Prefetch (opt-in):** `MLX_MTP_PREFETCH=1` — measured regression; default **off**

### 2.3 Gap list (updated)

| # | Gap | Severity | Status |
|---|-----|----------|--------|
| G1 | ~~Auto-disable when slow~~ | — | **HARD BAN** — not a throughput fix |
| G2 | Multi-token verify cost | P0 | **Mitigated** via sequential T=1 (C2) |
| G3 | Draft cost (MoE + full lm_head) | P0 | **Partial** C1 quant; residual → C5/C7 |
| G4 | MTP dequant at load | P1 | **Done** C1 runtime quant |
| G5 | Host eval/item barriers | P1 | Partial C3/C4; → C6 |
| G6 | Adaptive draft stub | P1 | **Done** C3 |
| G7 | Server n_draft default | P1 policy | open |
| G8 | MoE structural ceiling | P2 | **Documented** §0 — blocks 100 on this device |
| G9 | Microbench | Process | **Done** + C1–C4 ladder |
| G10 | Stream(cpu) | Done | StreamGuard |

---

## 3. Ranked fixes (execution status)

### Done (real cost cuts)

| ID | Fix | Result |
|----|-----|--------|
| **C1 / P1-2** | Runtime quant MTP head + reshape shared_expert | 6.05 → **15.87** t/s |
| **C2** | Sequential T=1 verify (not batch+capture re-run) | **19.72** t/s |
| **C3** | Adaptive n_draft + deferred barriers | hold **~19.6** |
| **C4** | Parallel draft ‖ first verify | **20.64** t/s |
| **C6** | Eval d0 before draft launch; device draft slices; no double-eval fill | Smoke green; joint still ~53 ms (see CRITICAL C6); **256-tok TBD D3** |

### Failed / rejected experiments

| ID | Result |
|----|--------|
| C2 no-capture batch verify | **11.78** t/s regression |
| C4 prefetch default | **19.42** t/s regression (flag remains opt-in off) |
| Auto-disable / LoopBrake | **HARD BAN** |

### Next D2 candidates (see §0.5)

C5 cheaper draft lm_head → C6 barriers → C7 draft MoE audit → C9 Maxwell quality.

### P0 product notes (non-stop)

- Prefer `--n-draft 2` (γ≈1) until draft≪T₁.  
- Do **not** ship deeper n_draft as default win.  
- Document that MTP may be slower than eager on gfx1150 MoE until C5+ land — without auto-disabling.

---

## 4. Microbench

See [`MICROBENCH.md`](./MICROBENCH.md): VERIFY_COST β_K≈1.5–1.7 pre-C2; draft δ≈4.1 pre-C1; post-C ladder tables; C4 **20.64** t/s.

---

## 5. Stop criteria

### 5.1 Scheduler stop (user bar)

ALL of: **MTP gen ≥ 100 t/s** (real log) + smoke green + docs updated + supervisors 5/5 PASS.  
**Current: UNMET** — ceiling analysis §0 says **not reachable** on gfx1150 35B single-seq without H1/H2.

### 5.2 Engineering plateau (local, not scheduler DONE)

1. Plan includes path-to-100 + hardware ceiling (**this fire D1**).  
2. MICROBENCH VERIFY_COST + C ladder present.  
3. Real cost cuts C1–C4 landed; residual C5+ queued.  
4. Best MTP still &lt; eager; gap ~21% — continue C5, do not declare product win.

When 5.1 impossible: keep iterating real cuts; main agent / human must choose H1/H2 or bar change — field agent **does not** `scheduler_delete` on fake PASS.

---

## 6. Next fire recommendation

1. **D2 C5:** cheaper draft lm_head / reduce full-vocab cost on draft tokens (measure with MTP_TIMING).  
2. **D3:** 256-tok probe after C5; update MICROBENCH + CRITICAL_ANALYSIS.  
3. Do **not** implement auto-fallback.  
4. If C5–C7 fail to approach eager: publish plateau (~20–26 t/s) and escalate hardware/model path for any ≥100 claim.

---

## 7. Citation / link dump

- https://github.com/ml-explore/mlx-lm/pull/990  
- https://github.com/Blaizzy/mlx-vlm/issues/981  
- https://mlx-optiq.com/docs/mtp  
- https://www.reddit.com/r/LocalLLaMA/comments/1rzntv5/multitoken_prediction_mtp_for_qwen35_is_coming_to/  
- https://github.com/ml-explore/mlx-lm/discussions/890  
- https://sebastianraschka.com/llm-architecture-gallery/mtp/  
- Local: `README.md`, `CRITICAL_ANALYSIS.md`, `MICROBENCH.md`, `C*_TPS_probe_*.txt`, `src/common/generate.cpp`
