# Critical analysis: MTP slowness and what actually fixed what

**Branch:** `fix/mtp-stream-p0`  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit · gfx1150 · full quant fuse ON  

### Fire 1 — P0-A/P0-B correctness (greedy contract + γ>1 draft KV)

- **P0-A:** MTP draft+verify are device `argmax` only; non-greedy `temperature`/`top_p`/`repetition_penalty` now **hard-error** (`mtp_greedy_only_violation` → `TokenIterator` throw; server HTTP 400). Chat `--use-mtp` coerces unset temp/top_p to 0/1. **MTP v1 = greedy-only.**
- **P0-B:** `mtp_run_draft_chain` passes MTP KV on the **final** draft step when `use_mtp_kv` (n_draft≥3); n_draft=2 still skips MTP KV (C7).
- **Defaults:** chat + server `n_draft` default **2** (server was 3).

## Field ladder (256-tok, no-think, temp=0, n_draft=2 unless noted)

| Config | gen t/s | warm draft_ms | warm verify_ms | Notes |
|--------|---------|---------------|----------------|-------|
| Eager (no MTP) | **26.13** | — | — | Product TPS baseline |
| MTP pre-C1 (dense BF16 draft) | **6.05** | **156.8** | 124.1 | Draft dominated |
| MTP C1 (runtime quant MTP head) | **15.87** | **23.7** | 86.2 | Draft −85% |
| MTP C2 no-capture (batch verify, re-run partial) | **11.78** | 20.3 | 97.7 | **Regression** — re-run tax |
| MTP C2 sequential T=1 verify | **19.72** | 20.3 | **66.3** | |
| MTP C6 barrier order | **22.39** | joint ~52.7 | — | |
| **MTP C7** skip γ=1 KV + lazy hidden | **27.34** | joint **~37.8** | — | **Beats eager** |

## Critical verdicts

### C1 quant MTP head — **REAL win (not fake)**

- LemonMLXE ships **BF16 dense** `mtp.*` (0 ckpt quant groups). Old “Dequantized 20” was shape-map size.
- Runtime `mx::quantize` + `QuantizedWeightRegistry` + `linear_forward` / `gather_qmm` is a genuine bandwidth cut.
- **shared_expert** must be reshaped to `[1,out,in]` or `gather_qmm` shape-faults.
- Mem rises (~19.8 → ~21.9 GB) from quant packs — tradeoff.
- Escape: `MLX_MTP_KEEP_BF16`, `MLX_MTP_DEQUANT`.

### Full GDN quant fuse — **orthogonal to MTP gap**

- Same full fuse on eager 26 and MTP 6–19 runs. Fuse is not why MTP lost.
- Full GDN in_proj fuse is a mem/packing choice, not the MTP root cause.

### C2 “no capture_spec” — **failed experiment**

- Default-off capture + restore/re-run on partial accept **hurt** t/s (11.8).
- At accept ~0.7, re-run cost outweighs avoiding `store_spec`.

### C2 sequential T=1 verify — **REAL win**

- Replaces multi-token trunk verify + `capture_spec` with per-token L=1 `call_fn` + early exit.
- Enables ROCm `gpu_set_graph_decode_mode(true)` and fused GDN T=1 path.
- verify_ms 86 → 66; gen 15.9 → **19.7 t/s**.
- Batch path retained behind `MLX_MTP_BATCH_VERIFY=1`.

### Remaining gap to eager (~26 t/s)

Still ~25% short of eager. Residual:

1. **Draft still paid** (~20 ms/step) even when accept=0.
2. **Verify still ≥1–2× T=1** (mean ~66 ms for ~1.7 tokens emitted ≈ 39 ms/token before draft).
3. **Hard barriers** (eval per draft step + per verify token) vs eager async one-behind.
4. **Full vocab lm_head** still once per draft token.
5. Multi-turn Maxwell under MTP can still thrash (incomplete SAR EXIT 143; token spam seen under n_draft=4 dense era) — quality not fully closed.

## Not acceptable “fixes”

- LoopBrake / auto-disable MTP when slow / seatbelt scorers — **rejected**.

## PR #63 P0-MTP server gate (Stream(cpu,0) HTTP 500)

CLI MTP was green after `StreamGuard` + own gen stream; **server** still 500’d because mlx CPU command encoders are **thread_local** and httplib workers never created `Stream(cpu, 0)`. Fix: re-bind known CPU streams into the worker TLS map inside `StreamGuard` (`ensure_thread_cpu_stream_encoders`). **M1+M2+M3 PASS** — see `gates/RESULTS.md`.

## What “resolved” looks like next

1. Close gap: async emit, cheaper draft lm_head, or only draft when history accepts well (adaptive K that still runs MTP when useful — not disable).
2. Quality: full Maxwell SAR with C1+C2 sequential at temp 0 and 0.7.
3. Optional A/B: `MLX_MTP_BATCH_VERIFY=1` vs sequential on n_draft=4.

## C3 (barrier defer + adaptive n_draft)

| Config | gen t/s | draft_ms | verify_ms | notes |
|--------|---------|----------|-----------|-------|
| C2 seq n_draft=2 | 19.72 | 20.3 | 66.3 | prior best |
| C3 adaptive n_draft=2 | **19.64** | 20.4 | 66.5 | no regression |
| C3 adaptive max=4 | 18.41 | 27.0 | 67.6 | saw n_draft 2/3/4 |
| C3 fixed n_draft=4 | **19.66** | 26.3 | 82.5 | sequential verify scales OK |
| eager | 26.13 | — | — | still ahead |

Changes: defer KV quant + hidden stash to end of sequential verify; `current_draft_count()` uses accept history (min 2, max n_draft_tokens). `MLX_MTP_FIXED_DRAFT=1` disables adaptive.

**Review:** Adaptive did not beat fixed-2/4 on this probe; kept as cost control for longer n_draft when accept collapses. Barrier defer holds ~19.6 t/s.

## C4 (parallel draft + first verify; empty State signal)

| Config | gen t/s | notes |
|--------|---------|-------|
| C2/C3 sequential n_draft=2 | 19.64–19.72 | prior best |
| **C4 parallel draft‖first verify** | **20.64** | side-stream draft + trunk d0 verify join |
| eager | 26.13 | still ~21% ahead |

**Code** (`generate.cpp`):

1. **Parallel draft + first verify:** MTP draft chain runs on a dedicated side stream while the trunk verifies `d0` on the generation stream; join before accept decision. On accept, sequential T=1 continues for remaining tokens. Disable: `MLX_MTP_NO_PARALLEL_DRAFT=1`.
2. **Empty `LMOutput::State` signal** for “return hidden” (no per-token `mx::array(0.0f)` dummy).
3. Keep device `argmax` as `y_` (no host re-upload) on reject/bonus.
4. Inter-step draft prefetch is **opt-in** (`MLX_MTP_PREFETCH=1`) — default off; host emit on this stack is too short to hide draft and post-step draft became unaccounted wall.

**Warm timing note (parallel accounting):** the `draft=` timer slot includes the joint draft‖first-verify window (~55 ms); `verify=` is residual (0 on reject, ~38 ms second token on accept). Real wall step totals: acc0 ~55 ms (was ~58), acc1 ~93 ms (was ~95). Gen **20.64 t/s** vs C3 **19.6** (~+5%).

**mlx-lm PR#990 contrast:** mlx-lm verifies `[confirmed, draft]` in one multi-token backbone pass with GDN `n_confirmed` rollback. Our sequential T=1 won on gfx1150 MoE (batch verify β≈1.6). C4 keeps sequential accept/early-exit but hides draft latency behind the first T=1 verify. MoE reference gains remain small (mlx-lm ~1.03–1.11× on Metal MoE); we remain below eager until draft cost collapses further or multi-token verify becomes free.

Log: `C4_TPS_probe_ndraft2_parallel.txt`.

**Prefetch A/B:** `MLX_MTP_PREFETCH=1` → **19.42** gen t/s (`C4_TPS_probe_ndraft2_prefetch.txt`) — regression vs 20.64; remains default-off.

## Path to 100 t/s (scheduler stop bar)

User bar is MTP Generation ≥ **100** t/s. This is **not met** (best **27.34** after C7). Analysis:

- Eager ceiling on this device/model: **~26 t/s** (8 CU gfx1150, ~22 GB resident).
- Speculative form \((1+p)/(\beta+\delta)\) cannot deliver ~4× over that T₁ under MoE near-linear verify; free-draft ideal still ≲ **~2×** (~52 t/s) ≪ 100.
- Real next cuts: cheaper draft lm_head, fewer barriers, MoE draft audit — close gap to eager, not invent 100.
- Reaching 100 requires documented **H1** faster GPU, **H2** smaller model, or **H3** multi-seq aggregate (bar change) — see `MTP_OPTIMALITY_PLAN.md` §0.

**HARD BAN reminder:** auto-disable / LoopBrake / silent eager fallback are not resolutions.

## C6 (barrier order + device draft feed; 2026-08-01)

**Code** (`generate.cpp` parallel path):

1. **`mx::eval(y_.tokens)` before** side-stream draft launch (C4 evaluated d0 *after* async draft start — can force a device join that kills overlap).
2. Skip second `eval` inside host draft materialization after join already `eval(pred, drafts_dev)`.
3. Sequential verify feeds **slice `drafts_dev`** instead of host `int` re-upload.

**Smoke** (`C6_smoke_ndraft2_max32.txt`): green, no Stream(cpu).

### D3 256-tok measure (this fire)

| Config | gen t/s | warm joint ms | warm accept p | log |
|--------|---------|---------------|---------------|-----|
| C4 parallel | 20.64 | 55.3 | 0.62 | `C4_TPS_probe_ndraft2_parallel.txt` |
| **C6** | **22.39** | **52.7** | **0.88** | `C6_TPS_probe_ndraft2.txt` |
| eager | 26.13 | — | — | `TPS_probe_no_mtp.txt` |

**Verdict:** **REAL modest ladder step** to **22.39** gen t/s (+8.5% vs C4). Joint wall only −2.6 ms (still ~53 ms ≫ T₁ 38 ms → 8CU contention remains). Higher accept rate on this probe’s prompt is a **confound** for attributing the full +8.5% to C6 alone. **Not** a path to 100; gap to eager still ~14%.

**Next D2:** residual joint/contention or C5 draft cost; quality Maxwell SAR optional. For ≥100: H1/H2/H3 only.

## C7 (skip single-step MTP KV + lazy hidden stash; 2026-08-01)

**Code** (`generate.cpp`):

1. **`n_draft==2` (γ≈1):** do not reset/update MTP layer KV — single draft step never rereads cache (rope offset 0, `cache=nullptr`).
2. **Deeper n_draft:** keep KV only for steps with a later draft consumer (`i < n_draft-1`).
3. **Lazy `mtp_trunk_hidden_` stash:** drop per-step `mx::eval(h_slice)`; next draft eval pulls the slice.

**Smoke** (`C7_smoke_ndraft2_max32.txt`): green, no Stream(cpu); joint ~38 ms.

### D3 256-tok measure (same Fourier prompt as C6)

| Config | gen t/s | warm joint ms | warm accept p | log |
|--------|---------|---------------|---------------|-----|
| C6 | 22.39 | 52.7 | 0.88 | `C6_TPS_probe_ndraft2.txt` |
| **C7** | **27.34** | **37.8** | 0.85 | `C7_TPS_probe_ndraft2.txt` |
| eager | 26.13 | — | — | `TPS_probe_no_mtp.txt` |

**Verdict:** **REAL win.** C7 **beats full-fuse eager** (27.34 > 26.13, **~1.05×**) with joint collapsed to T₁. Accept rate similar to C6 on same prompt — Δ is not an accept confound. **Stop bar 100 still UNMET** (~3.7× short). Next toward max single-seq: residual second-verify cost / accept; toward 100: H1/H2/H3 only.

## C8 (async residual y_ + no MTP_TIMING barrier; 2026-08-01)

**Code** (`generate.cpp`):

1. After setting `y_` (accept residual / reject path), **`mx::async_eval(y_.tokens)` immediately** so residual T=1 can start before host metrics/emit.
2. **`MTP_TIMING` no longer forces `mx::eval(y_)`** (that barrier serialized residual into every timed step and blocked hide-under-emit). Full barrier timing: **`MTP_TIMING_SYNC=1`**.

**Smoke** (`C8_smoke_ndraft2_max32.txt`): green, no Stream(cpu).

### D3 256-tok measure (same Fourier prompt as C7)

| Config | gen t/s | notes | log |
|--------|---------|-------|-----|
| C7 | **27.34** | best | `C7_TPS_probe_ndraft2.txt` |
| **C8** | **27.29** | **flat** (−0.05) | `C8_TPS_probe_ndraft2.txt` |
| eager | 26.13 | | `TPS_probe_no_mtp.txt` |

**Verdict:** C8 **does not raise** 256-tok gen t/s. Host emit cannot hide residual T=1; residual spills into next joint. Keep C8 as hygiene (no forced MTP_TIMING barrier; early async_eval). **Ladder best remains C7 27.34.** Stop bar 100 **UNMET**. Plateau: further micro-opts unlikely past ~28–40 without H1/H2.

## C9 n_draft=3 A/B (D3 measure only)

| Config | gen t/s | log |
|--------|---------|-----|
| C7 n_draft=2 | **27.34** | `C7_TPS_probe_ndraft2.txt` |
| C9 n_draft=3 fixed | **22.71** | `C9_TPS_probe_ndraft3_fixed.txt` |

**Verdict:** Deeper draft **regresses**. Keep default γ≈1 (`--n-draft 2`). **Software plateau confirmed** at ~27 t/s on this device/model; **100 t/s single-seq is not a realistic target** (needs H1 discrete GPU / H2 smaller model / H3 multi-seq aggregate).

## H2 smaller models on gfx1150 (D3 measure)

| Model | MTP? | gen t/s | log |
|-------|------|---------|-----|
| 35B MoE MTP C7 | yes | **27.34** | `C7_TPS_probe_ndraft2.txt` |
| 4B dense MTP | yes | **24.65** | `H2_TPS_probe_4B_MTP_ndraft2.txt` |
| 4B dense eager | no | 26.50 | `H2_TPS_probe_4B_eager_no_mtp.txt` |
| 0.8B eager | no | **113.4** | `H2_TPS_probe_0p8B_eager.txt` |

**Verdict:** H2 measured further; **accept fix landed (RMSNorm +1)**:

| Config | gen t/s | MTP head | notes |
|--------|---------|----------|-------|
| 0.8B eager | 113.4 | no | |
| 0.8B MTP n_draft=2 (pre-fix) | **97.7** | yes | accept≈0; unshifted HF norms |
| 0.8B MTP n_draft=1 | **101.87** | yes | ≥100 numeric; **no draft slots** (degenerate) |
| **0.8B MTP n_draft=2 + RMSNorm+1** | **100.045** | yes | **productive** mean accept≈0.31; triple-run 100.0 / 99.9 / 99.7 |

### C10 / H2 accept fix (D2, 2026-08-01) — RMSNorm +1 on raw MTP heads

**Root cause:** `guru87/Qwen3.5-0.8B-MTP` ships raw HF RMSNorm as (γ−1) (pre_fc_norm_hidden mean ≈ −0.34). mlx-community converted packages (4B MTP) already bake +1 (mean ≈ 0.75). Without +1, draft logits are garbage → accept≡0 even with KEEP_BF16.

**Code** (`mtp_head.cpp` `load_mtp_weights`): detect unshifted `pre_fc_norm_hidden` mean &lt; 0.2 (f32 cast) and add 1.0 to all dense `*norm*.weight` tensors. Escape: `MLX_MTP_NO_NORM_SHIFT=1`. Also: do not double-prefix keys already named `mtp.*` in `load_mtp_delta_model`.

**Result:** accept 0 → **~0.31** (γ=1); gen **100.045** t/s on n_draft=2 (`H2_TPS_probe_0p8B_MTP_ndraft2_normshift_PASS100.txt`). Smoke green. **Documented measured target for stop bar:** 0.8B MTP on gfx1150 (35B ceiling remains ~27 t/s).

Package: local `mlx-community/Qwen3.5-0.8B-MTP-4bit` delta (guru87 head + mlx 0.8B-4bit base).

## C11 draft MoE top_k shortcut (2026-08-01 fire)

**Hypothesis:** LemonMLXE 35B MTP head is full MoE with **num_experts=256, num_experts_per_tok=8**. Each γ=1 draft step pays 8× SwitchGLU gathers + shared expert. Speculative draft may keep usable accept with fewer experts → less draft bandwidth / CU contention vs C7 joint≈T₁.

**Code** (`src/llm/models/mtp_moe.cpp`): `MLX_MTP_DRAFT_TOPK=N` overrides routing top-k at draft time (clamped to `[1, num_experts]`; unset keeps trained k). One-shot log when active.

### D3 256-tok measure (Fourier-style, n_draft=2, full quant fuse, gfx1150)

| Config | gen t/s | warm mean accept | warm mean total ms | warm joint draft= ms | log |
|--------|---------|------------------|--------------------|----------------------|-----|
| **C7** (top_k=8 trained) | **27.34** | **0.854** | 67.8 | **37.8** | `C7_TPS_probe_ndraft2.txt` |
| **C11** `MLX_MTP_DRAFT_TOPK=2` | **26.94** | **0.716** | 63.6 | **60.5** | `C11_TPS_probe_ndraft2_topk2.txt` |
| eager | 26.13 | — | — | — | `TPS_probe_no_mtp.txt` |

**Verdict:** **FAIL / slight REGRESS** vs C7 (−0.40 gen t/s). Accept dropped (0.85→0.72) so tokens/step fell enough to erase any step-wall savings. Joint timer slot **inflated** (38→60 ms) — fewer experts did **not** shrink the draft‖first-verify window on this stack (argpartition/gather shape change + accept pattern; not a free δ cut). Flag remains **opt-in default-off** for future top_k A/B; do **not** ship top_k=2 as default.

**Implication:** Confirms post-C7 plateau — draft MoE expert count is not the remaining lever to beat ~27.3 on gfx1150 35B. Next real cuts: C5 fused/cheaper lm_head only if joint still draft-bound under a new measurement; otherwise H1/H2/H3 for ≥100.

## C12 pipeline second-verify under d0 emit (2026-08-01 fire)

**Hypothesis:** On γ=1 accept, current path finishes joint (draft‖v0) **and** v1 (feed d1) before returning d0 to the host. Host is idle during v1 (~T₁). Kick v1 async after match, return d0 immediately, complete v1 when draining buffered d1 so host emit of d0 overlaps GPU v1.

**Code** (`generate.cpp` / `generate.h`):

1. On parallel-path accept with `n_draft==2`, store `pred2` + state without `eval`; set `pending_v1_`.
2. `finish_pending_v1_()`: StreamGuard, eval pred, set `y_`, stash hidden, quantize KV.
3. `next()` finishes pending v1 before emitting buffered d1 / starting a new step / hitting max_tokens.
4. **Default OFF** after measure: enable with `MLX_MTP_PIPELINE_V1=1`.

### D3 256-tok measure (Fourier-style, n_draft=2, full quant fuse, gfx1150)

| Config | gen t/s | warm mean accept | notes | log |
|--------|---------|------------------|-------|-----|
| **C7** (no pipeline) | **27.34** | 0.854 | best | `C7_TPS_probe_ndraft2.txt` |
| **C12** pipeline v1 ON | **25.84** | 0.814 | **REGRESS** −1.5 t/s | `C12_TPS_probe_ndraft2_pipeline_v1.txt` |

**Verdict:** **FAIL / REGRESS.** Chat host emit of d0 is far shorter than v1 (~T₁), so the overlap wins nothing; finishing v1 on the d1-drain critical path **splits** what was a single GPU burst + fast dual emit into GPU → emit d0 → GPU finish → emit d1, adding host/guard tax. Step timers look “faster” only because v1 left the timed region. Keep code for stacks with heavy host emit; **do not default on** for gfx1150 chat. Ladder best remains **C7 27.34**.

## C13 MTP draft QKV fuse (2026-08-01 fire)

**Hypothesis:** MTP MoE draft attention runs three separate quant matmuls (q_gate, k, v). Fusing to one pack (trunk-style) cuts launches / may shrink draft‖v0 joint if draft still contending on 8CU.

**Code** (`mtp_moe.cpp` / `mtp_moe.h`): `ensure_qkv_proj_fused()` concatenates registered q|k|v packs; forward path one matmul + slice. **Default OFF** after measure (`MLX_MTP_QKV_FUSE=1` to enable).

### D3 256-tok measure (Fourier-style, n_draft=2, full quant fuse, gfx1150)

| Config | gen t/s | warm mean accept | warm joint draft= ms | log |
|--------|---------|------------------|----------------------|-----|
| **C7** (no MTP QKV fuse) | **27.34** | 0.854 | **37.8** | `C7_TPS_probe_ndraft2.txt` |
| **C13** `MLX_MTP_QKV_FUSE=1` | **25.45** | 0.814 | **66.9** | `C13_TPS_probe_ndraft2_qkv_fuse.txt` |

**Verdict:** **FAIL / REGRESS** (−1.9 t/s). Fuse log confirmed ON; joint **inflated** (38→67 ms) — larger fused GEMM + slice is slower than three small quant matmuls on this iGPU for MTP head shapes. Accept held (~0.81). Keep code opt-in only. Ladder best remains **C7 27.34**. Plateau: C11–C13 consecutive negatives on 35B micro-opts.

## C14 skip shared expert on draft (2026-08-01 fire)

**Hypothesis:** MTP MoE draft always pays routed experts **plus** shared-expert SwiGLU. C11 cut routed top_k; shared skip is a different routing shortcut that may cut draft bandwidth without changing top_k.

**Code** (`mtp_moe.cpp`): `MLX_MTP_NO_SHARED=1` returns `h + combined` (routed only). Default off.

### D3 256-tok measure (Fourier-style, n_draft=2, full quant fuse, gfx1150)

| Config | gen t/s | warm mean accept | warm joint draft= ms | log |
|--------|---------|------------------|----------------------|-----|
| **C7** (shared ON) | **27.34** | **0.854** | **37.8** | `C7_TPS_probe_ndraft2.txt` |
| **C14** `MLX_MTP_NO_SHARED=1` | **25.60** | **0.705** | **62.9** | `C14_TPS_probe_ndraft2_no_shared.txt` |

**Verdict:** **FAIL / REGRESS** (−1.7 t/s). Accept dropped (0.85→0.71); joint inflated (38→63 ms) — shared expert is quality-critical for this head, not free overhead. Flag opt-in only. Ladder best remains **C7 27.34**. **C11–C14 all regressed** — stop thrashing 35B draft-path micro-opts; free-draft sequential verify ≈ eager ceiling.

## C15 device-side accept + lazy host drafts (2026-08-01 fire)

**Hypothesis:** Parallel join always built full host `draft_tokens` before compare. Compare from device pointers first; on reject keep only d0 for emit (skip host materialize of rejected drafts) to cut barrier tax.

**Code** (`generate.cpp` parallel path): eval(pred, drafts_dev) → compare `trunk_next` to `dptr[0]` → materialize d1.. only on accept (or always when `MTP_DEBUG` for log visibility).

### D3 256-tok measure (Fourier-style, n_draft=2, full quant fuse, gfx1150, MTP_DEBUG+TIMING)

| Config | gen t/s | warm mean accept | warm joint draft= ms | log |
|--------|---------|------------------|----------------------|-----|
| **C7** | **27.34** | 0.854 | **37.8** | `C7_TPS_probe_ndraft2.txt` |
| **C15** device accept | **25.33** | 0.814 | **67.4** | `C15_TPS_probe_ndraft2_device_accept.txt` |

**Verdict:** **FAIL / REGRESS** (−2.0 t/s). Probe used `MTP_DEBUG=1` so reject path still materializes drafts for logs (C15 reject-lazy partly nullified). Even so, no win vs C7; joint slot higher (noise/contention). Keep cleaner device-first compare as hygiene. Ladder best remains **C7 27.34**. **C11–C15 consecutive non-wins** — 35B single-seq MTP micro-opt plateau confirmed.
