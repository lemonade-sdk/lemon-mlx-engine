# Critical analysis: MTP slowness and what actually fixed what

**Branch:** `fix/mtp-stream-p0`  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit · gfx1150 · full quant fuse ON  

## Field ladder (256-tok, no-think, temp=0, n_draft=2 unless noted)

| Config | gen t/s | warm draft_ms | warm verify_ms | Notes |
|--------|---------|---------------|----------------|-------|
| Eager (no MTP) | **26.13** | — | — | Product TPS baseline |
| MTP pre-C1 (dense BF16 draft) | **6.05** | **156.8** | 124.1 | Draft dominated |
| MTP C1 (runtime quant MTP head) | **15.87** | **23.7** | 86.2 | Draft −85% |
| MTP C2 no-capture (batch verify, re-run partial) | **11.78** | 20.3 | 97.7 | **Regression** — re-run tax |
| MTP C2 sequential T=1 verify | **19.72** | 20.3 | **66.3** | Best MTP so far |

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

User bar is MTP Generation ≥ **100** t/s. This is **not met** (best **22.39** after C6). Analysis:

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
