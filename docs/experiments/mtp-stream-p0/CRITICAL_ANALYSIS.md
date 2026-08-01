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
