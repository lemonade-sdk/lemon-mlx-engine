# H1 / glue fix on LemonMLXE 35B (field model)

**Date:** 2026-08-01  
**Tip:** see git log  
**Question:** Does the RMSNorm +1 “glue” (raw HF / guru87 MTP heads) help **35B**?

## What “glue” is

| Item | Meaning |
|------|---------|
| Problem | Some MTP heads ship RMSNorm as **(γ−1)** (HF). Engine expects **γ**. |
| Fix | If `mean(pre_fc_norm_hidden) < 0.2`, add **1.0** to dense `*norm*.weight`. |
| Escape | `MLX_MTP_NO_NORM_SHIFT=1` |
| Success case | 0.8B guru87 head: accept 0 → ~0.31, **~100 t/s** (H2) |

## 35B result (this experiment)

| Check | Result |
|-------|--------|
| Model | `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| `pre_fc_norm_hidden` mean | **0.4937** (≥ 0.2) |
| `norm_shifted` | **0** (glue correctly **no-ops**) |
| auto_quantized | 13 |
| 256-tok n_draft=2 full fuse | **27.44 t/s** (log `H1_glue_35B_TPS_ndraft2.txt`) |
| vs C7 best | **~same** (27.34) |
| vs eager | ~1.05× |

**Conclusion:** LemonMLXE 35B MTP norms are already converted. Forcing +1 would **double-shift** and break accept. Glue is **required for raw HF delta heads**, **not** a 35B TPS lever.

## Distinction (do not confuse)

| Path | Glue needed? | ~t/s (gfx1150) |
|------|----------------|----------------|
| 35B native MTP (H1 product) | No (already γ) | ~27 MTP / ~26 eager |
| 0.8B stitched delta (H2) | Yes (guru87 raw) | ~100 MTP after glue |
| 0.8B base alone | N/A (no head) | ~113 eager |

Log: `H1_glue_35B_load_smoke.txt`, `H1_glue_35B_TPS_ndraft2.txt`.
