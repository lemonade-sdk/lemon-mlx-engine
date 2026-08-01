# MTP stream P0 — field runs

**Branch:** `fix/mtp-stream-p0` (includes #74 GDN + #76 quant-fuse + MTP StreamGuard)

## All quant-fuse flags + MTP

```bash
# Full quant fuse (attn/MLP + GDN in_proj) and MTP head + speculative decode
env \
  MLX_ENABLE_QUANT_FUSE=1 \
  MLX_ENABLE_QUANT_FUSE_GDN=1 \
  MLX_LOAD_MTP_HEAD=1 \
  MTP_DEBUG=1 \
  ./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
    --use-mtp \
    --n-draft 2 \
    --temperature 0 --max-tokens 128 --no-think
```

| Flag / env | Meaning |
|------------|---------|
| `MLX_ENABLE_QUANT_FUSE=1` | Fuse attn QKV + MLP gate\|up |
| `MLX_ENABLE_QUANT_FUSE_GDN=1` | Also fuse GDN in_proj `qkv\|z\|b\|a` (needs parent fuse) |
| `MLX_LOAD_MTP_HEAD=1` | Build MTP head from checkpoint |
| `--use-mtp` | Enable speculative decode path |
| **`--n-draft N`** | Draft length per step (**chat**). Default **1**. Use **2**, **4**, etc. |
| `--n-draft-tokens N` | Same idea on **server** (default **3**) |

### What `--n-draft` means

In the engine, `n_draft_tokens` is the speculative block size: **d0** (already sampled trunk token) plus **N−1** drafted tokens.  
So `--n-draft 2` → 1 draft token verified; `--n-draft 4` → 3 draft tokens.

## Results

| Log | Config | Result |
|-----|--------|--------|
| `M1_chat_short_temp0.txt` | MTP n_draft=4, fuse off | PASS no Stream err |
| `M1_fullfuse_ndraft2_temp0.txt` | **QUANT_FUSE+GDN**, MTP **n_draft=2** | **PASS** EXIT 0, mem 19.8 GB, n_draft=2, accept 1/1 drafts |

## Stream fix

- `StreamGuard` on `mtp_speculative_step`
- Own gen stream default on Linux/ROCm (`MLX_GEN_OWN_STREAM=0` to opt out)
