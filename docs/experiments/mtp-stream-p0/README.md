# MTP stream P0 — field runs

**Branch:** `fix/mtp-stream-p0` (includes #74 GDN + #76 quant-fuse + MTP StreamGuard)

**Complete branch map (all product / exp / historical / potential):** [`../BRANCH_MAP.md`](../BRANCH_MAP.md)  
**Lean product PR:** [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) (`fix/mtp-product`) · **Full tip alias:** `exp/mtp-stream-full`

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

- `StreamGuard` on `mtp_speculative_step` / `step` / `prepare`
- Own gen stream default on Linux/ROCm (`MLX_GEN_OWN_STREAM=0` to opt out)
- **Server workers:** re-bind CPU stream encoders into thread-local maps (`ensure_thread_cpu_stream_encoders`) — fixes PR #63 HTTP 500 `Stream(cpu, 0)` under `--use-mtp`

## P0-MTP gates (PR #63 close)

| Gate | Result | Evidence |
|------|--------|----------|
| M1 short HTTP MTP | **PASS** | `gates/raw/M1-short-v2.json` |
| M2 long HTTP MTP | **PASS** | `gates/raw/M2-long.json` |
| M3 thinking+MTP | **PASS** | `gates/raw/M3-*.json` |
| M4 CLI MTP | **PASS** | `gates/logs/M4-chat.log` |
| M6 pure XOR | **PASS** | `gates/logs/M6-xor.log` |

Details: `P0_MTP_GATES.md`, `gates/RESULTS.md`. Harness: `run_p0_mtp_gates.sh`.
