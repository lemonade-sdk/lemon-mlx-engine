# T1 attack — absolute gen t/s (not MTP relative)

**Branch:** `exp/mtp-t1-attack`  
**Parent:** `fix/mtp-stream-p0` @ `875a39d` (sibling of `exp/mtp-tps-ceiling`, `exp/mtp-c11-topk-close`)  
**Date:** 2026-08-01  
**Device:** gfx1150 / 890M · Model: LemonMLXE 35B-A3B-MTP-mlx-4bit  
**Prompt:** Fourier technical overview · 256 tokens · temp=0 · `--ignore-eos` · `--no-think`  
**Honesty rule:** Wins here are **T₁ / trunk** wins. Do **not** rebrand as MTP speculative speedup.

---

## 1. Matrix (same session, wall-clock Generation line)

| Cell | Config | gen t/s | Δ vs eager nofuse |
|------|--------|---------|-------------------|
| **T1_eager_nofuse** | no `QUANT_FUSE` | **28.900** | baseline |
| T1_eager_safe_fuse | `QUANT_FUSE=1` (attn/MLP; GDN in_proj off) | **29.500** | **+2.1%** |
| T1_eager_full_fuse | + `QUANT_FUSE_GDN=1` | **29.786** | **+3.1%** |
| T1_eager_safe_kv8 | safe fuse + `--kv-bits 8` | 29.664 | +2.6% (≈fuse noise) |
| T1_eager_safe_kv4 | safe fuse + `--kv-bits 4` | 29.651 | +2.6% |
| T1_mtp_safe_fuse | MTP n_draft=2 + safe fuse | **27.062** | −6.4% vs nofuse eager |
| T1_mtp_full_fuse | MTP + full fuse | **27.748** | −4.0% |
| T1_mtp_safe_kv8 | MTP + kv8 | 27.117 | — |
| T1_mtp_safe_kv4 | MTP + kv4 | 27.111 | — |

Runner: `run_t1_matrix.sh` + `T1_eager_nofuse.txt`. Logs: `T1_*.txt`.

**Note vs historical 26.13 eager:** older C-ladder probe; this session’s nofuse is **~28.9**. Use **within-session deltas**, not cross-day absolute claims.

---

## 2. dense_kept audit (W3.3) — **CLOSED**

With `MTP_DEBUG=1` / `MLX_MTP_LOG_DENSE` (code on this branch):

```
dense_kept=7
dense_kept keys: norm.weight layers.0.self_attn.q_norm.weight
  layers.0.post_attention_layernorm.weight pre_fc_norm_embedding.weight
  pre_fc_norm_hidden.weight layers.0.self_attn.k_norm.weight
  layers.0.input_layernorm.weight
```

**All 7 are RMSNorm weights** (correctly non-quantized). **Zero** residual dense linears on the MTP head.  
`auto_quantized=13` already covers all quantizable linears. **No T₁ left in “quant more of dense_kept.”**

Code: list keys when `MLX_MTP_LOG_DENSE=1` or `MTP_DEBUG=1` (`mtp_head.cpp`).

---

## 3. Verdicts by lever

| Lever | Result | Product |
|-------|--------|---------|
| **Quant fuse SAFE** | ~**+2%** eager vs nofuse (this day) | Keep opt-in `MLX_ENABLE_QUANT_FUSE=1` |
| **Quant fuse + GDN in_proj** | ~**+1%** more at **temp=0** | Opt-in `MLX_ENABLE_QUANT_FUSE_GDN=1` only where quality allows (temp0.7 thrash history) |
| **KV quant 4/8** | **Flat** @256 **and** @~2k prefill (r2: +1.0–1.4% ≪5%) | **KILL/park** — see `LONGCTX_KV.md` + `T1L_eager_*.txt` |
| **dense_kept** | **No linears left** | Closed |
| **MTP vs eager** | MTP still ≈ eager − few % on this stack | Plateau story unchanged; don’t credit MTP for fuse |

Kill / fund rules used:
- KV: need ≥5% gen t/s or long-ctx bandwidth story → **not met** at 256 tok  
- dense_kept: need quantizable matmul weights → **none**

---

## 4. What this does *not* reopen

- Batch verify (S4 **KILL**)
- C11–C15 draft fuses (**dead**)
- Claiming 100 t/s on 35B@890M
- LoopBrake / fake TPS

---

## 5. Recommended product posture

1. Ship/keep **SAFE quant fuse** as the default TPS knob for decode when quality-approved.  
2. **GDN in_proj fuse** remains double-gated; enable only after quality bar at target temp.  
3. **KV quant** — **parked** after long-ctx r2 KILL (`LONGCTX_KV.md`).  
4. **dense_kept** — no further work.  
5. Further absolute t/s: **kernel/ROCm T₁**, **H1 dGPU**, **H2 small model** — not more MTP draft thrash.

---

## 6. Code on this branch

| Change | Why |
|--------|-----|
| `mtp_head.cpp` dense_kept key list | Prove W3.3 closed (norms only) |
| `run_t1_matrix.sh` | Reproducible A/B |
| This RESULTS + README | Field record |
