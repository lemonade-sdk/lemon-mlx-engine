# S4 — Batch-verify re-probe + n_draft=3 post-P0-B (06-tps-ceiling)

**Branch:** `exp/mtp-tps-ceiling` (child of `fix/mtp-stream-p0` @ `875a39d`)  
**Date:** 2026-08-01  
**Device:** gfx1150 / Radeon 890M (8 CU)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Spec:** `docs/analysis/mtp-review/06-tps-ceiling.md` §4  
**Verdict:** **KILL batch verify on this stack. Software plateau ~27 t/s sequential n_draft=2. Deep draft still loses post-P0-B.**

---

## 0. MTP_TIMING was on (do not re-litigate)

| Check | Evidence |
|-------|----------|
| Env | `MTP_TIMING=1` in `COMMON_ENV` with `MTP_DEBUG=1`, full fuse, `MLX_LOAD_MTP_HEAD=1` |
| Code gate | `generate.cpp`: `kMtpTiming = (std::getenv("MTP_TIMING") != nullptr)` → prints `[mtp-t]` |
| Logs | 141–145 `[mtp-t]` lines per n_draft=2 run; 115–134 for n_draft=3 |

Without `MTP_TIMING`, there would be **zero** `[mtp-t]` rows. All four logs have them.

### Timer semantics (critical read)

| Path | What `draft=` measures | What `verify=` measures | Wall truth |
|------|------------------------|-------------------------|------------|
| **Sequential (C4/C7)** | Side-stream draft **‖** first trunk T=1 (joint) | Residual second T=1 after join (~4 ms residual, **not** full T₁) | **`total=`** (~66.5 ms) and **gen t/s** |
| **Batch** | Serial draft only (~5 ms) | Multi-token trunk forward + accept (~67–87 ms) | **`verify=` on accept** and **gen t/s** |

C6-era rows had verify ≈ 35.7 ms = T₁ because sequential T=1 costs were attributed to verify. Current sequential path folds first verify into the draft timer. **§4 kill comparison for batch uses batch `verify` on accept (and gen t/s), not the sequential residual 4 ms field.**

---

## 1. Protocol (matches 06 §4)

```bash
PROMPT='Write a technical overview of the Fourier Transform for engineers.'
COMMON_ENV='MLX_ENABLE_QUANT_FUSE=1 MLX_ENABLE_QUANT_FUSE_GDN=1 MLX_LOAD_MTP_HEAD=1 MTP_DEBUG=1 MTP_TIMING=1'

# Seq baseline n_draft=2
env $COMMON_ENV ./build/chat MODEL --use-mtp --n-draft 2 --temperature 0 --top-p 1 \
  --max-tokens 256 --no-think --ignore-eos

# Batch T=2 (THE lever)
env $COMMON_ENV MLX_MTP_BATCH_VERIFY=1 ./build/chat ... --n-draft 2 ...

# n_draft=3 post-P0-B (seq + batch)
env $COMMON_ENV ./build/chat ... --n-draft 3 ...
env $COMMON_ENV MLX_MTP_BATCH_VERIFY=1 ./build/chat ... --n-draft 3 ...
```

Logs: `S4_seq_ndraft2.txt`, `S4_batch_ndraft2.txt`, `S4_seq_ndraft3.txt`, `S4_batch_ndraft3.txt`.

---

## 2. Pre-committed kill table (06 §4) vs measured

**Break-even wall per step = 67.7 ms** (C7 E[tokens]≈1.85 @ 27.34 t/s).  
**Not 55 ms** — that was a chat approximation; algebra uses 67.7.

| Batch T=2 verify (on accept) | Decision | This probe |
|------------------------------|----------|------------|
| ≤ 50 ms | Reopen product batch verify | no |
| 50–60 ms | Worth building | no |
| 60–67 ms | Marginal | no |
| **> 67.7 ms** | **KILL — plateau** | **YES: mean 77.1 ms, median 71.2 ms** |

| Config | gen t/s | vs seq 27.22 | Decision |
|--------|---------|--------------|----------|
| **S4 seq n2** | **27.216** | baseline (≈ C7 27.34) | keep default |
| **S4 batch n2** | **20.890** | **−23%** | **KILL** |
| S4 seq n3 | 18.290 | −33% | kill deep draft |
| S4 batch n3 | 10.152 | −63% | kill |

---

## 3. Warm-step stats (skip step 1 cold)

| Config | warm steps | mean accepted | mean draft ms | mean verify ms | mean total ms | verify_on_accept mean/med ms | gen t/s |
|--------|------------|---------------|---------------|----------------|---------------|------------------------------|---------|
| seq n2 | 140 | 0.814 | 63.2 | 3.2 | **66.5** | 3.9 / 3.7 (residual only) | **27.22** |
| **batch n2** | 144 | 0.771 | 4.7 | 79.2 | **84.0** | **77.1 / 71.2** | **20.89** |
| seq n3 | 114 | 1.219 | 92.9 | 29.0 | **121.9** | 37.1 / 37.8 | 18.29 |
| batch n3 | 133 | 0.902 | 15.7 | 172.7 | **188.5** | 175.7 / 101.4 | 10.15 |

Batch n2 sample (warm):

```
[mtp-t] step=2 ... draft=5019us verify=67651us total=72748us
[mtp-t] step=3 ... draft=4690us verify=67555us total=72341us
```

C1-era batch was ~86 ms verify / 2 tok. Post-fuse stack is **not** amortized under 67.7 ms; still ~2× T₁-class multi-token cost.

---

## 4. How this confirms 06 §§1–3 (not a bug hunt)

1. **§2 identity:** sequential speedup cancels to ~1.0; observed +~4% is overlap (`total` ≈ 66.5 ms, gen 27.2). Acceptance cannot invent free tokens.
2. **§3 archaeology:** batch was built, measured bad (C1 86 ms), killed (C2). S4 re-measures on fuse+P0 stack → still bad (77 ms / 20.9 t/s).
3. **§4 kill line = 67.7 ms** — applied to batch verify-on-accept; both mean and median clear it.
4. **n_draft=3 post-P0-B:** old 22.71 was invalid (KV starve). New valid row **18.29** still loses to n2 — deep draft stays dead on this machine.
5. **§8 prior 0.85 fail** — outcome matches prior; probe was required because fuse-stack delta was unmeasured.

---

## 5. Product decisions (HARD-BAN compliant)

| Action | Status |
|--------|--------|
| Open “build real batch verify” product WS | **NO** |
| Change default from sequential T=1 | **NO** |
| Keep `MLX_MTP_BATCH_VERIFY=1` as opt-in experiment only | YES (leave compiled) |
| Stamp software plateau ~27 t/s single-stream 35B @ 890M | **YES** |
| Fund C11–C15-class draft micro-opts | **NO** (inert under seq verify) |
| Next strategic work | **H1 dGPU day** or **H2 small-model** product surface; optional T₁ work (eager+MTP equal) |

No LoopBrake / auto-disable / dual-load / fake TPS.

---

## 6. Parent stack (“previous PRs behind”)

This branch is **child of** `fix/mtp-stream-p0`, which already stacks:

- #74 GDN / fused2 lineage  
- Selective quant-fuse (#76 policy)  
- StreamGuard / server TLS / P0-MTP  
- C1–C7 plateau code + RS + residual + registry  
- P0-B final-draft KV  

Lean product surface remains PR **#77** (`fix/mtp-product`); this branch is **experiment evidence only**.
