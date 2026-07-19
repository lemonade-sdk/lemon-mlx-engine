# Explicit confirmation: do we need a NripeshN/mlx PR?

**Date:** 2026-07-19  
**Question:** Is there a confirmed defect in [NripeshN/mlx `rocm-support`](https://github.com/NripeshN/mlx/tree/rocm-support) that **requires** an issue/PR there **now**?  
**Answer (this confirmation run):** **NO — do not open an mlx PR yet.**

---

## 1. Pins (this machine)

| Item | Value |
|------|--------|
| Engine | `lemon-mlx-engine` branch `fix/eager-no-mtp-correctness` (see `meta.env`) |
| mlx FetchContent | `0dadb703d77301af29405cf7e12627efb88a6d0f` (`rocm-support` tip at fetch) |
| Model | `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| GPU | gfx1150 (Radeon 890M), ~8.5 GiB VRAM reported |
| Posture | Eager; no MTP head (`MLX_LOAD_MTP_HEAD` unset); pure-graph off |

---

## 2. Evidence bar (from FIX_AND_VERIFY_GUIDE §8)

Open **NripeshN/mlx** only if **all** hold:

| # | Requirement | This run |
|---|-------------|----------|
| 1 | Repro on canonical model via engine (curl/CLI) | Historical SEGV had CLI path; **not reproduced exclusive today** |
| 2 | Engine mitigations tried (pure off, no MTP, etc.) | Yes |
| 3 | Failure is stream/capture/arena/kernel — not EOS/ChatSession/stop/thinking | Historical crash was HIP kernel during copy — **class** matches, but **not stable** |
| 4 | Package: error, gfx, mlx SHA, engine SHA, A/B | Partial historical gdb; **no exclusive multi-fail matrix** |

**Bar not met for “file mlx PR now.”**

---

## 3. What we thought might be “the mlx issue”

| Candidate | Layer | Status after confirmation |
|-----------|--------|---------------------------|
| Chat warmup `SIGSEGV` in `hipLaunchKernel` / `copy_contiguous` (bf16→f32) during GDN T=1 | mlx ROCm **or** engine call-site timing | **Potential / intermittent** — not exclusive-GPU reproducible in 3/3 cold chat loads today |
| Concurrent double 18 GB load (server+chat) | Ops / OOM-ish | **Plausible confounder** for earlier “chat-only” fails |
| Quant fuse `concatenate` → HIP SEGV | Engine path + HIP | **Mitigated in engine** (`MLX_ENABLE_QUANT_FUSE` opt-in) — no mlx PR required for product path |
| MTP `Stream(cpu, 0)` | Engine MTP + mlx stream TLS | **Deferred** (operator not enabling MTP) — do not file as active product block |
| Mid-forward `eval(astype)` of GDN constants | Engine call pattern into mlx | **Engine hygiene** still recommended (materialize at load); not proof of mlx bug alone |
| ChatSession / stop / thinking budget / pure default | Engine | **Engine-owned; already addressed** — not mlx |

---

## 4. Exclusive-GPU matrix (this confirmation)

**Pack:** `docs/experiments/mlx-need-confirm-2026-07-19/`  
**Log:** `logs/matrix.log`

| Run | Result |
|-----|--------|
| chat cold #1 | **PASS_LOAD** (`Model loaded`) |
| chat cold #2 | **PASS_LOAD** |
| chat cold #3 | **PASS_LOAD** |
| server cold health + `Say ping.` | **PASS** (`"Ping!"`) |
| SEGV under exclusive clean GPU | **0 / 3 chat** |

Historical SEGV (gdb in `verify-eager-no-mtp-…/logs/C1-chat-gdb.log`) remains **real as an observation**, but **not confirmed as a stable, exclusive, minimal mlx repro**.

---

## 5. Decision table (explicit)

| Decision | Verdict | Why |
|----------|---------|-----|
| Need **NripeshN/mlx** issue **now**? | **NO** | No exclusive reproducible fail; product path green |
| Need **NripeshN/mlx** PR **now**? | **NO** | Same; no minimal kernel-level A/B |
| Need more **engine** work? | **Optional hygiene only** | e.g. materialize GDN f32 constants at `load_weights` to avoid mid-forward `eval` |
| Product blocked on mlx tip? | **NO** for eager/no-MTP | Server decode + chat load + C1 Ada (prior) work under engine mitigations |
| Keep watching mlx? | **YES, low priority** | If exclusive SEGV returns ≥2/N on cold chat/server with pins frozen, **then** escalate |

---

## 6. When we **would** open mlx (future gate)

All of:

1. Exclusive GPU (no second model process).  
2. Frozen engine SHA + mlx SHA.  
3. Fail rate ≥ **2/5** cold chat **or** server warmup on same recipe.  
4. Still fails with engine hygiene applied (load-time constant materialize, pure off, no MTP, no quant fuse).  
5. Prefer fails with `MLX_GDN_NO_FUSED=1` too **or** isolates to `copy_contiguous` only.  
6. Bundle: full gdb, `rocm-smi`, curl/CLI recipe, “server holds 18G → second chat dies” A/B if relevant.

Until then: **do not** claim “we must patch NripeshN/mlx.”

---

## 7. One-sentence answer for humans

**There was a real, intermittent HIP/mlx-looking crash under some conditions, but under exclusive clean runs today it did not reproduce; product is unblocked by engine-side work, so we do not need an mlx PR until we can fail it cleanly and repeatedly.**

---

## 8. Where work actually lives

| Repo | What we did |
|------|-------------|
| **lemonade-sdk/lemon-mlx-engine** `fix/eager-no-mtp-correctness` | All commits/pushes |
| **NripeshN/mlx** | No issue, no PR, no push |

*Supervisors: Decode/ROCm, QA, Ownership, Product/Ops, Clear Thought (scientific method: independent cold runs, exclusive GPU, falsify “must be mlx”).*
