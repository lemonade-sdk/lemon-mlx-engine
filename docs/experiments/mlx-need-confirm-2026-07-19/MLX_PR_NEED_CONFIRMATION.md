# mlx ownership confirmation (reconciled with dual-load investigation)

**Date:** 2026-07-19 (loop9 reconciliation)  
**Question:** Do we need a **code** PR on [NripeshN/mlx `rocm-support`](https://github.com/NripeshN/mlx/tree/rocm-support) for the **eager product path**?  
**Short answer:** **No code PR for exclusive single-process product.**  
**Separate confirmed finding:** **dual-process large-model load → SIGSEGV** — documented (investigation), not fixed in-kernel yet.

---

## 1. Pins

| Item | Value |
|------|--------|
| Engine | `lemon-mlx-engine` `fix/eager-no-mtp-correctness` (PR #63) |
| mlx FetchContent | `NripeshN/mlx` tag `rocm-support` (local tip `0dadb703…`) |
| Local mlx clone | `/home/antmi/mlx` branch `investigate/rocm-dual-load-segv` |
| Model | `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| GPU | gfx1150 (Radeon 890M) |
| Posture | Eager; no MTP decode; pure-graph off |

---

## 2. Two different questions (do not collapse)

| Question | Answer | Where |
|----------|--------|--------|
| Is **exclusive** eager load/decode broken on mlx tip? | **NO** (product green) | Engine loop6/loop8 packs |
| Does **second concurrent** ~18 GB load SEGV? | **YES** (exit 139) | mlx pack + dual A/B |
| File **kernel “fix GDN copy”** PR without isolation? | **NO** | Ownership |
| Document dual-load for upstream? | **YES (docs draft)** | [NripeshN/mlx#13](https://github.com/NripeshN/mlx/pull/13) |

### Upstream draft (investigation only)

| Link | Role |
|------|------|
| https://github.com/NripeshN/mlx/pull/13 | Docs draft: exclusive OK / dual FAIL |
| Head fork | https://github.com/antmikinka/mlx-rocm (`investigate/rocm-dual-load-segv`) |
| Pack in mlx tree | `docs/rocm-investigation-2026-07-19/RESULTS.md` |

**Not a product blocker:** single-process engine path is the supported operator posture.

---

## 3. Candidate inventory (M#)

| ID | Symptom | Confirmed? | Ownership | Action |
|----|---------|------------|-----------|--------|
| **M1** | 2nd process load SIGSEGV after MTP-skip (`copy_contiguous` → `hipLaunchKernel`) | **YES** dual A/B | mlx ROCm / HIP ungraceful OOM-ish + **ops dual-load** | Ops: one process; engine: soft warn if GPU already tight; upstream: docs #13, code later if pure-mlx repro |
| **M2** | Exclusive chat/server load | **PASS** | — | Product path |
| **M3** | MTP `Stream(cpu, N)` | Earlier with `--use-mtp` | engine MTP + mlx stream TLS | **Deferred** (operator no MTP) |
| **M4** | Pure-graph capture | Not exercised | deferred | Operator eager-only |
| **M5** | ChatSession / stop / thinking / tools 400 | Engine | **engine PR #63** | Shipped |
| **M6** | Quant fuse concat SEGV | Mitigated | engine opt-in fuse | Shipped |
| **M7** | Mid-forward GDN bf16→f32 eval | Hygiene | engine `materialize_decode_constants` | Shipped (`a4d1d99`) — **does not replace M1 dual-load rule** |

---

## 4. Evidence bar (FIX guide §8) — exclusive product

Open **code** PR on mlx only if exclusive path fails repeatedly after engine hygiene:

| # | Requirement | Exclusive product |
|---|-------------|-------------------|
| 1 | Repro on canonical model | Exclusive loads **PASS** |
| 2 | Engine mitigations applied | Yes (no MTP, pure off, fuse off, materialize) |
| 3 | Kernel/stream class failure stable | Dual-load yes; exclusive **no** |
| 4 | Package A/B + pins | Dual A/B in mlx pack |

**Exclusive product:** do **not** open kernel fix PR.  
**Dual-load:** docs/investigation OK (#13); code fix needs pure-mlx microbench or accepted multi-process policy.

---

## 5. Decision table

| Decision | Verdict |
|----------|---------|
| Need mlx **code** PR for eager single-process product? | **NO** |
| Need mlx **docs/issue** for dual-load SEGV? | **Done as draft PR #13** (optional keep open) |
| Product blocked on mlx tip? | **NO** (one process) |
| Engine still owns gibberish UX? | **Mostly yes** — thinking/ChatSession/stop/OWUI Memory; not dual-load SEGV |
| Keep watching mlx? | **YES** — dual-load robustness; MTP streams when productized |

---

## 6. One-sentence human summary

**Exclusive 35B eager works with engine mitigations; running a second full 35B process while the first still holds the model can SIGSEGV in mlx/HIP — treat that as ops + optional upstream robustness, not as “decode is broken.”**

---

## 7. Where work lives

| Repo | What |
|------|------|
| **lemonade-sdk/lemon-mlx-engine** PR #63 | Product correctness (ChatSession, stop, thinking, load hygiene, OWUI H0) |
| **NripeshN/mlx** PR #13 | Dual-load investigation docs (no kernel fix yet) |
| **/home/antmi/mlx** | Local clone tracking `investigate/rocm-dual-load-segv` |

*Loop9 supervisors: Clear Thought + ownership — reconcile exclusive “no PR” with dual-load “yes M1.”*
