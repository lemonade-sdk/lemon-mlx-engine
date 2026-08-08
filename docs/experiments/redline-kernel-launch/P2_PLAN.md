# P2b — Engine Redline session init (plan; E4 naming)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PLAN** (deferred)  
**Depends on:** P0 GREEN · P1 PASS · **loop P2 N-sweep GREEN** ([`P2_NSWEEP.md`](P2_NSWEEP.md)) · E4 design

> **Naming:** Continuous-loop **P2** is the **N-sweep measure** in `P2_NSWEEP.md`.  
> This file is **E4 “engine session init”** (dlopen / READY banner) — call it **P2b** until scheduled.

---

## 1. Goal (E4 §5)

> Engine links or **dlopen**s `redline-capi` / `libredline_dispatch`; session **init** smoke on gfx1150.  
> **No change to forward** (`call_fn` still product HIP).

Success bar:

| Gate | Rule |
|------|------|
| Default | `MLX_REDLINE_DECODE` unset → **no** dlopen, **no** log |
| `=1` | One-shot log: either `session READY` (init OK) or `session FAILED → fallback` |
| Forward | Still product eager path (P2 does not replace kernels) |
| CMake | `MLX_LM_WITH_REDLINE` remains **OFF** default; optional link OR pure dlopen |

---

## 2. Preferred approach (critical)

| Option | Choice | Why |
|--------|--------|-----|
| **A. dlopen only when env=1** | **Primary** | Default binary free of Redline deps (E4 §8) |
| B. CMake `MLX_LM_WITH_REDLINE=ON` hard link | Optional research | Heavier; keep OFF in product |
| C. full forward replacement | **Out of scope P2** | That's P3 |

---

## 3. Component sketch

```text
// New: src/common/redline_decode_session.{h,cpp}  (ROCm only)
//   RedlineDecodeSession::try_init_once()
//     if env != "1": disabled
//     if XOR pure: fail-closed (already P0)
//     dlopen(libredline_dispatch.so) from REDLINE_LIB_DIR or default search
//     optional: load HSACO dir MLX_REDLINE_HSACO_DIR (not required for init-only)
//     build no-op / dual floor dispatch if CO present — OR init runtime only
//     state = READY | FAILED
//   on FAILED + MLX_REDLINE_FALLBACK=1: silent product path forever
//
// generate.cpp P0 banner upgrade:
//   if READY: "[redline] session READY (init only; forward still product)"
//   if FAILED: "[redline] session FAILED → fallback HIP"
```

**Minimal P2 (recommended first cut):**

1. `dlopen` + resolve a few symbols (or C-API version query if present).  
2. `Runtime::initialize`-class smoke **or** C-API device enum — host only.  
3. Do **not** call into MLX kernels yet.  
4. Log READY/FAILED once.

If C-API is thin, P2 may use a tiny **out-of-process** check + document that in-process dlopen waits on symbol list — still progress if honestly blocked.

---

## 4. Env (extends E4 §3)

| Env | Default | P2 use |
|-----|---------|--------|
| `MLX_REDLINE_DECODE` | 0 | Master switch exact `1` |
| `MLX_REDLINE_LIB` | unset | Optional full path to `.so` |
| `MLX_REDLINE_HSACO_DIR` | unset | Optional; not required for init-only |
| `MLX_REDLINE_FALLBACK` | `1` | On fail → product |
| `MLX_REDLINE_LOG` | `0` | Extra counts later |

---

## 5. Kill / pass

| Result | Action |
|--------|--------|
| dlopen + init READY on gfx1150 | **PASS P2** → start P3 design/code |
| dlopen fails (missing .so) with FALLBACK | **Soft FAIL** — document path; not hard blocker if repro with `REDLINE_LIB` works |
| Hard crash / GPU hang | **Hard blocker** — stop product attempts; file log |
| Accidentally changes forward without measure | **REVERT** |

---

## 6. Deliverables this phase

| ID | Artifact |
|----|----------|
| P2.a | `redline_decode_session` stub sources (or documented blocker) |
| P2.b | Smoke logs `logs/p2-init-*.err` |
| P2.c | `P2_INIT.md` PASS/FAIL |
| P2.d | QUALITY_REVIEW_P2 |

**Not P2:** gen t/s, MoE multipath, qmm offload.
