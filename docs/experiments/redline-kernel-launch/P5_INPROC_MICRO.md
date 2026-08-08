# P5 — In-process C-API micro-op (engine session)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (in-process chat early probe; correctness only)  
**Depends on:** P0–P3 · P2b `gpu_new=ok` · standalone C-API `decode_kernargs` PASS on gfx1150  
**Not shipped:** product forward replacement · gen t/s claim · product default ON · HIP-graph re-enable · TokenIterator partial decode wire

---

## 1. Goal

Wire the **same** retained-PM4 load → patch kernarg → replay pattern that redline-capi `decode_kernargs.c` proves, **inside** the lemon-mlx engine session (`redline_decode_session`), still behind `MLX_REDLINE_DECODE=1` and **opt-in** HSACO path.

This is the first **product-adjacent** correctness smoke (engine binary + dlopen), not model gen throughput.

---

## 2. Env contract (all opt-in)

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_REDLINE_DECODE` | unset | Master; exact `"1"` only |
| `MLX_REDLINE_HSACO` | unset | Path to prebuilt CO; **unset → `micro=skip`** |
| `MLX_REDLINE_SYMBOL` | `acc_k.kd` | Kernel symbol in CO |
| `MLX_REDLINE_MICRO_TOKENS` | `64` | Patch+replay count (1..4096) |
| `MLX_REDLINE_LIB` | search path | Optional explicit `libredline_dispatch.so` |
| `MLX_DECODE_GRAPH_PURE=1` + REDLINE | — | XOR fail-closed (no session, no micro) |

Product default remains OFF. No coupling to HIP graphs.

---

## 3. Behavior

| Case | Result |
|------|--------|
| `DECODE` unset | 0× `[redline]`; no dlopen |
| `DECODE=1`, no `HSACO` | READY `gpu_new=ok micro=skip` |
| `DECODE=1` + `HSACO=acc_kernel…co` | READY `micro=PASS observed=expected … host_total_us=… (NOT gen t/s)` |
| XOR pure | fail-closed banner; no micro |

**Forward path:** still product `call_fn`. Micro runs once at session init (chat early probe and/or first L=1 log).

**Implementation:** [`src/common/redline_decode_session.cpp`](../../../src/common/redline_decode_session.cpp) — `try_micro_op`: HIP owns `d_acc`; Redline `rl_gpu_load_module` + single `rl_pm4_dispatch` + `rl_pm4_finalize` + T× (`rl_pm4_ib_set_kernargs` val@8 + `rl_pm4_replay`).

---

## 4. Smoke evidence (gfx1150)

Standalone precondition (same CO, host C-API):

```text
real-GPU C-ABI decode gate: acc = 2080 / 2080 over 64 tokens [PASS]
```

Engine chat early probe (`MLX_SKIP_WARMUP=1`; model load may 401 in this smoke env — **session probe is pre-load**):

| Case | Banner / count | Log |
|------|----------------|-----|
| off | 0× `[redline]` | [`logs/p5-off-20260808-112653.err`](logs/p5-off-20260808-112653.err) |
| on-skip | `micro=skip` | [`logs/p5-on-skip-20260808-112653.err`](logs/p5-on-skip-20260808-112653.err) |
| on-micro | **`micro=PASS observed=2080 expected=2080 tokens=64 host_total_us=769.283 (NOT gen t/s)`** | [`logs/p5-on-micro-20260808-112653.err`](logs/p5-on-micro-20260808-112653.err) |
| xor | XOR fail-closed | [`logs/p5-xor-20260808-112653.err`](logs/p5-xor-20260808-112653.err) |

CO: [`logs/acc_kernel-gfx1150.co`](logs/acc_kernel-gfx1150.co)  
Expected: `sum(1..64) = 2080` (single PM4 dispatch; unlike P3 AQL n=2 which expects 4160).

---

## 5. Repro

```bash
export PATH=/opt/rocm/core/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/core/lib:/tmp/redline-warpfront-target/release:$LD_LIBRARY_PATH
export MLX_REDLINE_DECODE=1
export MLX_REDLINE_LIB=/tmp/redline-warpfront-target/release/libredline_dispatch.so
export MLX_REDLINE_HSACO=$PWD/docs/experiments/redline-kernel-launch/logs/acc_kernel-gfx1150.co
export MLX_REDLINE_MICRO_TOKENS=64
# chat early probe prints READY micro=PASS even if model path fails later
./build/chat --model <local-snapshot> --max-tokens 8 --temperature 0 --raw -p hi
```

---

## 6. Honesty / non-goals

| Claim | Status |
|-------|--------|
| In-process PM4 patch+replay correctness | **YES** |
| Host total µs for T-token micro loop | **YES** (labeled NOT gen t/s) |
| Model gen t/s A/B vs eager | **NO** |
| Product default ON | **NO** |
| Replace qmm / call_fn | **NO** |
| E1 PM4 gfx12 hardcode path | **N/A** — C-API library PM4 (gfx11 map) used here; AQL still floor truth for multi-dispatch fence spectrum |

---

## 7. Next (not this fire)

- Optional: bind micro to `graph_decode_input` / pos stable ptrs (P3 design §3).  
- Optional: gen t/s A/B **only** after a real product-path op is replaced.  
- P4 MoE multipath remains design sketch.
