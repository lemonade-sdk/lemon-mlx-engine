# Quality review — P5 in-process micro-op

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Artifact:** [`P5_INPROC_MICRO.md`](P5_INPROC_MICRO.md) · `src/common/redline_decode_session.cpp` · `logs/p5-*-20260808-112653.err`

## Quintuple coverage (simulated labeled reviews)

### 1. explore — facts — **PASS**

- Session remains opt-in exact `MLX_REDLINE_DECODE=1`; default silent.  
- Micro only when `MLX_REDLINE_HSACO` set; otherwise `micro=skip`.  
- XOR pure-graph fail-closed unchanged.  
- Standalone `decode_kernargs` PASS on gfx1150 confirms C-API PM4 path before engine wire.  
- Forward still product; no HIP-graph enable.

### 2. plan — approach — **PASS**

- One primary advance: product-adjacent in-process correctness gate.  
- Reuses proven CO + kernarg layout `[acc@0][val@8]`.  
- Does not claim gen t/s; host_total_us labeled.  
- Scope fits one fire (session module + docs + smoke).

### 3. implement — honesty — **PASS**

- dlsym symbols; no hard link `MLX_LM_WITH_REDLINE` required.  
- HIP platform macro local to translation unit.  
- Failures append `micro=FAIL_*` without failing gpu_new Ready path incorrectly (load still Ready if gpu ok).  
- Build `chat` exit 0.

### 4. quality-review — evidence — **PASS**

- Logs: off 0×; skip; PASS 2080/2080; xor.  
- Doc states expected 2080 (single dispatch) vs P3 4160 (n=2 AQL).  
- Explicit non-goals: gen t/s, default ON, call_fn replace.

### 5. supervisor — bans + evidence — **PASS**

| Ban | Status |
|-----|--------|
| No LoopBrake / fake TPS | OK |
| No relabel host µs as gen t/s | OK (`NOT gen t/s` in banner) |
| No HIP graphs / pure-graph couple | OK (XOR + no graph flags) |
| No force-push / product default ON | OK |
| Evidence paths exist | OK |

**Supervisor verdict: PASS**

## Residual risks

- Chat model CLI 401 in this smoke environment; session probe is pre-model (still valid).  
- PM4 single-dispatch micro is not AQL BoundarySerialized multi-N floor (P1/P2/P3).  
- Toy `acc_k`, not MLX decode op.  
- No gen A/B until a real product path is replaced.
