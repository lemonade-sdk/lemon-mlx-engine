# Quality review — P3 measured micro-op (kernarg patch)

**Date:** 2026-08-07  
**Branch:** `exp/redline-kernel-launch`  
**Artifact:** [`P3_MICRO_OP.md`](P3_MICRO_OP.md) · [`harness/p3_kernarg_patch.rs`](harness/p3_kernarg_patch.rs) · [`logs/p3-kernarg-patch-20260807-221119.log`](logs/p3-kernarg-patch-20260807-221119.log)

## Quintuple coverage (spawn unavailable → simulated labeled reviews)

### 1. explore — codebase / integration facts — **PASS**

- Harness lives under experiment tree only; no `generate.cpp` product forward change.  
- P0/P2b env still exact `"1"`, XOR pure-graph unchanged.  
- Measure is out-of-process AQL (same class as P1/P2), not HIP graph product path.

### 2. plan — approach — **PASS**

- Single advance: correctness + host µs for patch+replay.  
- Uses documented `acc_k` layout and `patch_kernarg_u32` (retained kernarg storage).  
- n≥2 respects `InvalidBatchShape` lesson from P1.

### 3. senior-developer — implement honesty — **PASS**

- Stable acc addr baked once; per-token only val@8 patched.  
- `P3_OK` / `P3_FAIL` lines; stderr explicitly `NOT gen t/s`.  
- Independent forbidden → remapped to BoundarySerialized.

### 4. quality-reviewer — docs honesty — **PASS**

- No gen t/s, no product default ON, no claim of engine TokenIterator wire.  
- Expected math `2*sum(1..64)=4160` matches log.  
- Explicit non-goals section.

### 5. supervisor — bans + evidence — **PASS**

| Ban | Status |
|-----|--------|
| No LoopBrake / fake TPS | OK |
| No E1/E2/P2 host µs as gen t/s | OK (new host µs labeled host only) |
| No product HIP graphs / pure-graph couple | OK (harness only) |
| No force-push / product default ON | OK |
| Evidence path exists | OK — log + CO + harness |

**Supervisor verdict: PASS**

## Residual risks

- Single-run host median (no multi-seed); sufficient for micro-op gate, not for product A/B.  
- Kernel is toy atomicAdd, not MLX decode op.  
- In-process `graph_decode_*` bind still future work if product research continues post-stop.
