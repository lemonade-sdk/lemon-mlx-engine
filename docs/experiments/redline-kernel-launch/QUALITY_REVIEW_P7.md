# Quality review — P7 L1 sidecar

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Artifact:** [`P7_SIDECAR_L1.md`](P7_SIDECAR_L1.md) · `redline_decode_session.cpp` · `generate.cpp` · `logs/p7-*-20260808-113934.err`

## Quintuple (simulated)

### 1. explore — **PASS**
- P6 residual: consume retained path on L=1 without call_fn replace.  
- Env-gated SIDECAR default OFF; XOR pure unchanged.

### 2. plan — **PASS**
- Arm after micro; dedicated hipMalloc acc; inline correctness then L=1 ticks.  
- No gen t/s claim.

### 3. implement — **PASS**
- Prime rebind fix for off-by-one (135→136).  
- Build chat exit 0.  
- `maybe_redline_sidecar_l1` on L=1 only.

### 4. quality-review — **PASS**
- Logs: off / skip / micro skip sidecar / **sidecar PASS 136** / xor.  
- Explicit NOT gen t/s.

### 5. supervisor — **PASS**

| Ban | Status |
|-----|--------|
| No fake TPS | OK |
| No HIP graphs / pure couple | OK |
| No product default ON | OK |
| Evidence | OK |

**Supervisor verdict: PASS**

## Residual
- Full-gen L=1 tick verification needs successful model load.  
- Toy acc_k only.  
- Process-lifetime leak of gpu/ib (research OK).
