# Quality review — P6 graph_decode bind

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Artifact:** [`P6_GRAPH_DECODE_BIND.md`](P6_GRAPH_DECODE_BIND.md) · `graph_decode.cpp` · `redline_decode_session.cpp` · `logs/p6-*-20260808-113412.err`

## Quintuple (simulated)

### 1. explore — **PASS**
- P3 design residual: bake `graph_decode_*` for retained kernargs.  
- P5 micro proved PM4 path; P6 uses product pos as acc.  
- XOR / default OFF / no HIP-graph couple preserved.

### 2. plan — **PASS**
- Single advance: device-ptr helper + stability + bake correctness.  
- No gen t/s claim; host_total_us labeled.

### 3. implement — **PASS**
- `graph_decode_device_data_ptr` uses RocmBuffer layout (documented match).  
- Acc = pos VRAM ptr; restore `set_graph_decode_pos(0)` after smoke.  
- Build chat exit 0.

### 4. quality-review — **PASS**
- Logs: off 0×; skip; gd_bind+micro PASS 2080/2080; xor.  
- Explicit non-goals.

### 5. supervisor — **PASS**

| Ban | Status |
|-----|--------|
| No fake TPS / host µs as gen t/s | OK |
| No HIP graphs / pure couple | OK (XOR) |
| No product default ON | OK |
| Evidence paths | OK |

**Supervisor verdict: PASS**

## Residual
- RocmBuffer layout duplication if MLX backend ABI changes.  
- Early probe before model load; full gen path still product.  
- Toy `acc_k`, not decode qmm.
