# Actual issue analysis (critical re-open)

**Date:** 2026-08-01  
**Tip:** `fb6fc97`+ (quant-fuse selective branch)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` · gfx1150

## What we thought the issue was (wrong / incomplete)

| Belief | Status |
|--------|--------|
| “`MLX_ENABLE_QUANT_FUSE_GDN` / GDN in_proj fuse **is** the thrash root cause” | **Falsified** as sole cause |
| “Full fuse always fails Maxwell @ temp 0.7” | **Falsified** (FULL PASS this session) |
| “Selective GDN skip **fixed** thrash” | **Overclaimed** — skip co-landed with a more important change |

## What the issue actually is (layered)

### Layer 0 — REAL product issue (charter)

**Multi-turn GDN decode collapse** on ROCm hybrid Qwen: long chat stays OK for several turns then **token/phrase thrash**, especially on large code (Maxwell → Python).

| Root mechanisms | Status |
|-----------------|--------|
| Prefill ≢ decode recurrence / dtypes | **Fixed:** f32 SSM lifetime, prefill SSM f32 (`c7685d8`, `52d64de`) |
| Unstable softplus / g-dtype | **Fixed:** logaddexp softplus, cast `g` to act dtype |
| Fused2 kernel parity | **Landed** (auto-on; field matrix) |
| LoopBrake seatbelt | **Rejected / removed** (not a real fix) |
| H-KV stuck offsets | **Refuted** |

**Bar (from REAL_ISSUE.md):** Maxwell 5-turn SAR, no LoopBrake, usable Python.  
**Recent evidence:** SAFE and FULL fuse cells **PASS** without seatbelt.

→ **Layer 0 is largely resolved** by the GDN numeric stack (PR #74 lineage), not by quant-fuse policy.

### Layer 1 — Quant fuse × temp 0.7 (secondary, over-attributed)

| Evidence | Meaning |
|----------|---------|
| Tip `710135e`: FUSE@0.7 **FAIL** thrash mid-T5; NOFUSE@0.7 **PASS**; FUSE@0 **PASS** | Fuse env correlated with one thrash event under sampling |
| Same tip family: fuse used `concatenate` **without** `contiguous` | Packed quant layout may be wrong for `quantized_matmul` |
| Tip with `mx::contiguous(...)` packs: FULL fuse (`QUANT_FUSE`+`QUANT_FUSE_GDN`) @ **0.0 and 0.7** **PASS** | Full GDN pack is **not** inherently toxic on current code |
| SAFE (GDN skip) also PASS | Consistent with “packs fixed,” not “must skip GDN” |

**Best current model of Layer 1 thrash:**

1. **Primary suspect:** non-contiguous fused quant weight/scale packs → incorrect dequant → late multi-turn thrash under multinomial sampling.  
2. **Secondary:** residual sampling intermittency (n=1 historical FAIL, n=1 modern PASS).  
3. **Not supported:** “a/b softplus channels uniquely cannot be fused” as proven law.

### Layer 2 — Process / narrative issue (still real)

We encoded a **false certainty** into product comments and docs:

- GDN in_proj = thrash locus  
- `QUANT_FUSE_GDN` = debug-only poison switch  

That **misdirected** engineering attention away from (a) contiguous packs as the fuse correctness fix and (b) Layer 0 GDN numerics as the true product issue.

## What is resolved vs open

| Item | State |
|------|--------|
| GDN multi-turn collapse (default path) | **Resolved** for field bar (D7 + recent SAR) |
| Fuse thrash with contiguous packs | **No longer repro** on FULL@0.7 (single run) |
| Contiguous fuse packs in code | **Resolved** (must keep) |
| Causal docs / comments | **Corrected this commit** |
| `QUANT_FUSE_GDN` default skip | **Policy hold** (conservative), not proven necessary |
| Thrash rate under fuse (n≥3 seeds) | **Open** if product wants fuse default-on |
| MTP Stream(cpu,0) for TPS | **Open** (PR #63 deferred; separate) |
| PR #74 human merge | **Open** (process) |

## What “resolve” means now

1. **Stop treating selective GDN fuse skip as the root-cause fix.**  
2. **Keep `mx::contiguous` on all fuse packs** — non-negotiable.  
3. **Keep quant fuse opt-in** (memory + incomplete multi-seed rate).  
4. **Keep `QUANT_FUSE_GDN` optional** until n-run rate justifies defaulting GDN pack on.  
5. **Ship Layer 0 GDN correctness** (PR #74) as the real multi-turn product fix.  
6. Do **not** reintroduce LoopBrake.

## Logs (this session)

| Log | Cell |
|-----|------|
| `logs/FIELD_SAR_35B_FUSE_SAFE_temp07_think.txt` | QUANT_FUSE=1, GDN off, temp 0.7 PASS |
| `logs/FIELD_SAR_35B_FUSE_FULL_temp0_think.txt` | QUANT_FUSE+GDN, temp 0 PASS |
| `logs/FIELD_SAR_35B_FUSE_FULL_temp07_think.txt` | QUANT_FUSE+GDN, temp 0.7 PASS |

Historical FAIL log for `FIELD_SAR_35B_FUSE_temp07_think` was not retained in-tree (experiment archives dropped); narrative remains in `RESOLUTION_FUSE_TEMP07` (docs branch) and git history of `ROCM_TPS_…`.
