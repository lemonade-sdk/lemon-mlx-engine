# Loop8 — GDN materialize empirical + OWUI live H0

**Date:** 2026-07-19  
**Branch tip at smoke:** `a4d1d99` + docs  
**Binary:** `build/server` with `Qwen35MoEGatedDeltaNet::materialize_decode_constants`  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Posture:** eager, no `--use-mtp`, quant fuse off, single process  
**Hardware:** gfx1150 890M

## Supervisors

| Role | Loop8 focus |
|------|-------------|
| Decode/ROCm | Verify load-time GDN constant materialize on 35B |
| API/Client | Live O0a + O5 while server up |
| QA | Minimal smoke only (not full loop6 matrix) |
| Ownership | Engine-only; no mlx PR |
| Product/Ops | Evidence pack + LOOP_STATUS |

Clear Thought: collaborative + cause_elimination + sequential.  
Quality-reviewer on `a4d1d99`: **ship after smoke**; claim is load-time T=1 hygiene only (not mlx kernel root cause).

## Code under test (`a4d1d99`)

- Prefetch `q_norm_w_`, `k_norm_w_`, `a_log_f32_`, `dt_bias_f32_` at `load_weights`
- Goal: avoid first T=1 mid-forward bf16→f32 `copy_contiguous` / `hipLaunchKernel` race

## Empirical gates

| ID | Check | Result | Evidence |
|----|-------|--------|----------|
| Load | health, ~18.2 GB, MTP head skip log | **PASS** | `server.log` |
| **S1** | `2+2` → `4` | **PASS** | `raw/S1-2plus2.json` |
| **O0a** | `/v1/models` lists canonical MLX id (loaded) | **PASS** | `raw/O0a-models.json` |
| **O5** | `role:tool` → HTTP **400** + Memory guidance | **PASS** | `raw/O5-role-tool.txt` |

```text
LOOP8_SMOKE_OK
S1 '4' stop
O5 HTTP 400 + Memory/tools message
```

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Single-process 35B load+warmup+decode OK with materialize binary | **Yes** (this run) |
| Materialize eliminates all gfx115x GDN SIGSEGVs forever | **No** — intermittent dual-process/ops still exist; residual NO_FUSED/prefill paths |
| MTP / pure-graph fixed | **No** — not in scope |
| OWUI Memory product complete | **No** — O5 still 400 by design |

## Operator

```bash
# one process only
./build/server LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --host 127.0.0.1 --port 8080 --no-think
```
