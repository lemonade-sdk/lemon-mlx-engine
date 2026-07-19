# Loop9 — comprehend mlx dual-load findings + reconcile ownership

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**Trigger:** Operator: findings in local mlx repo (`/home/antmi/mlx`) as well as engine.

## What we found in mlx (comprehended)

| Fact | Source |
|------|--------|
| Exclusive 35B chat load **PASS** (repeated) | `/home/antmi/mlx/docs/rocm-investigation-2026-07-19/` |
| Dual load (server holds ~18GB + chat loads) **FAIL exit 139** | same pack `logs/dual-*.log` |
| Stack class: `copy_contiguous` → `hipLaunchKernel` | same + engine historical gdb |
| Draft upstream docs PR | https://github.com/NripeshN/mlx/pull/13 |
| Head fork | https://github.com/antmikinka/mlx-rocm `investigate/rocm-dual-load-segv` |
| mlx SHA | `0dadb703` (`rocm-support`) |
| MTP Stream(cpu) / pure-graph | deferred (not re-opened) |

## Engine implications

| Implication | Action this loop |
|-------------|------------------|
| Product path remains **single process** | Docs + soft warn at load if GPU already tight |
| Do **not** block PR #63 on mlx kernel fix | Ownership table updated |
| Prior “no mlx PR” meant **exclusive code PR** | Clarified vs dual-load **docs** #13 |
| GDN materialize ≠ dual-load fix | Documented (M7 vs M1) |

## Code this loop

| Change | Purpose |
|--------|---------|
| `ModelManager` ROCm pre-load warn if free/total tight | Operator-visible dual-load risk (stderr) |

## Docs this loop

| Path | Role |
|------|------|
| `docs/experiments/mlx-need-confirm-2026-07-19/MLX_PR_NEED_CONFIRMATION.md` | Full reconciliation |
| `docs/LOOP_STATUS.md` | Loop9 |
| `docs/OWUI_OPS_CHECKLIST.md` | Dual-load + mlx#13 link |

## Claims honesty

| Claim | Allowed? |
|-------|----------|
| Dual-load SEGV confirmed on this machine | **Yes** |
| Exclusive eager product unblocked | **Yes** |
| mlx kernel fixed dual-load | **No** |
| Engine “gibberish” all fixed by mlx PR | **No** |
