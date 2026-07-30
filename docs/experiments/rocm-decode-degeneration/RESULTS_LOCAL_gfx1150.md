# Local ROCm decode degeneration — results (gfx1150)

**Host:** Ubuntu, AMD Radeon 890M (`gfx1150`), ROCm present  
**Binary:** `build/chat` (ROCm-linked)  
**Model:** `mlx-community/Qwen3.5-0.8B-4bit` (hits `qwen35_moe` fused GDN path)  
**Date:** 2026-07-30  
**Constraints:** MTP off, pure-graph unset, eager decode, temp=0, single process  

## Pass definition used

| Term | Rule |
|------|------|
| **Hard loop** | Same 12-word phrase repeated ≥6 times in one turn, or same long line ≥6 |
| **Budget** | `--max-tokens 400` multi-turn; `128` short smoke |

## Repro (locked)

**Command (thinking on, multi-turn):**

```bash
printf '%s\n' \
  'Explain how a phased array radar steers a beam without moving antennas.' \
  'Now simplify for a non-expert.' \
  'What fails if phase synchronization drifts?' \
  'who are you?' \
  'quit' | ./build/chat mlx-community/Qwen3.5-0.8B-4bit \
  --temperature 0 --max-tokens 400
```

**Baseline (pre-fix, fused2 default ON):** hard loop from turn 1 on  
`standard phased array radar uses moving antennas to steer the beam (or…)`  
Logs: `logs/B0c_0.8B_multiturn_r1.txt`, `logs/B0_reconfirm.txt`.

**Controls:**

| Cell | Result |
|------|--------|
| Short `who are you?` + `--no-think` | **PASS** (no loop) |
| Multi-turn + `--no-think` | **PASS** (no loop) |
| Multi-turn + thinking | **FAIL** hard loop (field-like) |

## Env ladder (old binary, fused2 still default ON)

| Cell | Env | Hard loop? | Notes |
|------|-----|------------|-------|
| B0 | (none) | **Y** turns 1–4 | radar phrase |
| G1 | `MLX_GDN_NO_FUSED2=1` | turn1–3 **N**, turn4 **Y** | **H1 partial clear** |
| G2 | `MLX_GDN_NO_FUSED=1` | near-clear (maxphrase≤5) | strongest |
| C1 | `MLX_GDN_CONV_MXOPS=1` | **Y** like B0 | conv alone not root |
| P1 | `MLX_SYNC_DECODE=1` | **Y** like B0 | **H2 refuted** |
| P2 | `MLX_KV_INPLACE_OFF=1` | **Y** like B0 | **H2 refuted** |

**Outcome lock: S1 (H1)** — GDN fused recurrence path, not async/KV pipeline.

## Code fix applied (local)

File: `src/llm/models/qwen35_moe.cpp`

1. **`gdn_fused_decode` is opt-in** via `MLX_GDN_FUSED2=1` (exact `"1"`).  
   Default path = external `rms_norm` + `gated_delta_update` (old `MLX_GDN_NO_FUSED2` behavior).  
   `MLX_GDN_NO_FUSED2` still force-disables fused2 if set with opt-in.
2. **Cast recurrence outputs** (`o` / `ns`, and compile-path `out`/`ns`) back to model `dtype` before `norm_` / cache write. Without this, non-fused2 hit monomorphic HIP `gated_rms_norm` / residual dtype mismatches or garbage text after rebuild.

## Post-fix verification

| Cell | Result |
|------|--------|
| Default (no env), short no-think | **PASS** coherent (log `FIX2_default_nothink.txt`) |
| Default, multi-turn thinking | turn1 improved; turns 2–4 still soft/hard loop on “Wait, I need to check…” (`FIX2_default_mt.txt`) — **better than baseline turn1 radar lock**, not fully clean |
| `MLX_GDN_FUSED2=1`, multi-turn | still hard loops turns 2–4 (`FIX2_FUSED2_on_mt.txt`) — opt-in keeps old kernel for kernel debugging |

## Remaining work (not closed)

1. **Multi-turn residual loops** under thinking even without fused2 — may need ChatSession/template/thinking interaction investigation, or further GDN/state work.  
2. **True kernel fix** for `gdn_fused_decode` (numerics vs `gated_delta_update`) so fused2 can return as default for perf.  
3. **Field-size model** (35B hybrid) re-run — 0.8B must not over-claim.  
4. **q-norm `1/D` vs `1/√D`** A/B after path isolation.  
5. Move durable docs under `docs/experiments/rocm-decode-degeneration/`; do not treat root `ROCM_DECODE_DEGENERATION_TEST_PLAN.md` as proven root-cause claim.

## Supervisor consensus (quintuple)

| Lens | Verdict |
|------|---------|
| Code audit (explore) | Env names real; chat does not print gfx (use rocm-smi); MoE-only fused2 envs |
| QA protocol | Plan needed defs; early-stop bad; we ran full matrix |
| Senior ROCm | H1 first correct; H2 over-bundled; conv missing from original plan |
| Planning | Ubuntu-first correct; success codes S1–S5 |
| Local science | **H1 supported, H2 refuted** on this host/model |

## Bottom line

**Locally resolved for the catastrophic default fused2 hard-loop class** by making `gdn_fused_decode` opt-in and stabilizing dtypes on the safe path.  
**Not fully resolved** for all multi-turn thinking residual loops.  
**Do not wait on CI** to continue; next local step is residual multi-turn + optional 35B confirmation.
