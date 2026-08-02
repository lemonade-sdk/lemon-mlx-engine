# E1 — Dispatch-floor microbench on gfx1150

**Date:** 2026-08-02  
**Branch:** `exp/redline-kernel-launch`  
**Host:** gfx**1150** (Radeon 890M) · ROCm Core **7.13.0** · HIP **7.13.99004**  
**Redline:** warpfront `/tmp/redline-warpfront` @ `b505a72`  
**HSACO:** `floor_kernel-gfx1150.co` (`floor_k` no-op)  
**Verdict:** **AQL fence spectrum MEASURED** · **PM4 IB tail FAIL** (example hardcodes gfx12) · process **EXIT 1** after successful AQL table  

**Not claimed:** gen t/s, 35B decode win, product HIP-graph path, PM4 IB win on this GPU.

---

## Command (repro)

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
export REDLINE_FLOOR_HSACO=/tmp/redline-warpfront-hsaco/floor_kernel-gfx1150.co
# defaults: N=64, M=200 timed, warmup=20
/tmp/redline-warpfront-target/release/examples/dispatch_floor
```

**Log:** [`logs/e1-dispatch-floor-gfx1150-20260802-142850.log`](logs/e1-dispatch-floor-gfx1150-20260802-142850.log)

---

## Results (AQL retained batch — GPU-timed span)

Same retained batch of **N=64** no-op dispatches, median of **M=200** timed replays after **20** warmups.  
`us` = full batch span (µs); `us/disp` = `us / N`; `vs floor` = `SystemEveryDispatch_us / policy_us`.

| policy | us (batch) | us/disp | vs floor |
|--------|----------:|--------:|---------:|
| **SystemEveryDispatch** (HIP per-dispatch floor model) | 130.486 | **2.0388** | 1.000× |
| SystemAcquireAgentRelease | 123.512 | 1.9299 | 1.056× |
| AgentEveryInternalDispatch | 108.276 | 1.6918 | 1.205× |
| **BoundarySerialized** (Redline safe / decode-shaped) | 68.328 | **1.0676** | **1.910×** |
| BoundaryIndependent (aggressive / independent-only) | 60.674 | 0.9480 | 2.151× |

### Reading (honest)

- On **this** GPU, for **this** no-op kernel and methodology, Redline’s **BoundarySerialized** fence policy is about **1.91×** faster than system-fence-every-dispatch (~**2.04 → ~1.07 µs/dispatch**).  
- Historical Redline docs cited ~**1.8×** on other silicon/ROCm eras; **1.91× here is consistent in class**, not a copy of those tables.  
- **BoundaryIndependent** is slightly faster but is **not** the decode-safe policy when dispatches share writable state.  
- These are **dispatch-floor** numbers, not MLX token throughput.

---

## PM4 retained-IB section (failed)

After the AQL table, `dispatch_floor` always runs host-timed **PM4 IB** via `Gfx12Pm4CommandBuffer`:

```text
Error: ArchitectureMismatch { required: "gfx12", actual: "gfx1150" }
EXIT:1
```

| Item | Note |
|------|------|
| Cause | Example **hardcodes Gfx12** PM4 encoder (`measure_pm4_ib_host` in `dispatch_floor.rs`) |
| Library support | `redline-dispatch` `graph_pm4.rs` maps **`gfx11*` → Gfx11 / legacy Gfx10 encoder** — not used by this example’s PM4 tail |
| E1 impact | **Does not invalidate** the AQL fence-policy table printed above |
| Follow-up | Optional: gfx11 PM4 smoke via library API (not required to close E1 AQL) |

---

## Environment checklist

| Item | Value |
|------|--------|
| GPU | gfx1150 · AMD Radeon 890M |
| ROCm | 7.13.0 (`/opt/rocm/core`) |
| Binary | `/tmp/redline-warpfront-target/release/examples/dispatch_floor` |
| CO | `/tmp/redline-warpfront-hsaco/floor_kernel-gfx1150.co` (also under `logs/`) |

---

## Board status

| Gate | Status |
|------|--------|
| µs/dispatch numbers on **this** GPU | **YES** (AQL table) |
| Explicit fail log where applicable | **YES** (PM4 gfx12 mismatch) |
| Gen t/s / engine wire | **NO** — out of scope for E1 |

**Next:** **E2** — toy multi-kernel retained AQL vs HIP wall (or fixed N-launch chain), still on gfx1150; keep product decode graphs OFF.
