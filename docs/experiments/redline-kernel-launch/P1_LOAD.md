# P1 — Load + AQL replay correctness (measured)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (toy floor CO; out-of-process)  
**Harness:** [`harness/p1_load_hsaco.rs`](harness/p1_load_hsaco.rs)  
**Log:** [`logs/p1-load-hsaco-20260807-215318.log`](logs/p1-load-hsaco-20260807-215318.log)

---

## 1. Gate (E4 P1)

| Gate | Result |
|------|--------|
| Load HSACO/CO in Redline | **PASS** — `Executable::load` 8144 bytes |
| Resolve kernel symbol | **PASS** — `floor_k.kd` |
| Create retained AQL batch | **PASS** — `SingleQueueBatchGraph` BoundarySerialized |
| Warmup + timed replay | **PASS** — warmup 5, iters 20 |
| Gen t/s claim | **None** (host µs only) |
| Product wire | **None** |

---

## 2. Host / repro

| Item | Value |
|------|--------|
| GPU | gfx1150 |
| Redline tree | `/home/antmi/redline` (example install path only) |
| Cargo target | `/tmp/redline-warpfront-target` |
| CO | `docs/experiments/redline-kernel-launch/logs/floor_kernel-gfx1150.co` (E0/E1 artifact) |

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=/tmp/redline-warpfront-target

# Install harness into redline-dispatch examples (ephemeral; source of truth is this repo)
cp docs/experiments/redline-kernel-launch/harness/p1_load_hsaco.rs \
  /home/antmi/redline/crates/redline-dispatch/examples/p1_load_hsaco.rs
cd /home/antmi/redline
cargo build --release -p redline-dispatch --example p1_load_hsaco

REDLINE_P1_HSACO=$PWD/../lemon-mlx-engine/docs/experiments/redline-kernel-launch/logs/floor_kernel-gfx1150.co \
REDLINE_P1_SYMBOL=floor_k.kd REDLINE_P1_N=2 \
  /tmp/redline-warpfront-target/release/examples/p1_load_hsaco
```

---

## 3. Results

```text
[p1] device=gfx1150
[p1] Executable::load OK (8144 bytes)
[p1] SingleQueueBatchGraph create OK (n=2)
[p1] warmup OK (5)
[p1] host_median_us_per_replay=8.455 (NOT gen t/s)
P1_OK load+replay symbol=floor_k.kd n=2 host_median_us=8.455
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| host_median_us/replay | **8.455** | Host wall for n=2 no-op floor batch; **not** model gen t/s |
| n | 2 | Minimum for `SingleQueueBatchGraph` profiling shape |

### API note (blocker avoided)

First attempt with **n=1** failed:

```text
InvalidBatchShape("single-queue profiling batch requires at least two dispatches")
```

Harness defaults `REDLINE_P1_N` to **2** (`.max(2)`). Documented for P2/P3 subgraph builders.

---

## 4. What P1 does **not** prove

- MLX JIT `.hsaco` load (only E0 floor toy CO)  
- Product decode speedup  
- Engine `dlopen` / session (P2)  
- Replacing any product kernel (P3)  
- MoE multipath (P4)

**Partial next:** optional P1b — try one small MLX JIT module if cache present; failure is soft (document only).

---

## 5. Board

| Item | Status |
|------|--------|
| P1.a harness source in engine repo | **DONE** |
| P1.b env contract | **DONE** |
| P1.c log | **DONE** — `logs/p1-load-hsaco-20260807-215318.log` |
| P1.d this doc | **DONE** |
| P2 engine session init | **NEXT** |
