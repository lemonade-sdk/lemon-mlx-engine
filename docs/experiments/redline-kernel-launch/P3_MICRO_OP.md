# P3 — measured micro-op: kernarg patch + AQL replay

**Date:** 2026-08-07  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (toy acc CO; out-of-process; host µs only)  
**Depends on:** P0–P2b GREEN · P3 design doc  
**Not shipped:** product forward replacement · gen t/s claim · product default ON · HIP-graph re-enable

---

## 1. Goal

Measure one **fixed** Redline micro-op that models the P3 `graph_decode` contract:

1. Bake a **stable** device buffer address into kernargs once at record time  
2. **Patch** only a per-token scalar in-place (`patch_kernarg_u32`)  
3. **Replay** retained AQL `SingleQueueBatchGraph` (BoundarySerialized) without rebuild  
4. Prove **correctness** (device accumulator) and report **host µs** only

This is **not** model generation throughput.

---

## 2. Harness / artifact

| Item | Path |
|------|------|
| Source | [`harness/p3_kernarg_patch.rs`](harness/p3_kernarg_patch.rs) |
| Kernel | `/home/antmi/redline/bench/acc_kernel.hip` → `acc_k` |
| CO | [`logs/acc_kernel-gfx1150.co`](logs/acc_kernel-gfx1150.co) |
| Log | [`logs/p3-kernarg-patch-20260807-221119.log`](logs/p3-kernarg-patch-20260807-221119.log) |

**Kernarg layout (acc_k):** `[acc:u64 @0][val:u32 @8]`

**Shape:** `n=2` dispatches (API min for profiling batch), both write the same stable `acc`; each token sets `val=t` on every dispatch → expected `n * sum(1..T)`.

---

## 3. Repro

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=/tmp/redline-warpfront-target

hipcc --genco --offload-arch=gfx1150 \
  /home/antmi/redline/bench/acc_kernel.hip \
  -o docs/experiments/redline-kernel-launch/logs/acc_kernel-gfx1150.co

cp docs/experiments/redline-kernel-launch/harness/p3_kernarg_patch.rs \
  /home/antmi/redline/crates/redline-dispatch/examples/p3_kernarg_patch.rs
cd /home/antmi/redline
cargo build --release -p redline-dispatch --example p3_kernarg_patch

REDLINE_P3_HSACO=$PWD/../lemon-mlx-engine/docs/experiments/redline-kernel-launch/logs/acc_kernel-gfx1150.co \
REDLINE_P3_SYMBOL=acc_k.kd REDLINE_P3_N=2 REDLINE_P3_TOKENS=64 \
  /tmp/redline-warpfront-target/release/examples/p3_kernarg_patch
```

Env: `REDLINE_P3_HSACO` (req), `REDLINE_P3_SYMBOL` (default `acc_k.kd`), `REDLINE_P3_N` (≥2), `REDLINE_P3_TOKENS`, `REDLINE_P3_WARMUP`, `REDLINE_P3_ITERS`, `REDLINE_P3_POLICY` (default BoundarySerialized; Independent remapped).

---

## 4. Results (gfx1150)

```text
[p3] device=gfx1150
[p3] Executable::load OK (8136 bytes)
[p3] stable_acc_addr=0x... (baked once; not realloc per token)
[p3] SingleQueueBatchGraph create OK (n=2)
[p3] correctness observed=4160 expected=4160 (n*sum(1..T))
[p3] correctness PASS
[p3] host_median_us_patch_plus_replay=8.796 (NOT gen t/s)
P3_OK patch+replay symbol=acc_k.kd n=2 tokens=64 host_median_us=8.796 correctness=PASS
```

| Metric | Value | Interpretation |
|--------|-------|----------------|
| correctness | **PASS** | `observed == 2 * sum(1..64) = 4160` |
| host_median_us | **8.796** | patch all n + one `replay_and_wait`; **not** gen t/s |
| n / T | 2 / 64 | min batch shape · token loop |

---

## 5. Gates

| Gate | Result |
|------|--------|
| Load CO + resolve symbol | **PASS** |
| Stable ptr baked once | **PASS** (logged addr) |
| In-place `patch_kernarg_u32` per token | **PASS** |
| Correctness vs expected sum | **PASS** |
| Host µs logged without TPS label | **PASS** |
| Product wire / default ON | **None** |
| Gen t/s claim | **None** |

**Kill would have been:** correctness mismatch exit 2, or inability to patch/replay retained batch on gfx1150.

---

## 6. What this does **not** prove

- Product `TokenIterator` uses Redline for any MLX op  
- `graph_decode_input` / `graph_decode_pos` device ptrs bound in-process  
- qmm or lm_head offload  
- Multi-seed variance (single process run; host µs similar to P1 floor ~8.5 µs)  
- Model gen t/s improvement  

---

## 7. Board impact

| Item | Status |
|------|--------|
| P3 design doc | PASS (prior) |
| P3 measured micro-op | **PASS** (this file) |
| Stop-A (measured clause) | **met** with P0+P1+P2b `gpu_new=ok` |
