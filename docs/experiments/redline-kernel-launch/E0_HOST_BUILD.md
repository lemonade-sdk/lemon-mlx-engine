# E0 — Host build (warpfront Redline × local ROCm)

**Date:** 2026-08-02  
**Branch:** `exp/redline-kernel-launch`  
**Host:** antmi-AI-Series · gfx**1150** (AMD Radeon 890M) · Ryzen AI 9 HX 370  
**Verdict:** **BUILD_OK** on ROCm Core **7.13.0** (compile + floor HSACO for gfx1150).  
**Not claimed:** µs/dispatch, gen t/s, on-device AQL replay success, product wire.

---

## Host toolchain snapshot

| Item | Value |
|------|--------|
| `/opt/rocm/core` layout | TheRock-style; `core` → `core-7.13` |
| Core version file | `/opt/rocm/core/.info/version` → **7.13.0** |
| System `hipcc` | `/opt/rocm/core/bin/hipcc` → **HIP 7.13.99004-3309c611** · AMD clang 23.0.0git |
| Default shell `hipcc` (PATH) | Often **conda** `/home/antmi/miniforge3/bin/hipcc` → HIP **6.3** — **do not use** for Redline |
| GPU | `rocminfo`: **gfx1150** · Marketing **AMD Radeon 890M Graphics** |
| Rust | rustc/cargo **1.96.0** |
| Redline source | `/tmp/redline-warpfront` · git **b505a72** (`ci: promote rustfmt and clippy…`) |
| Cargo target dir | `/tmp/redline-warpfront-target` (out of tree) |

---

## Build command (repro)

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR=/tmp/redline-warpfront-target
cd /tmp/redline-warpfront
cargo build --release -p redline-dispatch -p redline-capi -p redline-hipgraph
```

### Result

| Field | Value |
|-------|--------|
| Status | **SUCCESS** |
| Wall | **8.45s** (cold crates.io download + compile) |
| Exit | **0** |
| Log | [`logs/e0-build-warpfront-20260802-142519.log`](logs/e0-build-warpfront-20260802-142519.log) |

Artifacts (not in git):

- `/tmp/redline-warpfront-target/release/libredline_dispatch.so`
- `/tmp/redline-warpfront-target/release/libredline_dispatch.a`
- `/tmp/redline-warpfront-target/release/libredline_hipgraph.so`
- examples: `aql_arch_smoke`, `dispatch_floor` (built)

`ldd` on `libredline_dispatch.so`: **no** hard link to `libhsa-runtime64` (runtime **dlopen** via `libloading` / `redline-rocr`) — expected.

---

## 7.13 vs ≥7.14 (blocker revision)

| Claim (pre-E0 research) | Post-E0 fact |
|-------------------------|--------------|
| Redline README: **Requires ROCm Core SDK ≥ 7.14** | **Not a hard compile gate** for `-p redline-dispatch -p redline-capi -p redline-hipgraph` on this host |
| Radiowave / product cert tables | Official scorecards remain ROCm **7.14** |
| Optional HIP 7.14-only FFI in dispatch | Dynamic resolve; **hard-fail only when those features are used** (batch mem, green/execution context, some library load paths) — see `crates/redline-dispatch/src/ffi_batch_mem.rs`, `ffi_execution_ctx.rs`, `ffi_library.rs` |

**Conclusion for this experiment:** E0 is **not blocked** by 7.13 for building the dispatch stack. Upgrade to 7.14 remains **recommended** for (a) matching upstream product certification, (b) optional 7.14 HIP surfaces, (c) radiowave “floor” toolchain policy when using radiowave compile gates. See [`INSTALL_UPGRADE.md`](INSTALL_UPGRADE.md).

---

## Floor HSACO (E1 prep, still E0 scope)

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
hipcc --genco --offload-arch=gfx1150 \
  /tmp/redline-warpfront/bench/floor_kernel.hip \
  -o /tmp/redline-warpfront-hsaco/floor_kernel-gfx1150.co
```

| Field | Value |
|-------|--------|
| Status | **SUCCESS** EXIT 0 |
| Size | 8144 bytes |
| Magic | `__CLANG_OFFLOAD_BUNDLE__` |
| Evidence copy | [`logs/floor_kernel-gfx1150.co`](logs/floor_kernel-gfx1150.co) |
| Compile log | [`logs/e0-hsaco-compile-gfx1150-20260802-142608.log`](logs/e0-hsaco-compile-gfx1150-20260802-142608.log) |
| Symbol expected by bench | `floor_k.kd` (per `bench/floor_kernel.hip` comments) |

**Not done this fire:** load CO + `dispatch_floor` replay (that is **E1**).

Examples without `REDLINE_FLOOR_HSACO` / `REDLINE_AQL_HSACO` exit with the expected env error (log: `logs/e0-smoke-20260802-142549.log`).

---

## Pitfalls

1. **PATH:** Conda HIP 6.3 shadows TheRock 7.13 — always prefix `/opt/rocm/core/bin`.  
2. **Do not** treat gfx1151 / gfx1201 published ratios as gfx1150 results.  
3. **Do not** enable product decode HIP graphs; Redline retained-PM4 is a separate lever.

---

## Next (E1)

```bash
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:$LD_LIBRARY_PATH
export CARGO_TARGET_DIR=/tmp/redline-warpfront-target
export REDLINE_FLOOR_HSACO=/tmp/redline-warpfront-hsaco/floor_kernel-gfx1150.co
# optional: REDLINE_FLOOR_N=64 REDLINE_FLOOR_M=200 REDLINE_FLOOR_WARMUP=20
/tmp/redline-warpfront-target/release/examples/dispatch_floor
```

Pass bar for E1: log µs/dispatch for `SystemEveryDispatch` vs `BoundarySerialized` on **this** GPU, or explicit fail log.
