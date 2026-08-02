# ROCm install / upgrade notes (Redline E0 host)

**Host baseline (2026-08-02):** TheRock-style `/opt/rocm` with **core-7.13** only.  
**GPU:** gfx1150 (Radeon 890M).

## Do you need 7.14?

| Goal | 7.13 sufficient? | Notes |
|------|------------------|--------|
| **Compile** `redline-dispatch` / `capi` / `hipgraph` | **Yes** (measured E0) | See [`E0_HOST_BUILD.md`](E0_HOST_BUILD.md) |
| **hipcc --genco** floor kernel for gfx1150 | **Yes** (measured E0) | |
| Core retained AQL / PM4 floor microbench | **Likely try first** | E1 not yet run; ROCr `libhsa-runtime64` present under core |
| Optional HIP batch-mem / green-context / library FFI | **No — need symbols from ≥7.14** | Dynamic load hard-fails if missing |
| Match Redline **published** ROCm 7.14 scorecards | Prefer **7.14** | Certification methodology, not our E0 compile |
| Radiowave HIP version **policy** (≥7.14 gate) | Prefer **7.14** when using radiowave toolchain gates | Dispatch stack still built radiowave as dep; policy may fire when *using* radiowave compile APIs |

## Current layout (this machine)

```text
/opt/rocm/
  core -> alternatives -> core-7.13
  core-7.13/          # HIP 7.13.99004, libhsa-runtime64.so.1.21.0
  lib/ -> core libs
```

There is **no** `core-7.14` (or newer) directory present.

## Build environment (required)

```bash
# Prefer TheRock hipcc over conda HIP 6.3
export PATH=/opt/rocm/core/bin:/opt/rocm/core/lib/llvm/bin:$PATH
export ROCM_PATH=/opt/rocm/core
export HIP_PATH=/opt/rocm/core
export LD_LIBRARY_PATH=/opt/rocm/core/lib:${LD_LIBRARY_PATH:-}
```

Verify:

```bash
which hipcc   # must be /opt/rocm/core/bin/hipcc
hipcc --version
cat /opt/rocm/core/.info/version
rocminfo | grep -E 'Name:|Marketing'
```

## Upgrade path (when/if needed)

Redline upstream expects **ROCm Core SDK ≥ 7.14** (TheRock layout, typically `/opt/rocm/core`). Practical options on this host class:

1. **TheRock / ROCm core package upgrade** that co-installs `core-7.14` (or replaces `core` symlink) while retaining gfx1150 support.  
2. Confirm after upgrade:
   - `cat /opt/rocm/core/.info/version` ≥ `7.14`
   - `hipcc --version` shows HIP 7.14.x
   - `rocminfo` still lists **gfx1150**
3. Rebuild Redline with the same `PATH`/`ROCM_PATH` exports into a fresh `CARGO_TARGET_DIR`.
4. Re-run E1 `dispatch_floor` and compare to any prior 7.13 attempt (do not mix scorecards).

**Do not** force-upgrade mid-fire if a 35B load is active or if another experiment owns the GPU.  
**Do not** claim performance wins from an upgrade without new logs on gfx1150.

## Partial-build note (E0)

If only a subset of crates is needed for engine C ABI experiments:

```bash
cargo build --release -p redline-dispatch -p redline-capi
# hipgraph optional (LD_PRELOAD path — low priority for lemon-mlx-engine)
```

Clone: prefer **https://github.com/warpfront/redline** (`/tmp/redline-warpfront`).  
Fork for reading: https://github.com/pwilkin/redline.
