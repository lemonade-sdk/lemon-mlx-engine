# Quality review — P2b engine session init

**Date:** 2026-08-08  
**Scope:** dlopen Redline C-API session + chat smoke  
**Verdict:** **PASS** (with residual)

| Check | Result |
|-------|--------|
| Default OFF | **PASS** — 0× `[redline]` when env unset |
| Exact `"1"` only | **PASS** — session path gated |
| One-shot banner | **PASS** — count=1 on on-log |
| No forward change | **PASS** — still product eager |
| XOR pure-graph | **PASS** — fail-closed |
| No HIP-graph enable | **PASS** |
| CMake default OFF | **PASS** — dlopen, not hard link |
| Build green | **PASS** — `chat` exit 0 |
| No gen t/s claim | **PASS** |
| Honest residual | **PASS** — root-caused RPATH/conda HSA; fixed with RUNPATH + core first |
| Naming | **PASS** — P2b (session) ≠ P2 N-sweep |
| Post-fix gpu_new | **PASS** — `logs/p2b-rpathfix-20260807-220731.err` shows `gpu_new=ok` |

## Residual (updated)

- Upstream Redline `load_symbols` stops at first successful `dlopen` of soname (does not fall through to absolute `/opt/rocm/core` if conda opens first). Prefer capable RPATH/RUNPATH on engine binaries.  
- P3 still must measure before any gen t/s claim.

**Reviewer notes:** ship P2b with RPATH fix; do not product-default-on.
