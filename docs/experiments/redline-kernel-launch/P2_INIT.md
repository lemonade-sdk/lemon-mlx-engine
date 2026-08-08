# P2 — Engine Redline session init (measured)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (dlopen + `rl_abi_version`; forward still product)  
**Plan:** [`P2_PLAN.md`](P2_PLAN.md)

---

## 1. Deliverables

| ID | Artifact | Status |
|----|----------|--------|
| P2.a | `include/mlx-lm/common/redline_decode_session.h` + `src/common/redline_decode_session.cpp` | **DONE** |
| P2.b | Smoke logs `logs/p2-{off,on,xor}-*.err` | **DONE** |
| P2.c | This doc | **DONE** |
| P2.d | QUALITY_REVIEW_P2 | pending same fire |

---

## 2. Behavior

| Env | Behavior |
|-----|----------|
| unset / not exact `1` | No dlopen, no `[redline]` line |
| `MLX_REDLINE_DECODE=1` | `dlopen` `libredline_dispatch.so` (or `MLX_REDLINE_LIB`); `rl_abi_version`; best-effort `rl_gpu_new(0)` |
| `=1` + `MLX_DECODE_GRAPH_PURE=1` | XOR fail-closed eager; no dlopen session |
| CMake `MLX_LM_WITH_REDLINE` | Still **OFF** (dlopen path; no hard product link) |

**Forward path:** unchanged product eager (`call_fn`). No gen t/s claim.

---

## 3. Smoke evidence (gfx1150)

Model: `mlx-community/Qwen3.5-0.8B-4bit` local snapshot.  
`MLX_SKIP_WARMUP=1`, `--max-tokens 8 --temperature 0 --raw`.

| Case | Result | Log |
|------|--------|-----|
| off | 0× `[redline]` | `logs/p2-off-20260807-215745.err` |
| on | **1×** `session READY ... abi=1 gpu_new=null (...)` | `logs/p2-on-20260807-215745.err` |
| xor | **1×** fail-closed XOR banner | `logs/p2-xor-20260807-215745.err` |

Banner (on):

```text
[redline] session READY (P2 init-only; forward still product; abi=1 gpu_new=null (try early init before MLX load; fallback ok))
```

### Standalone vs in-process `rl_gpu_new`

| Context | `rl_gpu_new(0)` |
|---------|-----------------|
| Tiny C `dlopen` smoke (no MLX) | **non-null** (PASS) |
| `chat` binary (linked MLX/HIP), even pre-`load_llm` | **null** |

**Interpretation:** P2 gate is **dlopen + C-API symbol smoke** (`rl_abi_version`). Full ROCr GPU bind inside the MLX-linked process is a **known residual** for P3 (may need careful HSA/HIP coexistence or out-of-process worker). Product fallback remains correct.

Early probe hook: `examples/chat.cpp` calls `maybe_log_redline_session_status()` after GPU select, before model load (still null on this host when MLX is linked).

---

## 4. Code sites

| File | Role |
|------|------|
| `src/common/redline_decode_session.cpp` | dlopen, state machine, one-shot log |
| `src/common/generate.cpp` | L=1 / next() call `maybe_log_redline_session_status` |
| `examples/chat.cpp` | Early probe when env=1 |
| `CMakeLists.txt` | Add source; `CMAKE_DL_LIBS` on Linux |

---

## 5. Honesty

- **No gen t/s** attributed to Redline.  
- E1/E2 floors remain microbench only.  
- `gpu_new=null` is documented, not hidden.  
- Default binary still free of hard Redline link.

---

## 6. Next (P3)

Document kernarg-patch integration with `graph_decode_*` stable buffers; design one micro-op or encoder shim. Measure only after a real product-owned dispatch path exists. See stop rule A: P0+P1 green + P3 doc + quality PASS.
