# P13 — Redline ↔ HIP stream bridge: work required for the PR(s)

**Status:** DESIGN / PR-READY SCOPE — **not implemented**  
**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Depends on:** P12–P12d OWN_RMSNORM ownership + measured PRE dual-queue tax  
**Related:** [`P12D_ORDERING.md`](P12D_ORDERING.md) · [`P12C_PRE_SYNC.md`](P12C_PRE_SYNC.md) · [`E3_HSACO.md`](E3_HSACO.md) · [`ROADMAP.md`](ROADMAP.md)

---

## 0. One-sentence goal

**Order Redline retained PM4 after product HIP producers (and before product HIP consumers) without host `hipStreamSynchronize` on every owned launch**, so OWN_RMSNORM can stop paying the ~2.5 ms/token PRE tax and we can re-measure gen t/s honestly.

---

## 1. Problem statement (measured)

### 1.1 What we own today

| Path | Flag | Product replace? | Gen t/s |
|------|------|------------------|---------|
| Glue pos_set / inc / scalar_copy | `OWN_GLUE=1` | **Yes** | ≈ baseline |
| Packed RMSNorm | `OWN_RMSNORM=1` | **Yes** (~31/37) | **~−2.5–3%** |
| Sidecar / small_op | `SIDECAR` / `SMALL_OP` | **No** (additive) | slower |

### 1.2 Why OWN_RMSNORM loses gen

Host profile (n=31 owns, P12d, **not** gen t/s):

| Phase | Host µs | Notes |
|-------|--------:|-------|
| set_k | ~5 | free |
| **PRE (`hipStreamSynchronize`)** | **~1800–2500** | stream always busy (`pre_wait=31`, `skip=0`) |
| replay | ~300–350 | `rl_pm4_replay` = submit+**wait** Redline |
| POST | ~3 | default `auto` trusts replay wait |

**PRE is long because the host joins the product HIP stream** while real producers (matmul / attn / …) finish. Product HIP RMSNorm is **same-stream** after those kernels → GPU orders, host does not join. Redline is **another queue** → without a GPU-side dependency we must host-join or race (`PRE=off`).

### 1.3 What is *not* the problem

- IB rebuild (retained set_k+replay)  
- Closing kernels after launch (IBs process-lifetime)  
- POST device fence (eliminated as primary tax in P12b/P12d)  
- Missing GTT (60 GiB GTT; measure still shows PRE tax)

---

## 2. Non-goals (explicit)

| Non-goal | Why |
|----------|-----|
| Product default ON without ≥2% B1 gen win | Hard ban |
| Claiming microbench µs as gen t/s | Hard ban |
| Drop-in qmm HSACO in this PR | E3: not drop-in |
| HIP-graph product re-enable | Separate; XOR with Redline |
| `PRE_SYNC=off` as ship path | Races |
| “All flags on” as speed stack | Measured ~−9% |

---

## 3. Repos and PR split

Work spans **at least two** codebases. Do **not** open a lone HIP PR without a Redline consumer.

| # | Repo | PR title (draft) | Owner skill |
|---|------|------------------|-------------|
| **PR-A** | **redline** — **[`antmikinka/redline`](https://github.com/antmikinka/redline)** branch `exp/hip-stream-bridge` (base: pwilkin; local `/home/antmi/redline`) | `capi: HIP stream wait + replay_after_hip_stream` (phase1 host join; phase2 device wait) | Redline/HSA/PM4 |
| **PR-B** | **lemon-mlx-engine** (`exp/redline-kernel-launch`) | `redline: use stream bridge; drop host PRE when armed` | MLX ROCm wire |
| **PR-C** (conditional) | **ROCm/HIP** or HSA runtime | Only if PR-A needs a missing primitive (stream→signal export, external wait) | ROCm |

**Order:** PR-A design + prototype → PR-B wire → measure → PR-C only if blocked on missing OS/runtime API.

---

## 4. API design options (PR-A must pick one primary)

### Option S — **Wait on HIP stream before replay** (minimum viable)

**Semantics:** Before reading producer VRAM, Redline waits until all work previously submitted on `hipStream_t S` is complete — **without** blocking the host CPU for the whole interval (GPU/HSA wait), *or* with a single cheaper wait if full async is harder v1.

```c
// Draft C ABI (names illustrative)
int32_t rl_pm4_replay_after_hip_stream(void* ib, void* hip_stream);
// or:
int32_t rl_gpu_wait_hip_stream(void* gpu, void* hip_stream);
int32_t rl_pm4_replay(void* ib); // existing submit+wait Redline
```

| Pros | Cons |
|------|------|
| lemon-mlx already has `encoder.stream()` | Still serializes Redline after HIP (no true overlap of RMSNorm with later HIP unless async) |
| Matches today’s correctness model | Host may still block if API is “wait on host” — **must** be device-side wait |
| Smallest conceptual change | Need reliable HIP stream completion signal |

**Success bar for S:** lemon-mlx can set `PRE_SYNC` path to **no host Synchronize**; gen B1 ≥ baseline or clearly less negative; correctness vs product text.

### Option E — **hipEvent / HSA signal bridge**

```c
// Product (or MLX helper) records event on stream after producers.
// Redline waits on that event/signal before replay.
int32_t rl_pm4_replay_after_hip_event(void* ib, void* hip_event);
// After replay, record Redline completion into an event HIP can wait on:
int32_t rl_pm4_replay_signal_hip_event(void* ib, void* hip_event_out);
```

| Pros | Cons |
|------|------|
| Standard HIP dependency style | Extra record sites if not automatic |
| Bidirectional (HIP→Redline and Redline→HIP) | Event lifetime / pooling |

**Success bar for E:** no host PRE; product HIP consumers can `hipStreamWaitEvent` without `hipDeviceSynchronize`.

### Option Q — **Same-queue submit** (best long-term)

Enqueue Redline PM4/AQL onto the **same** underlying compute queue as the product `hipStream_t` (or a stream that HIP treats as ordered with S).

| Pros | Cons |
|------|------|
| True product-like dependency | Highest invasiveness (HIP/HSA internals) |
| Host records freely | May require PR-C |

**Success bar for Q:** OWN_RMSNORM gen ≈ or better than HIP RMSNorm; no PRE host join.

### Recommendation for first PR series

1. **Ship Option S or E as MVP** (device-side wait, not host Synchronize).  
2. **Design Option Q** as follow-on if MVP gen ceiling &lt; 2%.  
3. Document which ROCm version / gfx1150 assumptions.

---

## 5. Detailed work checklist

### 5.1 PR-A — Redline C-API / dispatch (required)

| ID | Task | Done when |
|----|------|-----------|
| A1 | Spec: choose S and/or E; document memory visibility (L2, caches) on gfx11/APU | Written + reviewed |
| A2 | Map `hipStream_t` → waitable object (HSA signal / queue barrier / HIP driver API) on ROCm 7.13+ | Spike note + proof on gfx1150 |
| A3 | Implement `rl_pm4_replay_after_hip_stream` **or** `rl_gpu_wait_hip_stream` + existing replay | Symbol in `libredline_dispatch.so` |
| A4 | Optional: async replay + completion signal for Redline→HIP | API + test |
| A5 | Optional: `rl_pm4_replay_after_hip_event` | API + test |
| A6 | Correctness tests: HIP kernel writes buffer → Redline reads → host checks values | PASS |
| A7 | Negative tests: null stream, destroyed stream, wrong device | Fail-closed codes |
| A8 | ABI version bump / feature bit if needed (`rl_abi_version` or query) | Documented |
| A9 | Header + README: “host Synchronize not required when using *after_stream*” | Docs |
| A10 | Host wall microbench: N× (HIP no-op or light kernel → Redline replay) host join vs stream-wait | Numbers logged **not** as gen t/s |

**Out of scope for PR-A:** MLX model integration, product default ON, qmm.

### 5.2 PR-C — ROCm/HIP (only if A2 fails)

| ID | Task | Done when |
|----|------|-----------|
| C1 | Identify missing primitive (e.g. export stream completion as HSA signal) | Bug/feature filed |
| C2 | Minimal HIP/HSA patch or public API usage doc | Merged or vendor ACK |
| C3 | gfx1150 validation on lemonade host ROCm 7.13 | PASS |

**Do not start C1** until A2 spike proves the gap.

### 5.3 PR-B — lemon-mlx-engine (this branch)

| ID | Task | Done when |
|----|------|-----------|
| B1 | Detect bridge symbols (`dlsym`); capability bit in session READY string | Log once |
| B2 | `OWN_RMSNORM` hot path: if bridge OK → **skip** host `redline_pre_sync` / use `replay_after_stream` | Code path |
| B3 | Keep `PRE_SYNC=force|stream|off` for fallback / A/B | Env still works |
| B4 | POST: keep `auto` (replay wait); if Redline→HIP event lands, wire consumers only if needed | No double-wait |
| B5 | OWN_GLUE: optional same bridge if glue ever needs producer wait (today often cheaper) | Optional |
| B6 | Rebuild `chat`; smoke READY + OWN_RMSNORM log `bridge=stream` | PASS |
| B7 | Gen A/B 0.8B ×3: B0 vs B1 (bridge) vs B1 (host PRE force) | Doc `GEN_AB_P13_*.md` |
| B8 | Gen A/B 35B B0/B1 once GPU free | Doc |
| B9 | If B1 ≥ **+2%** vs B0 and quality OK → discuss default ON **only for measured path** | Gate |
| B10 | If B1 still &lt; +2% → document ceiling; no default ON | Honest |

**Default product flags remain OFF** until B9.

### 5.4 Validation matrix (mandatory before claiming win)

| Test | Pass criteria |
|------|----------------|
| Numeric smoke | f32 RMSNorm vs product reference (existing multi smoke + gen text) |
| XOR | `DECODE=1` + `MLX_DECODE_GRAPH_PURE=1` still fail-closed |
| Fallback | Missing bridge symbol → old PRE path, no crash |
| Gen B0/B1 | Same protocol as P12c/P12d (64 tok, fixed prompt, interleaved) |
| Profile | `pre_wait` drops or pre_us collapses when bridge on; `RMS_PROFILE=1` |
| All-flags | Still not required for speed claim |

---

## 6. lemon-mlx code touch list (PR-B, after PR-A)

| File | Change |
|------|--------|
| `src/common/redline_decode_session.cpp` | dlsym bridge; OWN_RMSNORM (and glue if needed) call site; READY banner |
| `include/mlx-lm/common/redline_decode_session.h` | Document new env/capability |
| `docs/experiments/redline-kernel-launch/P13_*.md` | Results |
| `docs/experiments/redline-kernel-launch/ROADMAP.md` | Close P13 when measured |
| Optional mlx-rocm patch | Only if event record must sit inside RMSNorm producer path (prefer stream wait at own site) |

**No change** to product default CMake flags.

---

## 7. Success / kill criteria

| Outcome | Action |
|---------|--------|
| Bridge works; B1 gen ≥ **+2%** vs B0; quality OK | Consider product default ON for **that** ownership path only |
| Bridge works; B1 ≈ baseline (±1%) | Keep research OFF; still valuable (ownership without tax) |
| Bridge works; B1 still ≤ −2% | Tax elsewhere (replay wait, mutex, residual); do not default ON |
| A2 impossible on gfx1150 ROCm 7.13 without multi-month HIP work | **KILL gen pursuit** for OWN_RMSNORM; keep as correctness ownership; focus residual **or** pause Redline gen |
| Only host-level “wait” API (still blocks CPU same as Synchronize) | **Not a success** — reject as P13 complete |

---

## 8. Effort sketch (planning only)

| Phase | Rough scope |
|-------|-------------|
| Spike A2 (1–3 days) | Prove stream→waitable on gfx1150 |
| PR-A MVP (1–2 weeks) | S or E + tests + .so |
| PR-B wire + A/B (2–4 days) | lemon-mlx + docs |
| PR-C | 0 or large — only if spike fails |

---

## 9. Suggested PR description template (Redline PR-A)

```text
Title: capi: order retained PM4 after HIP stream (gfx11)

## Summary
Add a C-API entry point so callers can wait for a HIP stream's prior work
before rl_pm4_replay, without requiring the host to hipStreamSynchronize.

## Motivation
lemon-mlx-engine OWN_RMSNORM replaces packed RMSNorm HIP launches but must
host-join the product stream today (~2ms/token PRE tax). Product HIP chains
dependencies on-stream; Redline is a separate queue.

## API
<exact signatures>

## Test plan
- [ ] unit: HIP fill → Redline read
- [ ] gfx1150 smoke
- [ ] null/fail-closed

## Non-goals
Product MLX default ON; qmm ownership; gen t/s claims in this repo alone.
```

---

## 10. Suggested PR description template (lemon-mlx PR-B)

```text
Title: redline: use HIP stream bridge for OWN_RMSNORM; drop host PRE when available

## Summary
If libredline_dispatch exports stream-ordered replay, OWN_RMSNORM skips
host hipStreamSynchronize and uses the bridge. Fallback to P12d PRE path.

## Measure
B0 vs B1 gen A/B 0.8B (and 35B). No default ON without ≥2% win.

## Related
docs/experiments/redline-kernel-launch/P13_STREAM_BRIDGE_PR.md
```

---

## 11. Are we “done” with Redline including glue + norm? (status board)

See companion section in chat / summary below — mirrored here:

| Area | Code ownership | Gen t/s | Product default |
|------|----------------|---------|-----------------|
| Session / dlopen / RUNPATH | **DONE** | n/a | OFF |
| OWN_GLUE retained | **DONE** | ≈ baseline | OFF (no ≥2% need for glue alone) |
| OWN_RMSNORM packed | **DONE** | **LOSS ~−3%** (PRE tax) | **OFF** |
| PRE/POST knobs P12b–d | **DONE** | post fixed; pre structural | n/a |
| Stream bridge P13 | **NOT DONE** | required for norm gen | n/a |
| Strided RMSNorm / CustomKernel / qmm | **NOT DONE** | unknown | OFF |
| Sidecar / small_op | DONE as lab | slower | OFF forever as speed |
| All-flags B2 | measured | **−7–10%** | never |

**Conclusion:** Research **ownership** of glue + packed RMSNorm is **implemented and measured**.  
**Product “Redline makes decode faster” is not done** until P13 (or equivalent) lands and passes the gen gate.

---

## 12. Immediate next actions (human / loop)

1. File or open **PR-A spike** in redline: stream→waitable on gfx1150.  
2. Keep lemon-mlx **defaults OFF**; do not pile flags.  
3. After spike: implement PR-A MVP → PR-B → gen A/B.  
4. If spike fails: write KILL note; freeze OWN_RMSNORM as correctness-only.

---

## 13. References

- P12d profile / gen: [`P12D_ORDERING.md`](P12D_ORDERING.md)  
- P12c PRE tax: [`P12C_PRE_SYNC.md`](P12C_PRE_SYNC.md)  
- All-flags P12c: [`GEN_AB_ALLFLAGS_P12C_20260808.md`](GEN_AB_ALLFLAGS_P12C_20260808.md)  
- redline-capi: `rl_pm4_replay` = submit+wait (`redline_dispatch.h`)  
- MLX hook: `mlx_redline_try_own_rmsnorm(..., hip_stream)` from product `encoder.stream()`
