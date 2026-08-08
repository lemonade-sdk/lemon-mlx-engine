# QUALITY_REVIEW — P8 engine-owned small op (graph_decode VRAM)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Slice:** P8 `MLX_REDLINE_SMALL_OP` product-buffer L=1 retained PM4  
**Supervisor verdict:** **PASS**

---

## Quintuple roles (simulated — spawn types not used this fire)

### 1) Explore / facts — **PASS**

| Fact | Evidence |
|------|----------|
| Prior board | MASTER: E0–E4 DONE; P0–P7b PASS; gen A/B NOT RUN; default ON forbidden |
| Gap | P7b synthetic n-sum; no live product `graph_decode_input` consume on L=1 |
| HSACO / lib | `logs/acc_kernel-gfx1150.co`; `/tmp/redline-warpfront-target/release/libredline_dispatch.so` |
| Model | local Qwen3.5-0.8B-4bit snapshot (offline) |
| Smoke | `logs/p8-{off,on-smallop,xor}-20260808-114957.err` |

### 2) Plan / strategy — **PASS**

| Decision | Choice |
|----------|--------|
| Primary | B — engine-owned small op on graph_decode VRAM (default OFF) |
| Rejected | A re-verify (closed); D gen A/B (no product replace yet); E empty |
| Clear Thought | sequentialthinking; decisionframework; metacognitivemonitoring; scientificmethod H-p8-small-op; mentalmodel first_principles |
| Bans | no LoopBrake; no fake TPS; no call_fn replace; no default ON; XOR fail-closed |

### 3) Senior-developer implement — **PASS**

| Change | File |
|--------|------|
| API | `include/mlx-lm/common/redline_decode_session.h` — `maybe_redline_small_op_l1` |
| Arm + tick + verify | `src/common/redline_decode_session.cpp` |
| L=1 wire | `src/common/generate.cpp` — `step` + dtor verify path |
| Build | `cmake --build build --target chat` exit 0 |
| Behavior | SMALL_OP arms IB; L=1 writes `graph_decode_input`, D2H VRAM, patch+replay; SIDECAR synthetic skipped when SMALL_OP; no `graph_external_pos` |

### 4) Quality-reviewer — **PASS**

| Check | Result |
|-------|--------|
| off → 0× `[redline]` | **PASS** |
| on → `small_op_armed=1` + L1 tick | **PASS** |
| fullgen `side_obs==side_exp` | **PASS** n=17 **15185/15185** |
| XOR fail-closed | **PASS** |
| Host µs / Generation t/s labeled product only | **OK** (no Redline A/B claim) |
| call_fn replace | **absent** |

### 5) Second supervisor quality-reviewer — **PASS**

| Ban / evidence gate | Status |
|---------------------|--------|
| No product default ON | **OK** — exact `"1"` env only |
| No invent gen t/s | **OK** — banners say NOT gen t/s |
| XOR fail-closed | **OK** — logged |
| call_fn still product | **OK** — tick before `call_fn`; no replace |
| Net-new slice (not empty doc thrash) | **OK** — code + measured fullgen |
| Honest next step | gen A/B only after measured product-path ownership |

---

## Supervisor summary

**PASS** — ship P8 docs + MASTER board update. Residual: gen t/s A/B still **NOT RUN**; product default ON still **FORBIDDEN**.
