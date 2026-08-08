# P12d — Ordering without double-wait (query-pre + auto post)

**Status:** **CODE IN** · product flags still **OFF** · same-queue HIP attach still **blocked** (no C-API)  
**Date:** 2026-08-08  
**Depends on:** P12 · P12b · P12c (pre dominates residual tax; post was noise)

---

## 1. First principles (Clear Thought)

| Fact | Implication |
|------|-------------|
| Product HIP writes `x` on stream S | Redline must not read until those waves retire |
| Redline PM4 is **not** bound to S | No free same-stream consumer edge |
| `rl_pm4_replay` = **submit + wait** (redline-capi) | Redline waves are done when replay returns |
| Host `hipDeviceSynchronize` after replay | **Double-wait** (P12c profile: post ~28 µs) |
| Host `hipStreamSynchronize` before every own | **Pre tax** (~1.8 ms / 31 owns) — often waiting real matmul |

**Slogan still:** replace product launches, don’t pile flags. P12d does **not** enable default ON.

---

## 2. API inventory (what we do **not** have)

From `libredline_dispatch.so` / `redline_dispatch.h`:

| Symbol | Role for P12d |
|--------|----------------|
| `rl_pm4_replay` | Host waits Redline completion ✅ |
| `rl_pm4_wait_idle` | **Builder** same-agent RMW fence in IB — **not** HIP bridge |
| `rl_pm4_wait_rmw` | Same, consumer-aware builder fence |
| HIP stream attach / event export | **Missing** → true same-queue blocked |

---

## 3. What P12d implements

### 3.1 Post-replay (`MLX_REDLINE_POST_SYNC`)

| Value | Behavior |
|-------|----------|
| **`auto` (default)** | **No** extra host fence — trust `rl_pm4_replay` wait |
| `off` | Same as auto |
| `device` | `hipDeviceSynchronize` (legacy / paranoid) |
| `stream` | `hipStreamSynchronize(product)` if non-null |

### 3.2 Pre-replay (`MLX_REDLINE_PRE_SYNC`)

| Value | Behavior |
|-------|----------|
| **`stream` (default)** | **P12d:** `hipStreamQuery`; if success → skip; else `hipStreamSynchronize` |
| `force` | Always `hipStreamSynchronize` (P12c historical) |
| `device` | Always `hipDeviceSynchronize` |
| `off` | No pre fence (tax isolation; may race) |

### 3.3 Unchanged (P12c)

- `pack + set_k ×3` **before** pre-sync (pointer patch only).
- `MLX_REDLINE_RMS_PROFILE=1` host timers + **`pre_query_skip` / `pre_wait`** counts.

### 3.4 Not claimed

- Gen t/s ≥2% win  
- Product default ON  
- Same-queue / completion-event bridge into HIP (needs Redline+HIP work → future)

---

## 4. Correctness

| Edge | Policy |
|------|--------|
| Producers still running | Query → NotReady → full stream sync (same as before) |
| Producers already done | Query success → skip sync (free) |
| Redline completion | Replay waited; auto post does not re-fence |
| Paranoid HIP visibility | `POST_SYNC=device` available |

---

## 5. Measure protocol

```bash
export MLX_REDLINE_DECODE=1
export MLX_REDLINE_LIB=.../libredline_dispatch.so
export MLX_REDLINE_OWN_RMSNORM=1
export MLX_REDLINE_RMS_HSACO=.../rms_norm_kernels-gfx1150.co
export MLX_REDLINE_RMS_PROFILE=1
# defaults: PRE_SYNC=stream (query), POST_SYNC=auto
./build/chat "$SNAP08" --max-tokens 64 --temperature 0 --raw

# Legacy pre (always sync)
export MLX_REDLINE_PRE_SYNC=force
```

Expect log:

```text
P12d query-pre + set_k-before-pre; PRE_SYNC=stream POST_SYNC=auto
host profile ... pre_query_skip=N pre_wait=M
```

---

## 6. Measured (TS 20260808-131405)

### Host profile (n=31, P12d defaults, NOT gen t/s)

| Phase | µs |
|-------|---:|
| set_k | 5.2 |
| **pre_sync** | **2548** |
| replay | 346 |
| post_sync | **3.0** (auto) |
| pre_query_skip | **0** |
| pre_wait | **31** |

**Interpretation:** stream is **never** idle at pre (Query never skips) — pre tax is waiting real HIP producers. Post auto works (≈ free).

### Gen t/s 0.8B ×3 interleaved

| Stack | Mean | vs B0 |
|-------|-----:|------:|
| **B0** | **114.0** | — |
| **B1** P12d default | **111.1** | **−2.6%** |
| **B1 PRE=force** | **112.1** | **−1.7%** (noise vs default) |

No ≥2% win. Default ON still forbidden. Logs: `logs/p12d-ab-*-20260808-131405.*`

---

## 7. Next (honest)

1. ~~Measure~~ **DONE**.  
2. Residual pre_wait=31/token = **dual-queue host bubble** → needs HIP-stream-bound Redline submit (upstream API) or fused ownership.  
3. Keep product defaults OFF.
