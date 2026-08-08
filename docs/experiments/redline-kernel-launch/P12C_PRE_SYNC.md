# P12c — OWN_RMSNORM pre-sync / host-path tax

**Status:** CODE IN · default flags still **OFF** · POST_SYNC default still **device**  
**Date:** 2026-08-08  
**Depends on:** P12 · P12b (post-sync **not** primary tax; residual ~−2.6%)  
**Decision:** prepare kernargs **before** pre-sync (safe overlap); env-gated pre fence for tax A/B.

---

## 1. Problem (from P12b)

| Finding | Implication |
|---------|-------------|
| `POST_SYNC=off` ≈ `POST_SYNC=device` (~−2.6% both) | Post `hipDeviceSynchronize` is **not** primary gen tax |
| Residual ~−2.6% under B1 OWN_RMSNORM | Pre-stream drain × ~31 / token, set_k+replay host path, dual-queue |

Historical OWN_RMSNORM host order:

```text
pre hipStreamSynchronize(product)   ← wait for producers
pack + set_k ×3                     ← host-only, but sat behind pre
replay Redline
post hipDeviceSynchronize           ← consumers (default)
```

`set_k` only patches **device pointers / scalars** into the retained IB. It does **not** read producer VRAM. Waiting for producers *before* set_k serializes free host work behind a GPU drain.

---

## 2. Safe change (default research path)

**New order (always when OWN_RMSNORM handles a launch):**

```text
pack + set_k ×3                     ← overlap with in-flight product HIP
pre-sync (default: product stream)  ← still required before VRAM read
replay Redline
post-sync (default: device)         ← unchanged P12b
```

| Step | Needs producer data complete? | Safe before pre-sync? |
|------|-------------------------------|------------------------|
| pack / set_k | No (addresses only) | **Yes** |
| replay | **Yes** | No |
| post-sync | N/A (consumer fence) | after replay |

**Not** removed: dual-queue producer drain before replay (correctness).  
**Not** product default ON.

---

## 3. Env controls

| Env | Values | Default | Notes |
|-----|--------|---------|-------|
| `MLX_REDLINE_PRE_SYNC` | `stream` \| `device` \| `off` | **stream** (historical) | `off` may race — tax isolation only |
| `MLX_REDLINE_POST_SYNC` | `device` \| `stream` \| `off` | **device** | P12b; unchanged |
| `MLX_REDLINE_RMS_PROFILE` | `1` | off | Host-phase timers; one log after n≥31 owns |

Log once:

```text
[redline] OWN_RMSNORM ... P12c set_k-before-pre; PRE_SYNC=stream POST_SYNC=device; NOT gen t/s
[redline] OWN_RMSNORM host profile (n=31): set_k=…us pre_sync=…us replay=…us post_sync=…us (host wall; NOT gen t/s)
```

---

## 4. What this is not

- Not a gen t/s ≥2% claim (no multi-rep A/B in this slice — GPU VRAM held by lemonade)
- Not product default ON for `OWN_RMSNORM` / `PRE_SYNC=off` / `POST_SYNC=off`
- Not completion events / same-queue bridge (future)
- Not coalescing set_k chunks (RL_ERR_RECORD oversize limit still forces 24+8+8)

---

## 5. Bench status — DONE (cleared GPU / GTT headroom)

**TS:** 20260808-130444  
**Host:** gfx1150 · GTT **60 GiB** total (~16 GiB used at start) · sys mem ~64 GiB available · GPU use ~2%  
**Note:** lemonade `llama-server` may still be present at low CPU; GTT headroom is the important unlock for large weights.

### Host profile (n=31 owns, NOT gen t/s)

| Phase | Host µs |
|-------|--------:|
| set_k | **5.7** |
| **pre_sync** | **1802** |
| replay | 314 |
| post_sync | **28** |

### 0.8B gen (interleaved ×3)

| Stack | r1 | r2 | r3 | **Mean** | vs B0 |
|-------|---:|---:|---:|---------:|------:|
| **B0** | 115.3 | 116.7 | 115.9 | **116.0** | — |
| **B1** P12c default (PRE=stream) | 111.6 | 113.4 | 112.5 | **112.5** | **−3.0%** |
| **B1 PRE_SYNC=off** (tax iso; may race) | 113.7 | 114.5 | 114.1 | **114.1** | **−1.6%** |

### 35B LemonMLXE

| Stack | gen t/s | vs B0 |
|-------|--------:|------:|
| **B0** | **29.10** | — |
| **B1** P12c default | **28.20** | **−3.1%** |
| B0b | 29.05 | noise OK |

Logs: `logs/p12c-ab-*-20260808-130444.*`

### Interpretation

1. **pre_sync dominates** host OWN_RMSNORM path (~1.8 ms / 31 owns); post is noise (~28 µs).  
2. **`PRE_SYNC=off` recovers ~half** of the 0.8B gen gap (−3.0% → −1.6%) but is **not** a product path (races) and still **no ≥2% win**.  
3. Residual after pre-off (~−1.6%) ≈ replay/set_k/mutex dual-queue overhead.  
4. 35B same shape (−3.1%). **Default ON still forbidden.**

---

## 6. Next

1. ~~Multi-rep A/B~~ **DONE**.  
2. Real fix: **completion events / same-queue** so we do not host-drain every RMSNorm (P12d).  
3. Keep product defaults OFF.  
4. Do not ship `PRE_SYNC=off`.
