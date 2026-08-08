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

## 5. Bench status

| Item | Status |
|------|--------|
| Code + rebuild chat | **this slice** |
| Multi-rep B0 / B1-pre-default / B1-PRE_SYNC=off | **PENDING_BENCH** (VRAM ~96% lemonade; GPU use low but mem full) |

When GPU free:

```bash
# B1 with P12c path (defaults)
export MLX_REDLINE_DECODE=1
export MLX_REDLINE_LIB=.../libredline_dispatch.so
export MLX_REDLINE_OWN_RMSNORM=1
export MLX_REDLINE_RMS_HSACO=.../rms_norm_kernels-gfx1150.co
# unset PRE_SYNC → stream; unset POST_SYNC → device
export MLX_REDLINE_RMS_PROFILE=1   # optional host breakdown
./build/chat "$SNAP08" --max-tokens 64 --temperature 0 --raw

# Tax isolation only (may race)
export MLX_REDLINE_PRE_SYNC=off
```

Interpret: if B1-pre-off ≈ B0 while B1-pre-stream still slow → pre-sync **is** primary residual tax (still not shippable without a real completion bridge).

---

## 6. Next

1. **PENDING_BENCH** multi-rep gen A/B when VRAM free.  
2. If pre remains tax: device-side wait / same-queue / events (not host `off`).  
3. Keep product defaults OFF.
