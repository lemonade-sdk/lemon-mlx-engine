# P12b — Mid-eval sync tax (OWN_RMSNORM / OWN_GLUE)

**Status:** CODE IN (opt-in fence policy) · default fence still **device** · product flags still **OFF**  
**Date:** 2026-08-08  
**Depends on:** P12 OWN_RMSNORM · P10 OWN_GLUE · measured B1 gen ~−2–5%

---

## 1. Problem (measured)

| Mode | Gen effect |
|------|------------|
| B0 baseline | — |
| B1 OWN_RMSNORM only | **~−2% to −5%** (stable across clean/retry2) |
| B2 all flags | **~−8% to −15%** (additive + sync) |

Root cause is **not** closing retained IBs (they stay process-lifetime).  
Root cause is **host fences after every owned launch**:

- Pre: `hipStreamSynchronize(product_stream)` so HIP producers finish before Redline reads.
- Post: `hipDeviceSynchronize()` so HIP consumers see Redline writes.

Packed RMSNorm fires **~31× per L=1** (inventory). Each pays pre+post → multi-% gen tax.

---

## 2. Dual-queue truth (first principles)

| Path | Queue |
|------|--------|
| Product MLX HIP kernels | HIP stream |
| Redline retained PM4 | Redline/AQL queue (not the product stream) |

Therefore:

| Post fence | Waits Redline? | Correct default? |
|------------|----------------|------------------|
| `hipDeviceSynchronize` | **Yes** (whole device) | **Yes** (safe) |
| `hipStreamSynchronize(product)` only | **No** (unless Redline completion is on that stream) | Research only — may race |
| none | **No** | Research only — tax isolation A/B |

Retained IB objects stay open either way (P12 lifecycle). Fence ≠ close.

---

## 3. Code control (P12b)

Env **`MLX_REDLINE_POST_SYNC`**:

| Value | Behavior |
|-------|----------|
| unset / `device` | **Default** — `hipDeviceSynchronize` after OWN_GLUE / OWN_RMSNORM replay |
| `stream` | Post `hipStreamSynchronize` when `hip_stream` non-null; else device. **May race** |
| `off` | No post fence. **May race** — only for measuring sync tax vs B0/B1 |

Pre-sync on OWN_RMSNORM (product stream drain) is **unchanged** (producer deps).

Logs once:

```text
[redline] OWN_RMSNORM ... POST_SYNC=device|stream|off; NOT gen t/s
[redline] OWN_GLUE ... POST_SYNC=...; NOT gen t/s
```

---

## 4. How to measure tax isolation

```bash
# B0
unset MLX_REDLINE_*
./build/chat "$SNAP08" --max-tokens 64 --temperature 0 --raw

# B1 device (current product research path)
export MLX_REDLINE_DECODE=1
export MLX_REDLINE_LIB=.../libredline_dispatch.so
export MLX_REDLINE_OWN_RMSNORM=1
export MLX_REDLINE_RMS_HSACO=.../rms_norm_kernels-gfx1150.co
export MLX_REDLINE_POST_SYNC=device   # or unset
./build/chat ...

# B1-off (tax isolation — not a ship candidate)
export MLX_REDLINE_POST_SYNC=off
./build/chat ...
```

**Interpretation:**

- If B1-off ≈ B0 and B1-device is slow → post-sync **is** the tax (P12b hypothesis confirmed).  
- If B1-off still slow → pre-sync / replay overhead / other.  
- If B1-off faster but wrong outputs → races; need completion events / same-queue (P12c).

Never claim B1-off as product win without correctness proof.

---

## 5. What this is not

- Not product default ON  
- Not gen t/s ≥2% claim  
- Not qmm ownership  
- Not “close kernels after done” — objects still retained  
- Not a fix for dual-queue without a real completion bridge (future P12c)

---

## 6. Measured gen A/B (retry — interleaved ×3)

**TS:** 20260808-125232 · 0.8B · 64 tok · rebuild P12b · lemonade still ~96% VRAM  
**Protocol:** interleaved B0 / B1-device / B1-off ×3 + one B1-stream

| Stack | r1 | r2 | r3 | **Mean** | vs B0 |
|-------|---:|---:|---:|---------:|------:|
| **B0** baseline | 113.7 | 114.9 | 115.5 | **114.7** | — |
| **B1 POST_SYNC=device** | 111.1 | 111.6 | 112.5 | **111.7** | **−2.6%** |
| **B1 POST_SYNC=off** | 110.7 | 112.1 | 111.8 | **111.6** | **−2.8%** |
| **B1 POST_SYNC=stream** | 113.1 | — | — | **113.1** | **−1.4%** (n=1) |

Logs: `logs/p12b-ab-*-20260808-125232.*` · meta `logs/p12b-ab-meta-20260808-125232.txt`  
Health: all B1 arms `rms=PASS` + log `POST_SYNC=device|off|stream`.

### Interpretation (critical)

1. **POST_SYNC=off ≈ POST_SYNC=device** (~111.6 vs 111.7) → **post `hipDeviceSynchronize` is NOT the primary gen tax.**  
2. Residual **~−2.6%** vs B0 is still real under OWN_RMSNORM — likely **pre-stream sync × ~31 RMSNorm/token**, retained replay host path, mutex try_lock, or dual-queue overhead itself.  
3. **stream** n=1 looked slightly better (−1.4%) — treat as noise until n≥3; may also race.  
4. **No ≥2% win** → product default stays **OFF**. P12b fence knob is still useful for experiments; **P12c** should attack **pre-sync coalescing / same-queue / fewer host round-trips**, not only post-sync.

---

## 7. Next

1. ~~Rebuild + B0 / B1-device / B1-off~~ **DONE** (above).  
2. **P12c:** coalesce or skip redundant **pre** `hipStreamSynchronize` when producers already complete; profile host time in `redline_try_own_rmsnorm_packed`.  
3. Optional: n≥3 for `POST_SYNC=stream` only if correctness checked.
