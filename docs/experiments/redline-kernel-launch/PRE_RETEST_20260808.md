# PRE retest (phase1 + phase2 owning)

**TS:** 20260808-141111 · redline `9df1dfe` · both paths **own** (`phase1-used` / `phase2-used`)

## Gen (0.8B, 64 tok, ×2)

| Stack | r1 | r2 | **Mean** | vs B0 |
|-------|---:|---:|---------:|------:|
| B0 | 115.9 | 117.5 | **116.7** | — |
| B1 phase1 | 113.1 | 112.2 | **112.7** | **−3.4%** |
| B1 phase2 | 111.8 | 112.4 | **112.1** | **−3.9%** |

## Host profile n=31 (µs sum; NOT gen t/s)

With bridge, HIP join lives **inside** ordered replay (counted as `replay`), not the thin `pre_sync` timer (~124µs = query/setup only).

| Path | set_k | pre (timer) | **replay (join+RL)** | post | pre+replay |
|------|------:|------------:|---------------------:|-----:|-----------:|
| phase1 mean | ~4 | ~124 | **~2247** | ~3 | **~2371** |
| phase2 mean | ~5 | ~124 | **~2279** | ~3 | **~2403** |

- **~2.2–2.5 ms / 31 owns** ≈ **~72–80 µs per OWN_RMSNORM** host wall for producer join + Redline kernel  
- **post ~3 µs** (auto)  
- **No ≥2% gen win**; phase2 ≈ phase1 (slightly slower)

Logs: `logs/pre-retest-*-20260808-141111.*`
