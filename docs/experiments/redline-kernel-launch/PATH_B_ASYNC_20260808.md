# Path B async PRE cut — critical remeasure (20260808)

**Clear Thought:** sequential + scientific method + cause elimination + Occam + decision framework.  
**TS:** 20260808-143154 · model Qwen3.5-0.8B-4bit · 64 tok · fixed prompt `hi`  
**redline:** `9df1dfe` + local signal-mem fence fix · **lemon:** ordered_join profile + async hostwait default

## User claims → verdict

| Claim | Verdict | Evidence |
|-------|---------|----------|
| “replay is WAY TOO HIGH” | **Partly right** | Old label `replay=2022us` (n=31) was a **SUM**, not per-call. Mean ≈ **65–90 µs/call**. Also **not pure RL doorbell** — bucket is PRE drain + submit + completion wait. Renamed `ordered_join` with TOTAL+MEAN. |
| “async not working” | **Confirmed** | `phase2-async-used` (WaitValue) → **rc=124**, no Generation (twice before fix, once after signal-mem). |

## Profile honesty (B1p1)

```
ordered_join TOTAL=2793.82us n=31 → MEAN/call=90.12us
pre_sync TOTAL=129.22us → MEAN=4.17us
```

Gen stays near baseline (111–117 t/s). A true **2 ms × 31** per-token tax would destroy gen; flat t/s confirms sum interpretation.

## Path B matrix (DECODE+OWN_RMSNORM+RMS_HSACO)

| Arm | Mode | rc | gen t/s | Notes |
|-----|------|----|---------|-------|
| B0 | product | 0 | **116.48** | |
| B1p1 | phase1-used | 0 | **111.35** | ordered_join mean 90 µs |
| B1p2 | phase2-used | 0 | **112.89** | ordered_join mean 103 µs |
| B1async WaitValue | phase2-async-used | **124** | — | hang after profile n=31 (host enqueue ~15 µs mean) |
| B1async hostwait | phase2-async-hostwait | 0 | **114.35** | submit_after works; host `rl_pm4_wait` |
| B0b | product | 0 | **117.42** | |

**No ≥2% B1 win. Product default stays OFF.**

## Root cause slice

1. **Working:** phase1, phase2 sync (WriteValue + WAIT_REG_MEM + host wait_signal).  
2. **Working:** async **submit_after** + host `rl_pm4_wait` (proves doorbell path).  
3. **Broken:** `hipStreamWaitValue32` on consumer fence after PM4 `WRITE_DATA` — product stream never unblocks.  
4. **Tried:** fence via `hipExtMallocWithFlags(hipMallocSignalMemory)` + mask `0xFFFFFFFF` — **still hangs**.  
5. **Remaining:** WRITE_DATA vs signal-memory coherence, or WaitValue beta on gfx1150/ROCm 7.13.

## Product policy (fail-closed honesty)

| Flag | Role |
|------|------|
| `MLX_REDLINE_PHASE2_ASYNC=1` | submit_after + **host wait** (`phase2-async-hostwait`) — safe diagnostic |
| `MLX_REDLINE_ASYNC_WAITVALUE=1` | experimental GPU WaitValue — **may hang**; not default |
| profile | `ordered_join` TOTAL + MEAN; labeled not pure doorbell / not gen t/s |

## Ship bar

Path B is **not shippable** until WaitValue path completes gen **and** B1 shows ≥2% gen win under contention. Until then OWN_RMSNORM remains research / default OFF.
