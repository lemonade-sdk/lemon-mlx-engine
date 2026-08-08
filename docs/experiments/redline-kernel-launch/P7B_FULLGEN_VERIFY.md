# P7b — Full-gen L=1 sidecar correctness verify

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (device `side_acc` == host triangular sum after model L=1 steps)  
**Depends on:** P7 L1 retained sidecar arm  
**Not shipped:** gen t/s A/B · product default ON · `call_fn` replace · HIP graphs

---

## 1. Goal

Verify that **process-lifetime retained PM4 L=1 ticks** during a real model generation accumulate correctly on the dedicated sidecar accumulator — without replacing product `context_.call_fn`.

Prior P7 smoke only proved **inline arm** correctness (`sum(1..16)=136`) before model load. Full-gen residual is closed here.

---

## 2. Mechanism

| Piece | Behavior |
|-------|----------|
| Arm | Unchanged P7 (`MLX_REDLINE_SIDECAR=1` after micro PASS) |
| L=1 tick | `maybe_redline_sidecar_l1()` patches val=`n`, replays IB; host tracks `g_sidecar_expected += n` |
| End of gen | `TokenIterator::~TokenIterator` → `maybe_redline_sidecar_verify()` |
| Verify | `hipDeviceSynchronize` + D2H `side_acc`; PASS iff `side_obs == side_exp == n(n+1)/2` |

Env (all opt-in, defaults OFF):

| Env | Role |
|-----|------|
| `MLX_REDLINE_DECODE=1` | Master |
| `MLX_REDLINE_HSACO` | CO path (micro + arm) |
| `MLX_REDLINE_SIDECAR=1` | Arm + L=1 ticks + verify |
| pure-graph XOR | fail-closed (no arm / no tick / no verify) |

---

## 3. Smoke (gfx1150, Qwen3.5-0.8B-4bit local snapshot)

Model:  
`/home/antmi/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462`  
`MLX_SKIP_WARMUP=1` `HF_HUB_OFFLINE=1` `--max-tokens 16 --temperature 0 --raw` prompt `hi` then `quit`.

| Case | Result | Log |
|------|--------|-----|
| off | **0×** `[redline]` | `logs/p7b-off-20260808-114354.err` |
| on-fullgen | arm PASS; L1 tick; **`sidecar L1 fullgen PASS n=17 side_obs=153 side_exp=153`** | `logs/p7b-on-fullgen-20260808-114354.err` |
| xor (`MLX_DECODE_GRAPH_PURE=1`) | fail-closed; no fullgen PASS | `logs/p7b-xor-20260808-114354.err` |

Math: `sum(1..17) = 17*18/2 = 153`.  
`Generation:` line reports product-path token counts / t/s for the run context only — **not** a Redline gen t/s A/B claim.

### Fullgen banner excerpt

```
[redline] session READY ... micro=PASS ... sidecar=PASS side_obs=136 side_exp=136 ... sidecar_armed=1
[redline] sidecar L1 tick (retained PM4; call_fn still product; NOT gen t/s)
[redline] sidecar L1 fullgen PASS n=17 side_obs=153 side_exp=153 (... NOT gen t/s)
Generation: 16 tokens, ... tokens/s ...   # product path only; NOT redline A/B
```

Note: L=1 tick count (`n=17`) can exceed labeled `Generation:` tokens when prepare/`next` schedule an extra L=1 forward; correctness uses tick count and device acc, not the t/s line.

---

## 4. Code

- `maybe_redline_sidecar_verify()` — `redline_decode_session.{h,cpp}`  
- `TokenIterator::~TokenIterator` calls verify before pure-graph teardown — `generate.cpp`

---

## 5. Honesty

| Claim | Status |
|-------|--------|
| Full-gen L=1 retained PM4 acc correctness | **YES** (this fire) |
| Inline arm smoke (P7) | **YES** (prior) |
| Model gen t/s A/B vs eager | **NO** |
| Product default ON | **NO** |
| `call_fn` / qmm replace | **NO** |

---

## 6. Next

- Gen t/s A/B **only** after a **product-owned** op is replaced by Redline (measured path).  
- Optional: real engine-owned small op consuming `graph_decode_*` VRAM ptrs (still default OFF).
