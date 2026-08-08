# P8 — Engine-owned L=1 small op (live `graph_decode_*` VRAM)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PASS** (arm + L=1 product-buffer ticks + fullgen token-sum verify)  
**Depends on:** P5 micro · P6 gd_bind · P7 retained IB arm pattern  
**Not shipped:** gen t/s A/B · product default ON · `call_fn` replace · HIP graphs

---

## 1. Goal

Advance beyond synthetic sidecar counters: own a **real engine L=1 small op** that:

1. Writes the previous sample into product **`graph_decode_input`** (fixed VRAM).  
2. Resolves **`graph_decode_device_data_ptr`** each tick (consume product VRAM identity).  
3. D2H the live token id from that pointer and drives retained-PM4 `acc_k` patch+replay.  
4. Leaves **`context_.call_fn` product** (no forward replace).  
5. Never enables `graph_external_pos` (product RoPE/KV stay host-offset on eager path).

---

## 2. Env (all opt-in, defaults OFF)

| Env | Default | Meaning |
|-----|---------|---------|
| `MLX_REDLINE_DECODE` | unset | Master exact `"1"` |
| `MLX_REDLINE_HSACO` | unset | CO for micro + arm |
| `MLX_REDLINE_SMALL_OP` | unset | Exact `"1"` → arm + product-buffer L=1 ticks |
| `MLX_REDLINE_SIDECAR` | unset | Synthetic n ticks (skipped when SMALL_OP=1 owns IB) |
| pure-graph XOR | — | fail-closed; no session / no ticks |

---

## 3. Behavior

1. Session micro (P5/P6) on product `graph_decode_pos` as acc → PASS.  
2. If `SMALL_OP=1` (or `SIDECAR=1`): hipMalloc side acc; rebind; prime; inline arm smoke; keep IB → `sidecar_armed=1` + `small_op_armed=1` when SMALL_OP.  
3. `maybe_redline_small_op_l1(prev)` on L=1 in `TokenIterator::step` (before `call_fn`):  
   - `set_graph_decode_input_from(prev)`  
   - bookkeep pos via `set_graph_decode_pos` **without** `graph_external_pos`  
   - assert input VRAM ptr stable vs arm  
   - D2H token id from product input ptr → patch val → replay  
   - host `expected += token_id`  
4. `TokenIterator` dtor → `maybe_redline_sidecar_verify()`:  
   - SMALL_OP: `side_obs == sum(token_ids)`  
   - SIDECAR-only: triangular sum (P7b)

---

## 4. Smoke (gfx1150, Qwen3.5-0.8B-4bit)

Model:  
`/home/antmi/.cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/da28692b5f139cb0ec58a356b437486b7dac7462`  
`MLX_SKIP_WARMUP=1` `HF_HUB_OFFLINE=1` `--max-tokens 16 --temperature 0 --raw` prompt `hi` then `quit`.

| Case | Result | Log |
|------|--------|-----|
| off | **0×** `[redline]` | `logs/p8-off-20260808-114957.err` |
| on-smallop | arm PASS; L1 tick; **`small_op L1 fullgen PASS n=17 side_obs=15185 side_exp=15185`** | `logs/p8-on-smallop-20260808-114957.err` |
| xor (`MLX_DECODE_GRAPH_PURE=1`) | fail-closed; no small_op fullgen | `logs/p8-xor-20260808-114957.err` |

### Banner excerpt (on-smallop)

```text
[redline] session READY (... micro=PASS ... small_op=want ... sidecar_armed=1 small_op_armed=1)
[redline] gd_bind PASS ... stable=1
[redline] small_op L1 tick (product graph_decode_input VRAM val=5834; ... NOT gen t/s)
[redline] small_op L1 fullgen PASS n=17 side_obs=15185 side_exp=15185
  (product graph_decode_input token-sum; call_fn still product; NOT gen t/s)
Generation: 16 tokens, ... tokens/s ...   # product path only; NOT redline A/B
```

---

## 5. Code

| Piece | Location |
|-------|----------|
| Arm under `SMALL_OP` | `redline_decode_session.cpp` `try_micro_op` |
| L=1 product-buffer tick | `maybe_redline_small_op_l1` |
| Fullgen verify (token-sum) | `maybe_redline_sidecar_verify` (mode-aware) |
| Wire | `generate.cpp` `TokenIterator::step` L=1 + dtor |

---

## 6. Honesty

| Claim | Status |
|-------|--------|
| Engine-owned L=1 op consumes live `graph_decode_input` VRAM | **YES** |
| Full-gen token-sum correctness | **YES** (`15185/15185`, n=17) |
| `call_fn` still product | **YES** |
| Model gen t/s A/B vs eager | **NO** |
| Product default ON | **NO** |

---

## 7. Next

- Optional: measure host µs of product-buffer small-op chain (still not gen t/s).  
- Gen t/s A/B **only** after a measured product-path op replace (still forbidden as default ON).  
- Do **not** relabel product `Generation:` t/s as Redline win.
