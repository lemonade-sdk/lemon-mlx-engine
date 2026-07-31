# ROCm TPS optimization — operators + KV (this branch)

**Branch context:** `fix/rocm-gdn-fused2-optin` (PR #74)  
**Focus:** What further optimizations can raise **gen tokens/s**, given operators (GDN, quant matmuls, residual/norm) and **KV cache** behavior.  
**Not the focus:** macOS CI device selection (separate), LoopBrake, experiment log archives.

---

## Short framing

This cycle was mostly **correctness** (f32 SSM lifetime, softplus/`g`-dtype, prefill≡decode).

**Measured ROCm gen speed on 35B (gfx1150 class):** still about **~25–27 t/s**.  
Further TPS work is a **separate track** from thrash fixes.

---

## Where time goes on decode (product path)

Per token (rough stack, hybrid MoE):

```text
for each layer:
  attention layers:  QKV quant matmuls → RoPE → SDPA → out proj → residual+norm
  GDN layers:        in_proj (qkv|z|b|a) → conv step → SSM step → out → residual+norm
  MoE:               router + expert gate/up/down quant matmuls
lm_head:             quantized matmul (if packed)
KV:                  inplace write at offset
generate:            async one-behind sample
```

| Cost class | What’s hot |
|------------|------------|
| **Dominant on 35B** | Quantized matmuls (attention QKV, MoE experts, lm_head) |
| **GDN step / fused2** | Fewer launches help a bit — field matrix ~**+0.5–1.5 t/s**, not 2× |
| **KV** | Inplace already avoids full-cache copies under async; “stuck offset” thrash was **refuted** (not a free TPS win) |

---

## What this branch already has (ops / KV)

| Piece | Default | Perf role |
|-------|---------|-----------|
| **KV inplace** (`kv_inplace_update`) | **ON** (`MLX_KV_INPLACE_OFF` to disable) | Avoids COW full-cache copy each token under async one-behind — **already the fast path** |
| **Device-pos KV** (`update_at_pos` + `graph_decode_pos`) | Pure-graph only | Needed for capture-once; not default |
| **Eager L=1** `gpu_set_graph_decode_mode(true)` | **ON** (ROCm) | ExecUpdate-style whole-forward refresh — already on |
| **async_eval** one-behind | **ON** | Overlaps sample with next setup — keep unless debugging |
| **f32 SSM lifetime** | **ON** (this PR) | Correctness; small GPU tax, big CPU tax |
| **fused2** `gdn_fused_decode` | **Auto-on** at tip (opt-out: `MLX_GDN_FUSED2=0` or `MLX_GDN_NO_FUSED2=1`) | 1 launch vs many for GDN T=1 — modest TPS |
| **Quant fuse** QKV / in_proj / gate_up | **OFF** (`MLX_ENABLE_QUANT_FUSE=1`) | **Largest remaining “switch” on this stack** if stable |
| **Pure-graph** | **OFF** (`MLX_DECODE_GRAPH_PURE=1`) | Capture-once; note: **eager often faster on gfx1151 APU (~68 vs ~64)** |
| **MTP** | **OFF** (head skip) | Speculative multi-token — biggest theoretical TPS if acceptance is good |
| **KV quant** (`--kv-bits`) | Off | Memory; sometimes latency tradeoff |
| **MLX_KV_OFFSET_LOG** | Off | Debug only — **hurts** TPS if left on |

---

## Further optimizations ranked (for higher gen t/s)

### Tier 1 — Measure first, already wired (highest ROI / lowest code)

| # | Lever | How | Expected TPS | Risk |
|---|--------|-----|--------------|------|
| **1** | **Quant matmul fusion** | `MLX_ENABLE_QUANT_FUSE=1` | Often **meaningful** (fewer quant launches: QKV, GDN in_proj, SwiGLU gate\|up) | Numeric/quality; must field A/B |
| **2** | **Keep fused2 on** (tip auto-on) | Avoid `NO_FUSED2` / `FUSED2=0` unless A/B | **Small** (~0–2 t/s on 35B iGPU) | Historical thrash; re-check quality |
| **3** | **Leave KV inplace on** | Don’t set `MLX_KV_INPLACE_OFF` | **Baseline** — turning it off usually **hurts** TPS | — |
| **4** | **Leave pure-graph off on APU** | Don’t set `MLX_DECODE_GRAPH_PURE=1` on gfx115x APU | Avoids **regression** (~few t/s) | Pure can help **launch-bound dGPU** (e.g. R9700-class) |

#### Locked measure recipe

One binary, same 35B prompt, compare gen t/s from `[TPS]` lines:

```bash
# baseline (product)
./build/chat MODEL --max-tokens 512 ...

# quant fuse
MLX_ENABLE_QUANT_FUSE=1 ./build/chat MODEL ...

# fused2 off (control)
MLX_GDN_NO_FUSED2=1 ./build/chat MODEL ...

# pure (only if discrete GPU to compare)
MLX_DECODE_GRAPH_PURE=1 ./build/chat MODEL ...
```

Do **not** optimize blind.

---

### Tier 2 — Real product TPS upsides (more work / quality gates)

| # | Lever | What | Expected TPS | Risk |
|---|--------|------|--------------|------|
| **5** | **MTP speculative decode** | `MLX_LOAD_MTP_HEAD=1` + `--use-mtp` | Can be **large** if accept rate high | Quality, memory, thrash class |
| **6** | **MoE routing / expert packing** | Fewer launches, better expert batching | Medium if MoE is hot | Large engineering |
| **7** | **lm_head path** | Already prefers packed quant matmul | Already good | — |
| **8** | **Prefill step size / chunking** | Affects **prefill** more than decode | First-token latency | Not decode TPS |

---

### Tier 3 — Do **not** chase for ROCm 35B iGPU TPS

| Lever | Why not now |
|-------|-------------|
| More f32↔bf16 churn “for speed” | Undoes thrash fix; GPU win tiny |
| LoopBrake | Not a TPS optimizer; rejected for product |
| KV “fix stuck offset” | **Refuted** under thrash; not a speed path |
| Expecting fused2 to 2× 35B | Field matrix: **no** |
| Expecting ~63 t/s on 35B gfx1150 | That’s **0.8B-class** rates |

---

## Operator / KV map (what actually moves tokens)

```text
                    ┌─────────────────────────────────────┐
  HOT (TPS)         │ Quant matmuls (attn QKV, MoE, head)  │  ← quant fuse, MTP
                    │ SDPA / attention                     │
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
  WARM              │ GDN conv+SSM (fused2 vs multi-op)   │  ← fused2 modest
                    │ residual+RMSNorm fused kernels      │  already there
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
  ALREADY OPT       │ KV inplace write                    │  default ON
                    │ async one-behind sample             │  default ON
                    │ L=1 graph decode mode (ROCm)        │  default ON
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
  CORRECTNESS TAX   │ f32 SSM store (this PR)             │  small on GPU
                    │ softplus logaddexp / g cast         │  negligible
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
  OPT-IN / CAUTION  │ pure-graph (often slower on APU)    │
                    │ MTP (quality + memory)              │
                    │ MLX_SYNC_DECODE (slower, debug)     │
                    └─────────────────────────────────────┘
```

---

## How correctness work interacts with TPS

| Change | On **ROCm GPU 35B** | On **CPU** |
|--------|---------------------|------------|
| f32 SSM lifetime | **~Flat** gen t/s in matrix (~25–27) | **Much slower** (CI mac path when forced CPU) |
| softplus / g-dtype | Negligible | Negligible |
| fused2 on | **Small +** or noise | N/A / little |
| Quant fuse (not default) | Potential **real +** | Helps CPU too if used |

**Takeaway:** You did **not** “pay half your ROCm TPS” for f32 SSM. You paid for **stability**. Headroom for more t/s is mostly **matmul fusion + speculative decode**, not more GDN dtype micro-opts.

---

## Concrete next experiments (TPS-only)

1. **Quant fuse A/B** on 35B gfx115x — same prompt length; report gen t/s and any thrash.  
2. **Fused2 on vs `NO_FUSED2=1`** — confirm tip polarity still ~flat TPS + quality.  
3. **MTP smoke** (if head loaded) — short fixed prompt; measure tokens/s and accept rate; stop if thrash returns.  
4. **Pure-graph only on discrete GPU** if you care about launch-bound dGPUs; **not** first for 890M-class APU.

---

## Bottom line

| Question | Answer |
|----------|--------|
| What further TPS opts exist? | **Quant fuse (wired, off), MTP (wired, off), fused2 (modest), pure-graph (situational)**; KV inplace already on. |
| Biggest likely win without new architecture? | **`MLX_ENABLE_QUANT_FUSE=1` + measure**; then **MTP** if quality holds. |
| More GDN f32/softplus churn? | **Wrong lever for TPS** on GPU. |
| Pure-graph for APU? | **Usually not** — eager often beats pure on gfx1151-class. |

---

## Related code (this repo)

| Area | Path |
|------|------|
| Decode loop / pure-graph / KV offset log | `src/common/generate.cpp` |
| KV inplace / device-pos write | `src/common/kv_cache.cpp` |
| GDN HIP operators (step, fused2, conv, norms) | `src/common/gated_delta.cpp` |
| MoE T=1 path, quant fuse, fused2 gate | `src/llm/models/qwen35_moe.cpp` |

---

*Operational guidance for TPS follow-on work after GDN correctness on this branch. Not a claim that any single opt is proven without A/B measurement.*
