# Prefill HIP graphs — design + blockers (gfx1150)

**Status after F1–F3:** not product-viable on gfx1150 with engine-only / existing mlx ExecUpdate path.  
**Stop condition met:** three consecutive fires fail ≥10% long-prompt pp/s bar.

## What “prefill HIP graph” means on this stack

| Layer | Mechanism | Shape key | Product state |
|-------|-----------|-----------|---------------|
| Chunked prefill | `llm_default_prepare` + `prefill_step_size` | full-size `T=step` chunks | default eager |
| Manual-node graphs | `use_hip_graphs()` + `CommandEncoder::commit` | topology string | **default OFF** (net loss; historical RDNA3.5 SEGV) |
| Prefill ExecUpdate | `MLX_GRAPH_PREFILL_REPLAY` + `prefill_key_` | stable full-chunk topology | dead unless graphs on |
| ONE_GRAPH | `graph_decode_mode=true` in prepare only | whole forward one graph | F2 opt-in; ~+3% |
| Pure stream capture | `decode_capture_*` + decode arena | L=1 fixed addresses | **decode-only**; leave OFF on APU |
| **Missing** | **PrefillArena + stream capture of T=step** | fixed temps + KV write policy | **required for true build-once** |

## Why F1–F3 failed the ≥10% bar

1. **Prefill is compute-bound** on 35B-A3B 4bit + WMMA on 890M — launch batching is a small fraction of wall time.  
2. **Rebuild/ExecUpdate tax** still paid per chunk without deterministic addresses / build-skip for captured library nodes.  
3. **Variable remainder / absorb** does not create free topology reuse worth 10% (F3 slightly slower than eager).  
4. **+~2.5 GB peak** under graph path without matching throughput gain.  
5. Historical mlx note: whole prefill graph still ~5–8% **under** eager before hard-off — F1/F2 only recovered small single-digit gains.

## PrefillArena design (next work, if ever — in **mlx** not lemon-only)

Goal: **record-once / replay** full-size prefill chunk like pure decode, but for multi-token.

```
PrefillArena(step T, max_layers, d_model, ...):
  - fixed device buffers for all chunk temps (attention, MLP/MoE, norms)
  - fixed input token buffer [1, T] int32
  - KV: either pre-reserve max_seq and write at device offset, or
        stream-capture only ops that don't bake advancing KV addresses
  - capture slot keyed by (T, batch=1, model_id)
  - first full-size chunk: warmup + capture into hipGraphExec
  - later full-size chunks: memcpy tokens into fixed input, set pos, hipGraphLaunch
  - remainder T_rem < T: eager (or separate bucket captures for powers of two)
```

### Blockers (must clear before product opt-in)

| Blocker | Detail |
|---------|--------|
| B1 Address stability | MLX allocators reallocate temps each forward; capture bakes pointers |
| B2 MoE routing | Dynamic expert selection can change launch grid / which weights touch |
| B3 GDN / conv state | Multi-token fused paths vs T=1; must match prefill≡decode numerics |
| B4 Memory | Arena size ≈ one full-chunk activation footprint; may exceed APU headroom for large T |
| B5 Correctness | Bit-identical vs eager on golden prompts before any default-on |
| B6 FetchContent | Lives in `NripeshN/mlx` `rocm-support`; lemon only consumes |

### Engine-only leftovers (optional, default off)

- `MLX_PREFILL_ONE_GRAPH`, `MLX_PREFILL_ABSORB_TAIL` — experiment flags only  
- `MLX_HIP_GRAPH_PREFILL` / `USE_HIP_GRAPHS` — require mlx opt-in patch; **do not product-default**  
- Prefer **larger `MLX_PREFILL_STEP`** for prefill throughput (F3: step 512 eager **132 pp/s** vs step 128 **112 pp/s**) — independent of HIP graphs

## Product recommendation

- **Leave prefill eager** on gfx115x APU.  
- **Do not** set pure decode for product (separate known slight regression).  
- Revisit prefill graphs only with a **mlx PrefillArena PR** + measured ≥10% on this recipe:

```bash
MLX_PREFILL_STEP=128 MLX_PROFILE_PREFILL=1 \
  ./build/chat LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit \
  --max-tokens 16 --temperature 0 --no-think
# fixed long prompt ≈1653 tokens (see F1/F3 logs)
```
