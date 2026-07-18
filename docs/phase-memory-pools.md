# Phase-scoped memory pools (ROCm)

**Branches:** `feat/phase-memory-pools` (engine) · `rocm-support-memory` (NripeshN/mlx)

## Problem

A single exact-size freelist + “never hipFree on free” fixed **training**
(no hipFree storms before Adam, stable B×T reuse) but regressed **decode**
(~62 → ~42 tok/s on gfx1151). Prefill intermediates recycled into the same
freelist and poisoned HBM accounting for L=1 decode.

## Design (lifetime over size)

| Pool / phase | Role |
|--------------|------|
| **Permanent** | Weights (unchanged — long-lived arrays) |
| **State** | KV / GDN (grow-only cache objects, not freelist thrash) |
| **Prefill workspace** | Generation-stamped freelist entries → bulk-dropped |
| **Decode workspace** | Looser freelist util (0.5); optional bump arena later |
| **Train workspace** | Exact freelist (1.0); optional TrainStepArena later |

### Phase API (`mlx::core::rocm`)

```text
Idle | Load | Prefill | Decode | Train
```

- `set_memory_phase(phase)` — stamps new allocs with a generation; sets freelist
  `min_utilization` (Train=1.0, else 0.5 unless `MLX_FREELIST_UTIL` override).
- `memory_end_prefill()` — `drop_generation(prefill_gen)` + enter Decode.
- `memory_drop_generation(gen)` — free freelist buffers with that stamp.
- `MLX_MEMORY_PHASE_DEBUG=1` — log transitions and drops.

### Engine wiring

`TokenIterator::prepare()`:

1. `set_memory_phase(Prefill)`
2. run prefill / first sample
3. `eval` first token
4. `memory_end_prefill()` → Decode

`generate()` ends with `set_memory_phase(Idle)`.

Engine header: `include/mlx-lm/common/memory_phase.h`.

## Phase map (future)

| Phase | Still TODO |
|-------|------------|
| Decode workspace arena | Reserve worst-case L=1 intermediates at load |
| TrainStepArena | Bump activations per step (reuse existing decode_arena hooks) |
| PrefillArena | Dedicated large arena instead of freelist stamp alone |
| Segment pool | Rare hipMalloc of 64–512 MB slabs under both arenas |

## A/B knobs

| Env | Effect |
|-----|--------|
| `MLX_FREELIST_UTIL=0.5\|1.0` | Force freelist utilization |
| `MLX_MEMORY_PHASE_DEBUG=1` | Phase / drop logging |
| `MLX_DECODE_GRAPH_PURE=1` | Enable pure-graph decode (default off; eager is preferred) |

## Validation checklist

1. Multi-request server: no segfault after long prefill → decode
2. TPS A/B vs previous (~42 eager baseline on gfx1151 Qwen3.6 6-bit)
3. Train step: still no hipFree storm (phase Train + util 1.0)
4. `MLX_MEMORY_PHASE_DEBUG` shows non-zero drop after prefill on large prompts

## Related

- Allocator train fixes (exact freelist, no free-path hipFree) remain the Train policy.
- Pure-graph decode arena is orthogonal; shares the “fixed workspace” idea.
