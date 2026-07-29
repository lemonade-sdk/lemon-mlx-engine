# ROCm Decode Degeneration — Where and How to Test

**Purpose:** Practical guidance on whether to investigate same-token infinite loops on a local Ubuntu ROCm machine, on Windows, or via CI/CD.

**Related context:** Discord field reports (gfx1152, Qwen3.5 hybrid models) show coherent generation then hard collapse into repeated tokens/phrases (e.g. `I am not human.`, `synchronization`). Engine analysis points at **decode/GDN/ROCm state**, not CLI amplifiers alone.

**Date:** 2026-07-28

---

## Short answer

| Question | Answer |
|----------|--------|
| Where should I test degeneration? | **Ubuntu + AMD ROCm** (your local machine with a real GPU). |
| Should I wait for CI/CD? | **No** — not to prove or disprove this class of bug. |
| Is Windows useful? | **Yes for editing/PRs.** **No for reproducing** the ROCm GDN collapse. |

**Test the degeneration on Ubuntu + ROCm. Do not wait for CI. Use Windows as the edit/PR machine only.**

---

## Why Ubuntu (ROCm), not CI, not Windows-first

| Environment | Good for | Bad for this bug |
|-------------|----------|------------------|
| **Ubuntu + ROCm (gfx1151 / gfx1152)** | Same class of hardware as field reports; real GDN HIP kernels; env A/B in minutes | — |
| **Windows** | Source edits, reading diffs, opening PRs | No real ROCm path for this engine; you will not hit fused GDN collapse |
| **CI `ubuntu-rocm`** | Green builds, dual-lane “2+2” smoke, load hygiene | Short smokes; **not** long multi-turn / Maxwell-style collapse; self-hosted may be **gfx1151** while field report was **gfx1152** |
| **CI CPU / macOS** | Unrelated unit/smoke coverage | No ROCm GDN path |

The failure class is **ROCm T=1 GDN / decode state**. CI will not exercise that the way a local `./chat` session does (short or multi-turn).

---

## Framing (engine vs amplifiers)

| Item | Role |
|------|------|
| **Root class** | Engine decode path: GDN fused T=1 kernels, async one-behind pipeline, in-place KV, possible q-norm scale issues |
| **Amplifiers only** | `--repetition-penalty 1.0` (engine no-op), huge `--max-tokens` (only lengthens a stuck loop) |
| **Separate issue** | `hip/hip_runtime.h` missing → `ROCM_HOME` / include layout (build/JIT), not mid-decode loops |

Do **not** treat “fixing a better repetition penalty” as the investigation. Env toggles and local ROCm repros do.

---

## What “a lot to test” actually means

You do **not** need every experiment. Use a **minimal ladder** on Ubuntu.

### Must-run (about 1–2 hours on Ubuntu)

#### 1. Baseline (reproduce once)

- **Short:** `who are you?`
- **Optional longer:** multi-turn science/radar thread if short does not loop
- Note GPU line from chat (e.g. `gfx1151` vs `gfx1152`)
- Prefer **MTP off** for the first matrix if using an MTP-tagged model

#### 2. One env at a time (stop when loops disappear)

Same binary, same model, same prompt(s). Change **only** the env var:

```bash
# Baseline (no extra env)
./chat <model> ...

# GDN path bisect
MLX_GDN_NO_FUSED2=1 ./chat <model> ...
MLX_GDN_NO_FUSED=1 ./chat <model> ...

# Pipeline / KV bisect
MLX_SYNC_DECODE=1 ./chat <model> ...
MLX_KV_INPLACE_OFF=1 ./chat <model> ...
```

Optional observational (does not “fix” by itself):

```bash
MLX_STATE_CKSUM=1 ./chat <model> ...
```

#### 3. Log results

For each run record:

| Field | Example |
|-------|---------|
| GPU | gfx1152 |
| Model | mlx-community/Qwen3.5-0.8B-4bit |
| Env | `MLX_GDN_NO_FUSED2=1` |
| Loop? | Y / N |
| When | e.g. after ~N tokens / first repeated phrase |
| Notes | pure graph unset? MTP off? |

That is enough to rank **H1 (GDN fused)** vs **H2 (async / KV)**.

---

## Defer (do not block on these)

- Full multi-axis ablation matrix (all env combinations)
- Pure-graph path (`MLX_DECODE_GRAPH_PURE=1`) until eager path is understood
- MTP-on runs until non-MTP baseline is clean
- Waiting for dual-lane CI (#73) or other smoke PRs
- Windows-local “smoke” as a substitute for ROCm

---

## Optional later (after one env clearly helps)

- `MLX_STATE_CKSUM=1` focused on the collapse step
- q-norm scale patch A/B (`1/D` vs `1/√D` on GDN `q_norm_w`) if GDN env fixes the loop
- Teacher-force prefill vs free decode (logits compare) for a gold engine-vs-sampler split
- gfx1151 vs gfx1152 same binary if both machines exist

---

## Practical split of work

| On **Ubuntu (ROCm)** | On **Windows / GitHub** |
|----------------------|-------------------------|
| Reproduce + env A/B | Draft PR after data |
| Confirm model (0.8B vs 35B-MTP) | Code review of `gated_delta` / `generate` / GDN |
| Note GPU arch | CI only after you know what to assert |
| Capture logs / phrases | Issue/PR description |

---

## Ranked engine hypotheses (reminder)

| Rank | Hypothesis | Discriminator |
|------|------------|---------------|
| **H1** | Prefill vs T=1 `gdn_fused_decode` (and/or conv step); possible q-norm under-scale | `MLX_GDN_NO_FUSED2`, `MLX_GDN_NO_FUSED`, later q-norm patch |
| **H2** | Async one-behind + in-place KV + ROCm `graph_decode_mode(L=1)` | `MLX_SYNC_DECODE`, `MLX_KV_INPLACE_OFF` |
| **H3** | Lazy GDN state not retired before next step | Often moves with `MLX_SYNC_DECODE` |
| **Low** | Pure-graph / inplace GDN | Only if `MLX_DECODE_GRAPH_PURE=1` |
| **Not root** | Rep-penalty / max_tokens | Amplifiers only |

---

## Recommended sequence

1. **Today (Ubuntu):** Baseline repro + four env toggles (table above).
2. **Do not** block on CI for this investigation.
3. Use **Windows** only as the edit/PR machine.
4. After **one env clearly stops the loop**, implement a targeted fix and only then consider a **cheap CI assertion** (if any).

---

## What to paste back for a fast lock on H1 vs H2

When reporting results, include:

1. GPU string from chat / `rocm-smi` (gfx1151 vs gfx1152)
2. Exact `./chat` command (model path, flags; note MTP on/off)
3. Baseline: loop Y/N and sample of collapsed text
4. For each of:
   - `MLX_GDN_NO_FUSED2=1`
   - `MLX_GDN_NO_FUSED=1`
   - `MLX_SYNC_DECODE=1`
   - `MLX_KV_INPLACE_OFF=1`  
   → **loop Y/N**

With that, the investigation can lock **GDN kernel path** vs **async/KV pipeline** in one pass.

---

## Bottom line

| Do | Don’t |
|----|--------|
| Test on **Ubuntu + ROCm** | Wait for CI to “find” long-gen degeneration |
| Run a **short env ladder** | Try to repro on Windows as the primary plan |
| Treat knobs as **amplifiers** | Treat rep-penalty as the fix for root cause |
| Use CI later for **build + targeted checks** | Expect dual-lane smoke to replace local ROCm chat |

---

## Related documents / code (for deeper dive)

| Area | Location |
|------|----------|
| Decode loop | `src/common/generate.cpp` (`TokenIterator::step`, `next`) |
| GDN fused HIP | `src/common/gated_delta.cpp` (`gdn_fused_decode`) |
| T=1 path selection | `src/llm/models/qwen35_moe.cpp` (and `qwen35` / `qwen3_next` GDN) |
| In-place KV | `src/common/kv_cache.cpp` |
| Field symptoms | Discord screenshots / multi-turn + short “who are you?” loops |

---

*This document is operational guidance for investigation prioritization, not a claim that any single root cause is already proven without the Ubuntu A/B results.*
