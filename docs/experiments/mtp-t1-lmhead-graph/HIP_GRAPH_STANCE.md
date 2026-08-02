# HIP graphs: prefill-only stance (critical review of external advice)

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  
**Method:** Clear Thought (sequentialthinking, metacognitivemonitoring, structuredargumentation, scientificmethod) + field logs on this branch + prior F1–F3 / S4.  
**Verdict confidence:** **0.88** — agree with the **direction** of the advice; grounded in **our** measures, not authority alone.

---

## What “he” said (paraphrase)

1. Leave HIP graphs enabled **only for prefill**.  
2. **Pure** and **decode** graph paths don’t work / aren’t worth it.  
3. Decode needs a **stable memory address space** for graphs.  
4. Decode-cycle graphs in MLX are a **headache**, **not stable**.  
5. Product: **disable HIP graph for decode (and pure)**; only consider HIP graph for **prefill**.

---

## Is he right?

### Yes on mechanism (code + physics)

| Claim | Our evidence |
|-------|----------------|
| Decode needs stable addresses | **True.** Engine builds fixed buffers exactly for that: `graph_decode_input()` / `graph_decode_pos()` lazy-resident arrays (`graph_decode.cpp:58–67`, `22–28`) + in-place KV pos kernels so capture addresses don’t move. That complexity exists *because* decode graphs are fragile. |
| MoE/GDN hurts capture | **True.** Expert routing is data-dependent; GDN is recurrent. S4 multi-token batch verify **KILL** (topology/cost). Same class of problem as full-forward HIP capture. |
| Pure decode graph broken | **True on field.** `L4_E0_pure_graph.txt`: garble (`Overview`×N), **829 t/s** for 128 tok in 0.15s — **VOID** (not real decode). Quality fail + fake TPS. |
| Decode HIP graph not worth it | **Supported.** Same-session L4: |

| Cell | gen t/s (128 tok, temp0, no-think) | vs ctrl | Log |
|------|-------------------------------------|---------|-----|
| Eager ctrl | **29.808** | — | `L4_E0_eager_ctrl.txt` |
| `MLX_HIP_GRAPH_DECODE=1` + `USE_HIP_GRAPHS=1` | **28.733** | **−3.6% REGRESS** | `L4_E0_hip_graph.txt` |
| Pure graph | **829** (VOID) + garble | invalid | `L4_E0_pure_graph.txt` |

Under **NOTABLE_WINS** policy: decode HIP graph is a **notable regress**, not a win.

### Prefill-only — partially right, weakly measured upside

| Claim | Our evidence |
|-------|----------------|
| Prefill is the better surface for graphs | **Plausible.** Multi-token chunks amortize launch; F1–F3 program targeted prefill not decode. |
| Enable HIP graph for prefill as product default | **Not yet.** F1–F3 on 35B gfx1150: ~**+2–4% pp/s**, missed ≥10% bar; needs mlx `use_hip_graphs` opt-in patch historically hard-off upstream. **NOTABLE** if measured, **low FUND** for default-on. |
| “Prefill pure and decode don’t work” | Read as: **pure-stream capture + decode HIP graphs** are the bad pair. Prefill **ExecUpdate / ONE_GRAPH** is separate and still experimental. |

### Nuance (don’t over-agree)

1. **`gpu_set_graph_decode_mode(true)` on L=1** (engine ROCm path in `generate.cpp`) is **not** the same as “HIP graph always on.” It is a mode bit for the MLX backend (ExecUpdate / graph-friendly decode). Product HIP **capture** flags (`MLX_HIP_GRAPH_DECODE`, `MLX_DECODE_GRAPH`, pure) should stay **opt-in default off**.  
2. One L4 session is short (128 tok); direction is clear (regress + pure void) but not a 10-seed study.  
3. dGPU (H1) might change decode-graph economics; this stance is **890M / current mlx** first.

---

## Product / experiment decision

| Surface | Stance |
|---------|--------|
| **HIP graph decode** (`MLX_HIP_GRAPH_DECODE`, `MLX_DECODE_GRAPH`) | **OFF by default.** Do **not** fund product enable. Research opt-in only. |
| **Pure decode graph** (`MLX_DECODE_GRAPH_PURE`) | **OFF.** Field VOID (quality + fake t/s). |
| **HIP graph prefill** (`MLX_HIP_GRAPH_PREFILL`, `MLX_PREFILL_ONE_GRAPH`, replay) | **Opt-in experiment only.** Keep measuring small notables; not default-on without better bar. |
| **Lever 4 “decode graph for +20–35% T₁”** | **CLOSED as product bet** on this stack (field regress; expert+code agreement). |

---

## Clear Thought decision log

- **Thesis:** prefill-only HIP; decode/pure off for product.  
- **Support:** L4 logs + pure garble + address-stability code + MoE/S4.  
- **Limits:** prefill default-on not proven; engine L=1 mode bit remains for backend.  
- **Next:** do not thrash decode HIP flags; optional prefill-only notables if free; T₁ work stays fuse/lm_head residual / H1.

**HARD BAN:** never cite 829 t/s pure-graph as a win.
