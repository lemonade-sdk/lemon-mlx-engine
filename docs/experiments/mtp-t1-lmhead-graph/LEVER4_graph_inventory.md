# Lever 4 — Graph decode MoE+GDN 35B

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Status:** **LEVER4_KILL** (2026-08-02 field probe)  
**HARD BAN:** do not cite pure-path **829 t/s** as real product speed (fake TPS + garble).

---

## Existing machinery (code)

| Mechanism | Location | Notes |
|-----------|----------|-------|
| `gpu_set_graph_decode_mode(L==1)` | `generate.cpp` decode path | T=1 decode can enter graph mode |
| Fixed `graph_decode_input()` buffer | `graph_decode.h` / `.cpp` | Capture/replay address stability |
| `graph_decode_enabled()` | `MLX_DECODE_GRAPH` env presence | Opt-in fixed buffers |
| `MLX_HIP_GRAPH_DECODE` / `MLX_USE_HIP_GRAPHS` | env (mlx side) | HIP graph capture for decode mode |
| `MLX_DECODE_GRAPH_PURE=1` | `generate.cpp` pure relaunch path | XOR with MTP; opt-in only |
| Prefill graphs | `MLX_PREFILL_ONE_GRAPH`, F1–F3 | **Missed ≥10% pp/s bar** on this stack |
| MTP sequential verify | default | Uses graph-decode mode for T=1 trunk |
| MTP batch verify | `MLX_MTP_BATCH_VERIFY=1` | **S4 KILL** on gfx1150 |

---

## Field probe (2026-08-02) — same-fire A/B

**Device:** gfx1150 / Radeon 890M · model LemonMLXE 35B MTP mlx-4bit · tip `b677fd8`  
**Common:** `MLX_ENABLE_QUANT_FUSE=1` `MLX_LOAD_MTP_HEAD=1` · temp=0 · `--no-think` · max_tokens=128 · Fourier short prompt · no MTP  

| Cell | Env | gen t/s | wall gen s | T₁ ms = 1000/tps | Quality | Log |
|------|-----|---------|------------|------------------|---------|-----|
| **Eager ctrl** | fuse only | **29.8084** | 4.2941 | **33.55** | Coherent Fourier | [`L4_E0_eager_ctrl.txt`](L4_E0_eager_ctrl.txt) |
| **HIP graph** | `MLX_DECODE_GRAPH=1` `MLX_HIP_GRAPH_DECODE=1` `MLX_USE_HIP_GRAPHS=1` | **28.733** | 4.4548 | **34.80** | Coherent Fourier | [`L4_E0_hip_graph.txt`](L4_E0_hip_graph.txt) |
| **Pure graph** | `MLX_DECODE_GRAPH_PURE=1` + HIP flags | **829.673** | 0.154278 | n/a | **GARBLE** (`Overview` loop) | [`L4_E0_pure_graph.txt`](L4_E0_pure_graph.txt) |

### Arithmetic (logs only)

| Metric | Value |
|--------|--------|
| HIP − eager | **−1.075** t/s (**−3.61%**) |
| HIP T₁ ≥ 32 ms? | **Yes** (34.80) |
| Gain ≥5%? | **No** |
| Pure TPS claimable? | **No** — HARD BAN fake TPS; quality EXIT |

HIP log confirms decode flags active:

```
[prefill-graph] ... HIP_GRAPH_DECODE=1 USE_HIP_GRAPHS=1 ...
```

### Kill criteria applied

| Criterion | Result |
|-----------|--------|
| T₁ stays ≥32 ms | **HIT** (HIP 34.80 ms) |
| gen t/s gain &lt;5% | **HIT** (−3.6%) |
| Capture fail / SEGV / thrash | No SEGV; HIP coherent but slower |
| Pure path product usable | **No** — degeneration + invalid speed |

---

## Verdict: **LEVER4_KILL**

1. Opt-in HIP graph decode does **not** improve 35B MoE+GDN T=1 gen on gfx1150 vs eager fuse; slight regression.  
2. `MLX_DECODE_GRAPH_PURE=1` is **broken** for this model/stack (token loop garble); **do not product-enable**; **do not** quote 829 t/s.  
3. MoE+GDN decode graph is **not a fundable T₁ lever** here (matches prefill F1–F3 miss and historical pure-flat note).  
4. **Do not re-open** without new capture/kernel evidence.

### Hypothesis (closed)

*If launch overhead dominates T₁, full HIP graph could cut T₁ 38→28–32 ms* — **refuted** on field: T₁ remains ~34–35 ms with graphs on; pure path invalid.

---

## Next program

- Lever 2: already **KILL**.  
- Lever 3: design parked; free-head ceiling sketch ~+13% only if C1/C2 later implemented.  
- Lever 4: **CLOSED / KILL**.  
- Optional: close L3 residual as **C4 leave-as-is** (accept ~11.5% head tax) → full loop STOPPED; or one dedicated C1 implement day.
