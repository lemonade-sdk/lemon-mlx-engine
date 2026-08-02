# Design C — residual lm_head cut (after 4-bit already landed)

> **Primary design doc:** [`DESIGN_C.md`](DESIGN_C.md) (loop fire). This file is the short product-mode checklist (temp×think gates).

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Status:** DESIGN ONLY — no ship claim until microbench + quality  
**Context:** Package already has 4-bit `lm_head` (~3.87 ms ≈ 11.5% of T₁). “Quantize to 4-bit” is **void**. Residual upside is **sparsity of logits materialization**, not more weight quant.

---

## Goal

Reduce **effective** lm_head work per token below ~3.9 ms without quality regression at:

| Mode | Why it matters |
|------|----------------|
| temp=0 greedy | Argmax only needs max logit |
| temp=0.7 + top_p | Needs full (or large) distribution for categorical / RS |
| thinking ON | Long traces; quality sensitive |
| MTP RS | Trunk needs logprobs for accept ratio + residual |

---

## Options

### C1 — Greedy-only fast path (temp=0)
When `temperature==0` and no top_p/rep: compute **argmax without full softmax**.
- Engine already uses `argmax` on trunk for greedy MTP verify.
- Check whether `linear_fwd` still materializes full vocab logits every step — if yes, explore fused argmax-from-qmm or chunked max reduction.
- **Does not help** temp>0 or RS residual path.

### C2 — Two-stage / candidate set (all modes)
1. Coarse score: cheap projection or chunked max over vocab tiles.
2. Materialize full logits only for top-K candidates (K~2k–8k) + always-keep set (EOS, specials).
3. Sample / argmax on reduced set.

**Risks:** top_p mass leakage; RS ratio bias if draft and trunk use different support; thinking-mode diversity collapse.

### C3 — Chunked matmul + early exit (greedy)
Stream vocab in tiles; keep running max; skip soft fully. Same as C1 family.

### C4 — Leave as-is
If product modes (0.7+think+RS) cannot use C1 and C2 quality fails: **close residual** with measured 11.5% tax accepted.

---

## Quality gates (must pass before product)

| Gate | Pass |
|------|------|
| temp=0 no-think Fourier 256 | gen t/s ≥ baseline; text coherent |
| temp=0.7 no-think | coherent multi-sentence |
| temp=0.7 think ON max_tokens≥512 | thinking closes or usable; no garble |
| MTP RS temp=0.7 | accept rate not collapsed; Maxwell-style short PASS |
| Goldens | emit/accept tests still green |

---

## Notable vs fund (see [`../NOTABLE_WINS.md`](../NOTABLE_WINS.md))

**User policy:** any measured performance improvement is **notable** — including &lt;5%.

| Result | Notable? | Fund / ship |
|--------|----------|-------------|
| C1 any **logged** gen t/s or head-ms win at temp=0, quality OK | **YES** | Prefer fund if ≥~3–5% e2e **or** cheap; still document +1–2% |
| C2 any logged win at temp=0.7 / think / RS, quality OK | **YES** | Fund if multi-% or strategic product path |
| Measured regress | **YES (as regress)** | Do not ship |
| No measured delta | — | Design only; no claim |

**HARD BAN:** no +15–25% marketing from BF16 math on this package.

---

## Implementation sketch (if funded)

1. Feature flag e.g. `MLX_LM_HEAD_GREEDY_CHUNK=1` / `MLX_LM_HEAD_CANDIDATES=K`.
2. Hook only in `Qwen35MoEModel::call_impl` / sampler boundary.
3. Microbench before/after isolated + full gen matrix (temp×think).
4. PR separate from MTP stream product.

---

## Decision for next measure fire

Run **product-mode gen matrix** (temp × think × eager/MTP) to establish **baselines** before any C1/C2 code. Baselines live in `T_matrix_*.txt`.
