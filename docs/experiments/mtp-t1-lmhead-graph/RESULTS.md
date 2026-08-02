# lm_head / embed inventory — LemonMLXE 35B MTP mlx-4bit

**Branch:** `exp/mtp-t1-lmhead-graph`  
**Date:** 2026-08-02  
**Method:** safetensors **header-only** parse (no full tensor load) + `config.json` + code path cites.  
**Model snapshot:**  
`~/.cache/huggingface/hub/models--LemonMLXE--Qwen3.6-35B-A3B-MTP-mlx-4bit/snapshots/5f638dff286ea1a97a6a0b673f50acc9b3c9aa4b`  
**Single file:** `model.safetensors` size **21187756703** bytes (~21.19 GB).

Honesty: **no wall-clock ms and no gen t/s invented** this fire.

---

## 1. Config facts

| Field | Value | Source |
|-------|--------|--------|
| `model_type` | `qwen3_5_moe` | `config.json` |
| `quantization` | `{bits: 4, group_size: 64}` | top-level |
| `tie_word_embeddings` | **false** | top-level + `text_config` |
| `text_config.vocab_size` | **248320** | `text_config` |
| `text_config.hidden_size` | **2048** | `text_config` |
| `text_config.dtype` | `bfloat16` (activation/default) | `text_config` |

### Program claim correction

| Sketch claim | Field package truth |
|--------------|---------------------|
| vocab ≈ 151936 | **248320** |
| BF16 full head ≈ 622.33 MB | BF16 full = 248320×2048×2 = **1017118720 B ≈ 1017.12 MB** |
| “full matrix read every token at BF16” | **Head is already 4-bit packed on disk** |

---

## 2. Weight-map inventory (header)

### `lm_head` (untied, quantized)

| Tensor | dtype | shape | nbytes | MB |
|--------|-------|-------|--------|-----|
| `lm_head.weight` | **U32** | [248320, 256] | 254279680 | **254.280** |
| `lm_head.scales` | **BF16** | [248320, 32] | 15892480 | **15.892** |
| `lm_head.biases` | **BF16** | [248320, 32] | 15892480 | **15.892** |
| **Total store** | | | **286064640** | **286.065** |

Layout consistency with bits=4, group_size=64:

- 4-bit values per `u32` = 8 → `hidden/8 = 2048/8 = 256` packed cols.  
- groups per row = `2048/64 = 32` → scales/biases `[vocab, 32]`.

### `model.embed_tokens` (also quantized on disk; separate from lm_head)

| Tensor | dtype | shape | nbytes | MB |
|--------|-------|-------|--------|-----|
| `model.embed_tokens.weight` | **U32** | [248320, 256] | 254279680 | **254.280** |
| `model.embed_tokens.scales` | **BF16** | [248320, 32] | 15892480 | **15.892** |
| `model.embed_tokens.biases` | **BF16** | [248320, 32] | 15892480 | **15.892** |
| **Total store** | | | **286064640** | **286.065** |

`tie_word_embeddings=false` ⇒ **two** independent quantized tables in the checkpoint (not a shared tied head).

---

## 3. Runtime path (code cites)

Decode logits path for this MoE model:

```1071:1091:src/llm/models/qwen35_moe.cpp
LMOutput Qwen35MoEModel::call_impl(...) {
    auto hidden = model_.forward_prenorm(input.tokens, cache);
    auto post_norm = model_.apply_norm(hidden);
    // ...
    lm_head_weight_.has_value() ? linear_fwd(post_norm, lm_head_weight_.value())
                                 : model_.embed_as_linear(post_norm);
}
// forward_impl also linear_fwd(out, lm_head_weight_) when present
```

`linear_fwd` → `linear_forward` (quant-aware):

```152:155:src/llm/models/qwen35_moe.cpp
static mx::array linear_fwd(const mx::array& x, const mx::array& w,
                              const std::optional<mx::array>& bias = std::nullopt) {
    return linear_forward(x, w, bias.has_value() ? &bias.value() : nullptr);
}
```

Quant registry: non-embedding prefixes with `.scales` register for **`quantized_matmul`**; **embeddings dequantize at load** for `take()`:

```70:87:src/common/quantize_utils.cpp
// Embedding weights use mx::take() for lookup, not matmul.
// They must be dequantized at load time (quantized_matmul won't help).
bool is_embedding = (prefix.find("embed") != std::string::npos);
if (is_embedding) {
    packed = mx::dequantize(...);
} else {
    reg.register_weight(member_ptr, scales, biases, group_size, bits);
}
```

Constructor comment (intent for large vocabs) — lm_head should stay quantized:

```1058:1064:src/llm/models/qwen35_moe.cpp
// Always allocate lm_head_weight_ so it is part of weight_map(). For TIED
// embeddings, sanitize() wires a packed quantized copy of the embedding into
// it, so the lm_head matmul runs through quantized_matmul (~4x less memory
// than the dequantized embedding table — the single largest per-token load).
lm_head_weight_ = mx::zeros({args.vocab_size, args.hidden_size});
```

This **field package is untied** and already contains explicit `lm_head.{weight,scales,biases}` — so load uses those keys (not the tied-embedding duplicate path at L1258–1275).

Convert tool default for *building* packages is `--lmhead-bits 8` (`examples/convert.cpp` L13–14, L41, L61–62), but **LemonMLXE's published 4bit package uses 4-bit lm_head** (header above).

---

## 4. Implications for Lever 3

| Question | Answer this fire |
|----------|------------------|
| Is lm_head BF16 dense on LemonMLXE 35B MTP mlx-4bit? | **No — already 4-bit pack** |
| Is embed_tokens BF16 on disk? | **No — 4-bit pack; dequantized at load for embedding lookup** |
| Can we fund “quantize lm_head to 4-bit” as free +15–25%? | **No** — already 4-bit; % **unmeasured** |
| Residual upside? | Only if **measured** 4-bit head still ≥5 ms or ≥8–10% of T₁ → two-stage sampler / further cut designs |

### Bandwidth *sketch* (hypothesis only — **not a field result**)

Store size ratio vs BF16 full: `286.06 / 1017.12 ≈ 0.281`.  
Any “~13–14 ms BF16” style estimate from the program sketch **does not apply** to this package without re-derivation **and** microbench B. **No ms claimed here.**

---

## 5. Microbench B — isolated 4-bit lm_head qmm (MEASURED)

**Tool:** `examples/bench_lm_head.cpp` → `build/bench_lm_head`  
**Weights:** real `lm_head.{weight,scales,biases}` extracted from LemonMLXE package (header-matched U32/BF16).  
**Op:** `mx::quantized_matmul(x, w, scales, biases, transpose=true, gs=64, bits=4)` with `x` shape **[1, 2048]** BF16.  
**Device log:** `[mlx-rocm] bound HIP device 0: gfx1150 … cus=8`  
**Full log:** [`B_lm_head_qmm.txt`](B_lm_head_qmm.txt)

| Phase | wall_ms |
|-------|---------|
| warm[0] (cold) | 40.0794 |
| warm[1] | 4.02701 |
| warm[2] | 3.87389 |
| **timed mean (n=10)** | **3.86958** |
| timed min | 3.76766 |
| timed max | 4.12021 |

### Same-fire T₁ denominator (eager SAFE fuse, 128 gen tok)

Log: [`B_t1_eager_ref.txt`](B_t1_eager_ref.txt)

```
Prompt:     23 tokens, 30.5125 tokens/s, 0.753791s
Generation: 128 tokens, 29.68 tokens/s, 4.31267s
```

| Derived (arithmetic from logs only) | Value |
|-------------------------------------|--------|
| T₁ ms = 1000 / 29.68 | **33.6927 ms** |
| head fraction = 3.86958 / 33.6927 | **11.48%** |
| Kill &lt;5% T₁? | **No** |
| Fund ≥8–10% T₁? | **Yes** |
| Fund ≥5 ms abs? | **No** (3.87 &lt; 5) |
| Free-head ceiling (sketch only): T₁−mean → t/s | ~33.53 t/s (~+13% vs 29.68) — **not measured** |

### Limitations (honesty)

- Isolated qmm **≠** full `call_impl` residual; no stop-before-lm_head delta this fire.  
- Fraction uses T₁ that **includes** head cost (ratio, not pure leave-one-out).  
- Do **not** claim product +15–25% or two-stage wins until implemented and measured.

### Verdict (after B)

**Lever 3 stayed OPEN for design C** (kill &lt;5% T₁ not met). Already-4-bit conversion path remains **void**.

---

## 5b. Design C (no new GPU numbers)

Full plan: [`DESIGN_C.md`](DESIGN_C.md).

| Decision | |
|----------|--|
| Primary design | Two-stage shortlist + exact K-row head (temp=0 first); embed-as-proxy **rejected** (BF16 dequant embed ~1GB worse than 4-bit head) |
| Secondary | Kernel-only faster full qmm (quality-neutral) |
| Implement this fire | **No** |
| Upside claim | Cap narrative at free-head sketch **~+13%**; **no** +15–25% |
| Next | Lever 4 graph inventory; implement C only if gates in DESIGN_C §3 met later |

---

## 6. Lever 2 cross-ref (no re-run)

| Config | gen t/s | verify_on_accept | Verdict |
|--------|---------|------------------|---------|
| S4 seq n2 | 27.216 | residual ~3.9 ms (not full T₁) | baseline |
| S4 batch n2 | 20.890 | mean **77.1** ms &gt; 67.7 | **KILL** |

Source branch: `exp/mtp-tps-ceiling` → `docs/experiments/mtp-tps-ceiling/RESULTS.md` + `S4_*.txt`.

---

## 7. Product-mode matrix: temperature × thinking (2026-08-02)

**Why:** Product defaults are temp **0.7** and **thinking ON** unless `--no-think`. Greedy temp=0 only was under-representing lm_head/sampler cost and quality risk for Design C.

**Runner:** `run_temp_think_matrix.sh`  
**Env:** `MLX_ENABLE_QUANT_FUSE=1` `MLX_LOAD_MTP_HEAD=1` · model LemonMLXE 35B MTP 4bit · gfx1150  
**HARD BAN:** numbers only from logs below.

| Cell | Mode | max_tok | gen t/s | log |
|------|------|---------|---------|-----|
| **E0** | eager temp=0 `--no-think` | 128 | **29.610** | `T_E0_temp0_nothink.txt` |
| **E07** | eager temp=0.7 top_p=0.9 `--no-think` | 128 | **29.879** | `T_E07_temp07_nothink.txt` |
| **E07T** | eager temp=0.7 **think ON** | 512 | **29.929** | `T_E07T_temp07_think.txt` |
| **M0** | MTP n2 temp=0 `--no-think` | 128 | **27.146** | `T_M0_mtp_temp0_nothink.txt` |
| **M07** | MTP n2 temp=0.7 **RS** `--no-think` | 128 | **26.135** | `T_M07_mtp_temp07_nothink.txt` |
| **M07T** | MTP n2 temp=0.7 **RS + think** | 512 | **25.216** | `T_M07T_mtp_temp07_think.txt` |

### TPS takeaways

1. Eager gen t/s is **flat** across temp0 / temp0.7 / think (~29.6–29.9) on these short runs — sampling + think template do **not** dominate wall vs trunk T₁.
2. MTP greedy ~**27.1**; MTP RS ~**26.1** (−3.7% vs M0); MTP RS+think ~**25.2** (−7.1% vs M0) — matches historical RS tax, not a new bug.
3. Design C **greedy-only** path (C1) only helps E0/M0; **product 0.7+think** needs C2 two-stage with quality gates or accept residual head tax.

### Quality takeaways (honest, short)

| Cell | Note |
|------|------|
| E07 no-think | Coherent Fourier overview structure |
| E07T think | Structured “thinking process” + Maxwell content; usable (no garble) |
| M07 no-think | Coherent Fourier sections; RS path OK |
| M07T think | Structured thinking + Gauss law content; no garble; **not** full Maxwell multi-turn bar |

Quality is **session-smoke**, not full Maxwell SAR re-certification.

### Implication for Design C

- **C1 (greedy argmax path):** **any** logged win on E0/M0 is **notable** ([`NOTABLE_WINS.md`](../NOTABLE_WINS.md)); fund priority rises with multi-% e2e; **does not** alone cover product default.
- **C2 (two-stage):** must pass E07 + E07T + M07 + M07T quality gates; any logged product-mode win is notable.
- **Baselines above** are the A/B denominators for any future C1/C2 implementation.
- Eager flat across temp/think is itself **notable**: product sampling does **not** regress decode t/s vs greedy on this stack.

---

## 8. Lever 4 inventory + field probe — **LEVER4_KILL**

Full writeup: [`LEVER4_graph_inventory.md`](LEVER4_graph_inventory.md).

**Same-fire tip `b677fd8` · fuse ON · temp=0 · no-think · 128 gen · Fourier · no MTP**

| Cell | gen t/s | T₁ ms | Quality | Log |
|------|---------|-------|---------|-----|
| Eager ctrl | **29.8084** | 33.55 | Coherent | `L4_E0_eager_ctrl.txt` |
| HIP (`MLX_DECODE_GRAPH` + `MLX_HIP_GRAPH_DECODE` + `MLX_USE_HIP_GRAPHS`) | **28.733** (−3.61%) | **34.80** | Coherent | `L4_E0_hip_graph.txt` |
| Pure (`MLX_DECODE_GRAPH_PURE=1`) | **829.673** | n/a | **GARBLE** Overview loop | `L4_E0_pure_graph.txt` |

| Kill bar | Outcome |
|----------|---------|
| T₁ ≥ 32 ms with graphs | **HIT** (HIP 34.80) |
| Gain &lt;5% | **HIT** (−3.6%) |
| Pure product usable | **No** — fake TPS ban + quality EXIT |

**Verdict: LEVER4_KILL.** Do not product-enable pure; do not re-open without new evidence.
