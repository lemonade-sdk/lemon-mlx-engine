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

## 5. Next measure (microbench B) — protocol sketch only

When GPU free, one of:

1. **Isolated:** warm ≥3 iters of `linear_fwd` / quantized lm_head on `[1,1,2048]` hidden; log wall ms.  
2. **Delta:** full decode step vs forward that stops before lm_head (or logits disabled) on same prompt/tokens; ≥3 warm.

Kill bar (program): **&lt;5% T₁** or **&lt;5 ms** → close lever 3.  
Fund bar: **≥8–10% T₁** or **≥5 ms** → design step C only after B.

---

## 6. Lever 2 cross-ref (no re-run)

| Config | gen t/s | verify_on_accept | Verdict |
|--------|---------|------------------|---------|
| S4 seq n2 | 27.216 | residual ~3.9 ms (not full T₁) | baseline |
| S4 batch n2 | 20.890 | mean **77.1** ms &gt; 67.7 | **KILL** |

Source branch: `exp/mtp-tps-ceiling` → `docs/experiments/mtp-tps-ceiling/RESULTS.md` + `S4_*.txt`.
