# Prompt-processing roots (I3 / P0 cluster)

**Branch:** `exp/prompt-processing-i3`  
**Parent:** `fix/mtp-stream-p0` @ `875a39d`  
**Sibling of:** `exp/redline-kernel-launch` (same parent; **not** forked from the Redline tip)  
**Status:** I3 **closed** by design (fresh KV every CLI turn). This doc archives the source analysis.

## Thesis

The classic multi-turn / “prompt processing” failure was a **cluster of independent, pre-existing engine defects** — not a single bug, and **not introduced** by merged PRs (#62, #64/#70, #71, #72, #74, #76, parent #63). Those PRs were the **corrective** response. The defects existed on `main` (and the tools branch) before them.

## Primary root: ChatSession multi-turn (I3 / P0)

**Defect:** reuse residual KV across CLI turns **and** re-apply the **full** chat template (system + entire history + new user) on the non-empty cache → **double-prefill**.

Effects: position mismatch, corrupted/overwritten context, after 1–N turns forget / echo / gibberish / “wrong prompt.”

**Current code** (`src/common/chat_session.cpp`):

```154:155:src/common/chat_session.cpp
        // Fresh KV every turn — residual reuse + full re-template double-prefills.
        kv_cache_ = ctx.new_cache_fn(generate_params_);
```

Every CLI turn **by default** still:

1. Rebuild the complete message list (`build_messages`).
2. Apply chat template to the **full** history.
3. New KV via `new_cache_fn`.
4. Generate.
5. Append user/assistant; **clear** KV.

Production residual (`GenerateParameters.chat_residual` default **true**; `MLX_CHAT_RESIDUAL=0` opt-out) keeps KV only after an exact token prefix (seq-append / lcp-suffix / body-suffix). Compound / rotating / quantized **must** allocate a fresh KV — never full-template onto leftover cache. See [`RESOLUTION.md`](RESOLUTION.md).

This is **correctness over efficiency**. HTTP multi-turn was never the same class: client sends full history; server starts a **fresh** KV per request.

## Secondary roots (independent “prompt processing” symptoms)

| ID | Symptom | Engine root | Fixed in |
|----|---------|-------------|----------|
| I2 | Short / collapsed answers with thinking | CoT shared `max_tokens`; low client defaults truncated the answer | #71 (floor only on `nullopt`; defaults → 4096) |
| I5 | Stop ignored; overrun | `stop` parsed but not applied; multi-id EOS collapsed | #71 + #70 (suffix match + multi-id EOS merge) |
| role:tool | Silent fail / bad Memory/RAG | No rejection of `role:tool` | #71 (explicit 400) |
| Tools + thinking | Tools missing / conflicted | No inject / parse / policy | #62 |
| GDN / quant thrash | Soup / degeneration after history + temp>0 | GDN numerics (g-dtype, softplus, in_proj fuse) | #74 / #76 (opt-in fused2, in_proj skip, stable softplus) |

## What was *not* the source

- HTTP request KV reuse (server does not reuse KV across independent requests).
- A single broken “prompt processing” module.
- The listed PRs (they **closed** the defects).
- Client OpenWebUI Memory/tools injection as the *only* cause (engine bugs were real and independent).

## Residuals (not the original I3 bug)

- Full re-prefill every CLI turn — linear cost with history. **This is now the “too long” source** — see [`RESOLUTION.md`](RESOLUTION.md).
- String-suffix stop sequences (OpenAI-compatible, post-detokenize; stream overshoot possible).
- No multi-turn tool results (`role:tool` still 400 by design).
- Aggressive GDN / quant-fuse paths remain **opt-in**.
- Template fidelity for unusual models.

## Bottom line

Source = incomplete ChatSession KV + template state (I3), **plus** missing server stop/thinking/EOS hygiene, **plus** GDN numerics that only showed once history grew. The PRs fixed those sources; they did not create them.

## Topology note

Do **not** treat `exp/redline-product-own` (forked from `exp/redline-kernel-launch` @ `53d9285`) as this sibling. That branch carries Redline research commits. **This** branch is a clean child of `fix/mtp-stream-p0` only.
