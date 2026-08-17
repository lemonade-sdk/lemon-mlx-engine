# Resolution: ChatSession efficiency without re-breaking I3

**Branch:** `exp/prompt-processing-i3`  
**Parent:** `fix/mtp-stream-p0` @ `875a39d` (sibling of `exp/redline-kernel-launch`)  
**Status:** Phase 0–1 + GDN snapshot + Qwen `body-suffix`; **production default is residual ON**. Full re-prefill is fail-closed fallback / `MLX_CHAT_RESIDUAL=0` only.

## Source of “it is taking too long”

After I3 (#64 → #70), `ChatSession` temporarily full re-prefilled every CLI turn to kill double-prefill / position mismatch. That workaround is **not** the product: the cache already *is* the prefix. Production now reuses residual KV and prefills only an exact suffix. Full re-prefill is fail-closed when the new template is not an extension of a stored prefix. HTTP is still one-request-one-prefill unless a server prefix cache exists (Phase 3).

Low-level KV (`update`, `set_position`, `trim`, TokenIterator external-cache) already exists. The gap is **ChatSession state**. Naïve “encode only the new user message” is **unsafe** (Qwen thinking / tools re-serialize history).

## Principles

1. Never apply a full chat template onto residual KV without an **exact token-level prefix**.
2. Default is residual ON. Full re-prefill is fallback / `MLX_CHAT_RESIDUAL=0`.
3. Reuse existing KV / TokenIterator; no new cache primitives.
4. Thinking polarity / tools / system / non-append history → full fallback.

## Phases

| Phase | What | Gate |
|-------|------|------|
| **0** | Turn log: `template_tok`, `prefill_tok`, `prefill_s`, `hist_msgs`, `residual=` | `MLX_CHAT_TURN_LOG=1` (also on when residual=1) |
| **1** | LCP residual: keep `last_templated_tokens_` + KV; accept only if LCP **== last template length** + suffix + same system | **default ON**; `MLX_CHAT_RESIDUAL=0` opt-out |
| **1b** | Pure-Mamba/GDN snapshot restore; `seq-append` if new template extends last template+generated | landed; still opt-in |
| **1c** | Qwen thinking: snapshot at `add_generation_prompt=false` body; restore + `body-suffix` | **field-green on 0.8B and Maxwell 35B GDN** |
| **2** | Ada-name / rewrite / rehydrate / Mamba / hybrid / seq-append / body-suffix / polarity unit tests | **landed** |
| **3** | Server prefix cache; incremental templates | orthogonal |

Qwen thinking templates are **not** an extension of `add_generation_prompt=true` T1 (prior assistant turns drop the `<think>\n` opener). `body-suffix` restores to the `apply(false)` body and prefills only the rewritten assistant + new user + gen-prompt.

Compound / rotating / quantized still refuse. Default is ON.

## Env

```text
MLX_CHAT_TURN_LOG=1     # Phase 0 visibility
MLX_CHAT_RESIDUAL=1     # Phase 1 LCP append-prefill (opt-in)
```

## Not this branch

`exp/redline-product-own` was wrongly forked from the Redline **tip**. This sibling is **only** `fix/mtp-stream-p0` + this ChatSession work.
