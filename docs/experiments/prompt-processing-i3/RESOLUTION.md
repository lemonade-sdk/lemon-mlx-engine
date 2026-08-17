# Resolution: ChatSession efficiency without re-breaking I3

**Branch:** `exp/prompt-processing-i3`  
**Parent:** `fix/mtp-stream-p0` @ `875a39d` (sibling of `exp/redline-kernel-launch`)  
**Status:** Phase 0 + gated Phase 1 landed; **default remains full re-prefill**

## Source of “it is taking too long”

After I3 (#64 → #70), `ChatSession` **deliberately** full re-prefills every CLI turn:

- Rebuild entire message history
- Full chat template
- Brand-new KV
- Prefill from scratch
- Clear KV after generate

That killed double-prefill / position mismatch. Cost is **linear (worse with attention) in conversation length**. HTTP is not the same class (each request already carries full history + new KV).

Low-level KV (`update`, `set_position`, `trim`, TokenIterator external-cache) already exists. The gap is **ChatSession state**. Naïve “encode only the new user message” is **unsafe** (Qwen thinking / tools re-serialize history).

## Principles

1. Never apply a full chat template onto residual KV without an **exact token-level prefix**.
2. Default stays full re-prefill.
3. Reuse existing KV / TokenIterator; no new cache primitives.
4. Thinking polarity / tools / system / non-append history → full fallback.

## Phases

| Phase | What | Gate |
|-------|------|------|
| **0** | Turn log: `template_tok`, `prefill_tok`, `prefill_s`, `hist_msgs`, `residual=` | `MLX_CHAT_TURN_LOG=1` (also on when residual=1) |
| **1** | LCP residual: keep `last_templated_tokens_` + KV; accept only if LCP **== last template length** + suffix + same system; `set_position`; suffix prefill | **`MLX_CHAT_RESIDUAL=1`** or `GenerateParameters.chat_residual` — default **off** |
| **2** | Ada-name / rewrite-fallback / rehydrate residual unit tests (stub model) | **landed** on this branch; field matrix (thinking + temp 0.7 + GDN) still **before default ON** |
| **3** | Server prefix cache; incremental templates | orthogonal |

Residual refuses Mamba / non-`is_trimmable` layers (`set_position` is a no-op there).

## Env

```text
MLX_CHAT_TURN_LOG=1     # Phase 0 visibility
MLX_CHAT_RESIDUAL=1     # Phase 1 LCP append-prefill (opt-in)
```

## Not this branch

`exp/redline-product-own` was wrongly forked from the Redline **tip**. This sibling is **only** `fix/mtp-stream-p0` + this ChatSession work.
