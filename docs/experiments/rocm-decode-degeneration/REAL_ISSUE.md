# Real issue — ROCm multi-turn GDN decode collapse

**Not the issue:** LoopBrake early-stop, CI “2+2”, claiming victory because maxphrase≤5 after truncation, or “fused2 off forever as product.”

**The issue:** On ROCm, long multi-turn chat on hybrid Qwen GDN models can stay coherent for several turns then **collapse into endless token/phrase loops** (field: “synchronization synchronization…”) especially when the last turn asks for a **large code generation** (SAR Python).

## System under test (locked)

| Item | Value |
|------|--------|
| Model | **`LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`** |
| Fast bisect only | `mlx-community/Qwen3.5-0.8B-4bit` (not field oracle) |
| Flags | `--max-tokens 20480 --repetition-penalty 1.0 --ctx-size 32768` |
| MTP | **Off** (default skip head; no `--use-mtp`) |
| Pure-graph | **Off** |
| LoopBrake | **Removed** — not a real fix |

## Field prompt sequence (acceptance)

```
tell me about maxwell's equations
tell me about fourier analysis
how about doppler
taken together, what are these good for?
that sounds really interesting, can you give me a python implementation?
```

Optional further turns adding features the model suggests.

**PASS:** coherent answers on turns 1–4; turn 5 produces **usable Python** (or a real plan+code), finishes or hits budget with real content — **no** endless single-token/phrase thrash.  
**FAIL:** starts code then collapses into repeated tokens/phrases; or only “works” via early-stop seatbelt.

## What the code actually does each turn

```
history messages
  → chat template (full re-prefill every turn)
  → Prefill T>1: attention + GDN (conv window, SSM via gated_delta_update*)
  → Decode T=1 loop: GDN step (default: rms_norm + gated_delta_update;
       opt-in: gdn_fused_decode if MLX_GDN_FUSED2=1)
  → logits → sample
  → append assistant text to history
  → drop KV/GDN cache (fresh next turn)
```

**Implication:** Cross-turn GDN cache reuse is **not** the design bug. Collapse = **many T=1 GDN updates** (long answer) and/or **re-prefill of bad prior text**, driven by **numerics/path agreement** (prefill ≡ decode, fused ≡ unfused).

## Hypotheses (ranked)

| ID | Claim | Status |
|----|--------|--------|
| **H-gdn** | Accumulated GDN T=1 numeric/state error (g/beta/norm/fused kernel) | **Primary engineering target** |
| **H-prefill-decode** | Prefill and decode disagree on recurrence math or dtypes | Supported by g-dtype / softplus work; need long-horizon proof |
| **H-async** | One-behind async corrupts state | Ladder: SYNC_DECODE did not clear short ladder; re-test on field 35B |
| **H-model** | Pure template/model self-reinforcement | Possible amplifier; 0.8B thrash ≠ 35B field; not excuse to skip engine proof |
| **H-seatbelt** | LoopBrake / low max_tokens “fixes” it | **Rejected** as product resolution |

## What “resolved” means

1. Field sequence on **LemonMLXE 35B** passes **without** LoopBrake and without requiring weird non-default env just to mute collapse.  
2. Default decode path is **intentionally correct** (if fused is desired for perf, it must match reference math — not stay permanently disabled as a hack).  
3. Prefer evidence: long T=1 vs prefill teacher-force / state checksum / path A/B on collapse turn.

## Loop charter (continuous work)

Every cycle must:

1. Use Clear Thought + domain subagents; quintuple-check.  
2. Work on `fix/rocm-gdn-fused2-optin` (or successor off `main`); **commit + push** (no force).  
3. Drive **field SAR sequence** on LemonMLXE 35B (and 0.8B only for cheap bisect).  
4. If fail: path A/B (`default` / `MLX_GDN_FUSED2=1` / `MLX_GDN_NO_FUSED=1` / `MLX_SYNC_DECODE=1`), log prompts/gens/collapse phrase.  
5. Fix **GDN math/dtypes/kernels**, not generation truncators.  
6. **Never reintroduce LoopBrake.**  
7. Human merge only on PR #74.

## Code hotspots

| Area | Path |
|------|------|
| T=1 GDN select | `src/llm/models/qwen35_moe.cpp` (`use_fused2`, materialize, dtype cast) |
| beta/g, softplus | `src/common/gated_delta.cpp` (`compiled_beta_and_g`, `logaddexp`) |
| fused HIP | `gdn_fused_decode` in `gated_delta.cpp` |
| Multi-turn session | `src/common/chat_session.cpp` (fresh cache, history append) |
| Decode async | `src/common/generate.cpp` (`TokenIterator::next`) |
