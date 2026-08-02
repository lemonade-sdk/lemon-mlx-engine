# T1 long-context KV retest

**Branch:** `exp/mtp-t1-attack`  
**Hypothesis (H-T1L-KV):** With long prefill (~2k+ prompt tokens), eager `--kv-bits 4|8` raises wall-clock **Generation** t/s by **≥5%** vs safe quant-fuse baseline (same session).  
**Why:** At 256-tok short decode, KV was **flat** (`RESULTS.md` §1). Residual story is bandwidth-bound only if cache is large.

## Protocol

| Item | Value |
|------|--------|
| Model | `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit` |
| Device | gfx1150 / 890M |
| Env | `MLX_ENABLE_QUANT_FUSE=1` (SAFE; **no** GDN in_proj) · `MLX_LOAD_MTP_HEAD=1` |
| Path | **Eager only** (no `--use-mtp`) — T₁ credit only |
| Prompt | `longctx_prompt.txt` — **must be one physical line** (~10k chars ≈ ~2.5k tok); `chat.cpp` uses `std::getline` |
| Gen | `--max-tokens 256 --temperature 0 --ignore-eos --no-think` |
| Cells | `T1L_eager_safe_fuse`, `T1L_eager_safe_kv8`, `T1L_eager_safe_kv4` |
| Runner | `run_t1_longctx_kv.sh` (serial; HARD BAN dual-load; collapses newlines) |
| Status file | `T1L_STATUS.txt` |

## VOID run r1 (2026-08-02) — multi-turn stdin bug

| Item | Detail |
|------|--------|
| Symptom | Each newline → separate chat turn; many ~82-tok gens; prompt grew ~29→2k+ across turns |
| Root cause | `examples/chat.cpp` `std::getline(std::cin, input)` + multi-line `longctx_prompt.txt` |
| Action | Job **killed**; logs under `void_multiturn_r1/`; **do not** use for KV kill bar |
| Fix | Single-line prompt + runner `tr '\n' ' '` before feed |

## Kill / pass

| Outcome | Decision |
|---------|----------|
| max(kv4, kv8) ≥ baseline × **1.05** gen t/s | **PASS** — fund product KV for long-ctx; document delta with logs |
| Neither ≥5% | **KILL / park** long-ctx KV on this stack; product stays default full KV |
| Load OOM / hang | **BLOCKED** — note in MASTER_LOOP; do not invent TPS |

**Honesty:** Within-session deltas only. Do not compare to short-ctx T1_*.txt absolutes as “win.”

## Results — r2 (valid single-turn) 2026-08-01 local / fire harvest 2026-08-02

Session: `T1L_STATUS.txt` complete `2026-08-01T19:20:28-07:00` · tip at run `419c428` · model 35B-A3B-MTP-mlx-4bit gfx1150.

| Cell | Prompt tok | gen tok | gen t/s | Δ vs fuse | Log |
|------|------------|---------|---------|-----------|-----|
| T1L_eager_safe_fuse | **2039** | **256** | **28.6272** | baseline | `T1L_eager_safe_fuse.txt` |
| T1L_eager_safe_kv8 | **2039** | **256** | **29.0367** | **+1.43%** | `T1L_eager_safe_kv8.txt` |
| T1L_eager_safe_kv4 | **2039** | **256** | **28.9138** | **+1.00%** | `T1L_eager_safe_kv4.txt` |

Pass threshold: baseline × 1.05 = **30.059** t/s.  
max(kv4, kv8) = **29.037** ≪ 30.059.

**Verdict:** **KILL / park** long-ctx KV on this stack at ~2k prefill. H-T1L-KV **refuted** under pre-committed ≥5% bar.  
Do **not** productize `--kv-bits` as a decode t/s win on gfx1150 35B from this evidence.  
Optional future: only if new evidence (e.g. dGPU bandwidth-bound 8k+) — not auto-reopen.

## Related

- Short-ctx flat: `RESULTS.md` §1–3  
- Master loop: `../MASTER_LOOP.md`
