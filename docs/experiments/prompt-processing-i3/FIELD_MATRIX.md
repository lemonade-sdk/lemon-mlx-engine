# Field matrix — Maxwell 5-turn (thinking ON, temp 0.7, GDN 35B)

**TS:** 20260816-193941  
**Branch:** `exp/prompt-processing-i3` @ `c1189b2`  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit (GDN / Qwen3.6-35B)  
**Prompts:** `docs/experiments/mtp-stream-p0/maxwell_full_prompts.txt` (T1–T5 Maxwell)  
**Budget:** `--temperature 0.7 --max-tokens 384` thinking **ON** (not 8192; this is I3/quality gate, not 8k TPS SAR)  
**Bar:** rc=0, 5 Assistant turns, no thrash (`maxwell`×thousands), no EXIT 143

## Results

| Cell | Residual | rc | Turns | T5 prefill tok | T5 prefill s | residual= log | Quality |
|------|----------|---:|------:|---------------:|-------------:|---------------|---------|
| **A** | OFF (default) | **0** | **5** | 1688 | 13.18 | all `full` | 5× thinking, maxwell count=7, no thrash |
| **B** | ON (`MLX_CHAT_RESIDUAL=1`) | **0** | **5** | 1689 | 13.42 | all `full` (not `lcp-suffix`) | same shape; fail-closed |

Prefill grows ~37 → 449 → 862 → 1275 → 1688. That **is** the “taking too long” cost of full re-prefill.

Cell B never logged `lcp-suffix`: GDN/Mamba residual **refuses and stays full**. No I3 suffix-on-stale-KV.

## Decision

| Question | Answer |
|----------|--------|
| Default Maxwell field-green at this budget? | **PASS** (quality bar MET for crash/thrash/5 turns) |
| Residual help on Maxwell/GDN? | **NO** — still full prefill |
| Promote residual to default? | **NO** |
| Keep residual opt-in? | **YES** for transformer-only; **off** for GDN |
| Move forward to default-ON residual? | **DO NOT** |

Logs: `logs/field-maxwell-A-default-20260816-193941.*` · `logs/field-maxwell-B-residual-20260816-193941.*`
