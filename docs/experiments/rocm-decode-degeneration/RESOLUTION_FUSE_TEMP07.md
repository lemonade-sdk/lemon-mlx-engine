# Resolution: quant-fuse × temp=0.7 Maxwell SAR thrash

**Date:** 2026-07-31  
**Tip:** `710135e` (`fix/rocm-gdn-fused2-optin`)  
**Binary mtime:** `1785425762` (`build/chat`)  
**Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Harness:** Maxwell SAR 5-turn + quit (`thinking ON`, `--max-tokens 20480`, `--ctx-size 32768`)

## Isolation result (fuse vs nofuse @ temp=0.7)

| Cell | Env | temp | Result |
|------|-----|------|--------|
| `FIELD_SAR_35B_FUSE_temp0_think` | `MLX_ENABLE_QUANT_FUSE=1`, fused2 auto-on | 0 | **PASS** `EXIT:0`, 5 gens (2352/2478/2307/2738/5561), max_same=2, python OK |
| `FIELD_SAR_35B_FUSE_temp07_think` | `MLX_ENABLE_QUANT_FUSE=1`, fused2 auto-on | 0.7 | **FAIL** `EXIT:143`, 4 gens then mid-T5 thrash `maxwell`×6357 |
| `FIELD_SAR_35B_NOFUSE_temp07_think` | fuse **unset**, fused2 auto-on | 0.7 | **PASS** `EXIT:0`, 5 gens (2735/3226/2639/2947/5629), max_same=2, python OK (3×\`\`\`python) |

Logs: `docs/experiments/rocm-decode-degeneration/logs/`.

### FUSE@0.7 thrash detail

- Clean turns 1–4 @ ~27 t/s; thrash mid **turn 5 thinking** after prompt ≈3835 tok  
- Onset: `...without maxwell's maxwell maxwell...`  
- Not `f_s_orig`; not LoopBrake; EXIT:143 = external kill after thrash  

### NOFUSE@0.7 control

- Same tip/binary/prompts/fused2-auto; only fuse env differs  
- Survived turn 5 with full python SAR/Doppler/FFT implementation  
- Prompts grew 17 → 1271 → 2811 → 3920 → 5092  

## Ranked causes (evidence)

1. **H-FUSE-SAMP (supported):** `MLX_ENABLE_QUANT_FUSE=1` + multinomial sampling @0.7 → late multi-turn collapse. Temp=0 argmax masks (FUSE@0 PASS). NOFUSE@0.7 PASS isolates fuse.  
2. **H-SAMP-CTX (refuted as sufficient):** sampling alone does not thrash under same harness (NOFUSE PASS).  
3. **H-FUSED2 (not needed / not primary):** both fuse and nofuse cells used fused2 auto-on; nofuse still PASS → fused2 alone does not explain. B2 (`MLX_GDN_NO_FUSED2=1`) **not required**.  
4. **H-GDN-F32 residual (weak):** same GDN path FUSE@temp0 full 5-turn PASS.  
5. **H-KV offset (previously refuted):** no B3.  
6. **Not root:** LoopBrake (forbidden / not engaged).

## Product / engine disposition

- **Quant fuse remains opt-in.** Default path already skips fuse when `MLX_ENABLE_QUANT_FUSE` is unset:

  `src/llm/models/qwen35_moe.cpp` — `fuse_quant_projections()` early-return if env unset.

- **Do not enable `MLX_ENABLE_QUANT_FUSE=1` for production multi-turn sampling** on this model/stack until numeric fix lands. Safe for temp=0 / short TPS benches (field PASS).  
- **No production code change in this resolution.** Reasons:
  - Default is already safe (fuse off).  
  - A “fix” would be speculative (clamp/logits/fuse shape) without a unit-level numeric repro.  
  - LoopBrake and default-flips of fused2 are out of scope / would mask.  

## Optional human follow-ups (not auto-justified)

1. Numeric A/B of fused vs unfused QKV/gate matmul outputs (layer0 / late layer) under long decode.  
2. If fuse must be default-on for TPS: find and fix exact bias in fused quant concat+matmul path; re-run FUSE@0.7 SAR think cell + temp0 regression.  
3. Document TPS opt-in only in operator guides (already partially covered by opt-in env).

## Bisect cells not run (and why)

- **B2** FUSE + `MLX_GDN_NO_FUSED2=1`: unnecessary after NOFUSE PASS.  
- **B3** `MLX_KV_OFFSET_LOG`: H-KV already refuted; fuse isolated.

## DONE criteria

- (a) fuse vs nofuse @0.7 both scored → **yes**  
- (b) resolution plan written; no further engine code justified without human → **yes**  
