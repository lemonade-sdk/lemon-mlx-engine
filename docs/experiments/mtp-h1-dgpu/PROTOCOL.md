# H1 — dGPU day protocol (notes only; no measure this fire)

**Status:** **FUNDED residual** — needs discrete GPU hardware; not runnable on gfx1150 890M field machine alone.  
**Parent story:** 35B single-seq MTP/eager plateau ~27–29 t/s on 8 CU iGPU; stop bar ≥100 needs H1/H2/H3 (`mtp-stream-p0/MTP_OPTIMALITY_PLAN.md`).  
**H2** already documents 0.8B path (`mtp-h2-small-model/`). This file is the **H1 measurement checklist** for when a launch-bound dGPU is available.

## Hypothesis

On a discrete GPU that is more **launch-/memory-bandwidth-bound** than 890M, MTP sequential verify (or residual batch) may show a **larger relative win vs eager** than on iGPU, and absolute gen t/s may clear product bars without fake TPS.

## HARD BANS (unchanged)

- No LoopBrake / auto-disable MTP  
- No dual-load of two 35B processes  
- No accept-rate-only TPS; wall-clock **`Generation:`** line only  
- No inventing numbers without logs under this tree  

## Minimum A/B matrix (one model session family)

| Cell | Config | Log name |
|------|--------|----------|
| H1_eager | no MTP, SAFE quant fuse | `H1_eager_safe.txt` |
| H1_mtp_seq | `--use-mtp --n-draft 2`, seq verify default | `H1_mtp_ndraft2_seq.txt` |
| H1_mtp_batch | only if product still ships flag | `H1_mtp_ndraft2_batch.txt` (expect KILL class unless device changes S4) |

Shared: Fourier-style prompt, **256** gen, temp=0, `--ignore-eos`, `--no-think`, `MTP_TIMING=1` optional for `[mtp-t]`.  
Record: GPU name, VRAM, driver/ROCm, model id, SHA, **Prompt:** + **Generation:** lines.

## Pass / kill (pre-commit)

| Bar | Decision |
|-----|----------|
| MTP gen t/s ≥ eager × **1.10** (same session) | Fund product “MTP wins on dGPU” claim with logs |
| MTP ≈ eager (±5%) | Document parity; MTP is quality/emit path not t/s product on that GPU |
| Batch verify still ≪ seq | Keep S4 KILL; do not reopen batch WS |
| Absolute ≥100 on 35B | Only if log shows it — never extrapolate from iGPU |

## Out of scope here

- Re-running C11–C15 draft fuses  
- Claiming iGPU 35B ≥100  
- Shipping H1 code without field logs  

## Related

- iGPU ceiling: `exp/mtp-tps-ceiling`, `exp/mtp-t1-attack`  
- Master loop: `../MASTER_LOOP.md`  
- Product PR: #77 `fix/mtp-product`
