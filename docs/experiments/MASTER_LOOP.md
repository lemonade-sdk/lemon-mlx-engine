# MTP research master loop

**Repo:** lemonade-sdk/lemon-mlx-engine  
**Canonical map:** [`BRANCH_MAP.md`](BRANCH_MAP.md)  
**Active field branch (this loop):** `exp/mtp-t1-lmhead-graph`  
**Parent archive:** `fix/mtp-stream-p0` @ `875a39d`  
**Product PR:** [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) `fix/mtp-product`  
**HARD BANS:** LoopBrake / auto-disable MTP; dual-load; fake TPS; invent numbers without logs; re-litigate S4/C11–C15/KV@256/dense_kept without new evidence.

Program state (high level):

| Area | State |
|------|--------|
| S4 batch verify + n_draft=3 | **LEVER2_CLOSED / KILL** (`exp/mtp-tps-ceiling`) |
| C11–C15 draft fuses | **Dead** (`exp/mtp-c11-topk-close`) — do not reopen |
| T1 fuse / KV@256 / dense_kept / long-ctx KV | **Closed** (`exp/mtp-t1-attack`) — do not reopen |
| **Lever 3 lm_head traffic** | **Inventory A done** — head **already 4-bit**; microbench B next (`mtp-t1-lmhead-graph/`) |
| **Lever 4 graph decode 35B** | Pending after #3 B or #3 close |
| Field scheduler | **ACTIVE** on `exp/mtp-t1-lmhead-graph` (new funded loop) |

---

## Fire 2026-08-02T02:29Z — PROGRESS (LEVER2_CLOSED + lm_head inventory A)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** |
| **Branch** | `exp/mtp-t1-lmhead-graph` (created/checked out from `fix/mtp-stream-p0` @ `875a39d`) |
| **GPU** | use **~6%** idle — **docs + header inventory only** (no model gen) |
| **Lever worked** | #2 status confirm + stamp; #3 step **A** inventory |
| **MASTER path** | `docs/experiments/mtp-t1-lmhead-graph/MASTER.md` + `RESULTS.md` |

### Clear Thought

- `sequentialthinking` — ordered levers 2→3A; no re-run S4; no invent ms  
- `metacognitivemonitoring` — BF16 622 MB claim is **speculation** until weight map; S4 numbers are **facts** from sibling logs  
- `decisionframework` — pick **A inventory** over B/C this fire  
- `scientificmethod` — observation stage: package already 4-bit lm_head  

### Reviewed

- `git show exp/mtp-tps-ceiling:docs/experiments/mtp-tps-ceiling/RESULTS.md`  
  - seq n2 **27.216** t/s; batch n2 **20.890** t/s; verify_on_accept mean **77.1 ms** / med **71.2 ms** &gt; **67.7** kill → **KILL**  
  - Logs: `S4_seq_ndraft2.txt` Generation 27.216; `S4_batch_ndraft2.txt` Generation 20.8899  
- Safetensors header of LemonMLXE 35B MTP mlx-4bit snapshot `5f638dff…`  
- Load path: `qwen35_moe.cpp` `call_impl` / `linear_fwd`; `quantize_utils.cpp` register vs embed dequant  

### Tested

- **No GPU probe** this fire (inventory is file header + config).  
- **Did not re-run** S4 / C11 / KV / dense_kept.  
- Quality: not re-run.  

### Decision

1. **LEVER2_CLOSED** — batch-verify stay killed; no product reopen.  
2. **Lever 3A:** document that primary “BF16 lm_head ~622 MB / ~13–14 ms” sketch is **wrong for this package** (vocab 248320; head already 4-bit; store ~286 MB).  
3. **Do not close lever 3 yet** — kill needs microbench B showing head **&lt;5% T₁** or **&lt;5 ms**.  
4. **Do not** design 4-bit conversion as the win (already 4-bit).  
5. Scheduler **continues** (not STOPPED).  

### Insight

On the field 35B MTP mlx-4bit model, **lm_head is already quantized (U32 pack + scales/biases)**; funding “make lm_head 4-bit” is **void**. Remaining question is **whether the 4-bit head is still expensive enough** to justify two-stage / further-cut work — **measure next**.

### Next step (one)

- **Microbench B** on gfx1150 if GPU free: isolated quantized lm_head matmul and/or full-forward vs stop-before-lm_head; ≥3 warm iters; log wall ms + T₁ fraction hypothesis **from logs only**.  
- If GPU busy: docs-only fire, state GPU_BUSY.  
- After B: close lever 3 or design C; then lever 4 inventory.  

### Confidence

**0.90** on inventory dtypes/shapes/bytes (header parse).  
**0.95** on S4 KILL (existing logs).  
**0.0** on residual head ms/% of T₁ (unmeasured).

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| S4 batch KILL 20.89 vs 27.22; verify mean 77.1 ms | **OK** | `exp/mtp-tps-ceiling` RESULTS + S4_*.txt |
| lm_head weight U32 [248320,256] 254279680 B | **OK** | safetensors header |
| lm_head total store 286064640 B | **OK** | sum weight+scales+biases |
| vocab 248320 hidden 2048 | **OK** | config.json text_config |
| BF16 full would be ~1017 MB | **OK** (arithmetic from dims) | RESULTS §1 |
| lm_head wall ms / % of T₁ | **NOT claimed** | needs B |
| +15–25% from quantizing head | **VOID** on this package | already 4-bit |
| Any new gen t/s this fire | **NONE** | — |

---

## Stop criteria

- **STOPPED** + `scheduler_delete` when: lever 3 CLOSED **and** lever 4 KILL/impossible **and** lever 2 already KILL; **or** three consecutive fires with no implement/measure.  
- This fire is **not** that condition.
