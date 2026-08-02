# MTP research master loop

**Repo:** lemonade-sdk/lemon-mlx-engine  
**Canonical map:** [`BRANCH_MAP.md`](BRANCH_MAP.md)  
**Product PR:** [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) `fix/mtp-product`  
**HARD BANS:** LoopBrake / auto-disable MTP; dual-load; fake TPS; invent numbers without logs.

Program state (high level):

| Area | State |
|------|--------|
| 35B decode micro-opt (C1–C15) | **Plateau ~27 t/s** on gfx1150 890M |
| S4 batch verify + n_draft=3 | **KILL** (`exp/mtp-tps-ceiling`) |
| C11–C15 draft fuses | **Dead** (`exp/mtp-c11-topk-close`) |
| T1 fuse / KV@256 / dense_kept | **Closed** (`exp/mtp-t1-attack`) |
| Quality Maxwell temp0.7 (35B) | Log EXIT:0 post-residual (cite path; not re-run every fire) |
| H2 0.8B formalize | **Docs MET with caveats** (`mtp-h2-small-model/`) |
| Remaining funded | **None on this iGPU field loop** — H1 is **external** dGPU day only |
| Long-ctx KV | **KILL** T1L r2 @~2k prefill (+1.0–1.4% ≪5%) |
| Field scheduler | **STOPPED** (plateau-complete for local levers) |

---

## Fire 2026-08-02T02:23Z — STOPPED (plateau complete + product hygiene)

| Field | Value |
|-------|--------|
| **Result** | **STOPPED** — local field levers exhausted; product claim hygiene landed |
| **Branch** | `exp/mtp-t1-attack` @ tip after this commit |
| **GPU** | use **~6%** idle; **no probe** (nothing fundable) |
| **Product tip** | `fix/mtp-product` @ `777f398` (#77 OPEN, mergeable) |
| **Scheduler** | `scheduler_delete` task **019fc039e578** |

### Clear Thought

- `sequentialthinking` — only residual was #77 hygiene or blocked H1 hardware  
- `metacognitivemonitoring` — do not thrash empty fires on dGPU absence  
- `decisionframework` — hygiene-stop vs keep scheduler thrash → **stop**  
- `collaborativereasoning` (pm/qa) — claim bounds doc; stop loop  

### Reviewed

- Closed: S4, C11–C15, T1 fuse/dense, T1L KV, H2 formalize, H1 protocol notes  
- #77: lean product OPEN; experiment bulk correctly off PR  
- No concurrent chat probe  

### Tested

- **Skipped all probes** — no open local hypothesis; re-litigating kills banned  
- Quality: not re-run; bounds cite existing Maxwell EXIT:0 path  

### Decision

1. **Product hygiene:** `docs/experiments/PRODUCT_CLAIM_BOUNDS.md` (ship-safe vs banned claims + log paths).  
2. **Program:** field research loop on this machine is **plateau-complete**.  
3. **H1** remains a human/hardware-day item (`mtp-h1-dgpu/PROTOCOL.md`), **not** a 2-minute auto fire.  
4. **STOPPED** + delete scheduled task.  

### Next step

- **None for this scheduler.** Human: merge/review #77; optional H1 on dGPU later.  

### Confidence

**0.91** — lever inventory closed; stop avoids empty thrash.

### Supervisor honesty

| Claim | Verdict |
|-------|---------|
| Local measure levers remain | **No** (H1 external only) |
| New TPS this fire | **None** |
| Product hygiene invents wins | **No** — bounds + anti-claims only |

---

## Fire 2026-08-02T02:21Z — PROGRESS (T1L r2 harvest KILL + H1 protocol)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** — long-ctx KV **KILL**; H1 protocol notes landed; not STOPPED |
| **Branch** | `exp/mtp-t1-attack` @ tip after this commit |
| **GPU** | use **3%** at harvest (matrix already complete) |
| **Product tip** | `fix/mtp-product` @ `777f398` (#77) |

### Clear Thought

- `sequentialthinking` — harvest r2 vs ≥5% bar  
- `metacognitivemonitoring` — log-only numbers  
- `scientificmethod` — H-T1L-KV **refuted**  
- `collaborativereasoning` (qa/perf) — park KV; fund H1 hardware later  

### Reviewed

- `T1L_STATUS.txt` complete + three `T1L_eager_*.txt`  
- Single-turn validity: Prompt **2039**, Generation **256** each cell  
- r1 remains VOID under `void_multiturn_r1/`  

### Tested

- **Harvest only** (no new GPU load)  
- Quality: not re-run  

| Cell | gen t/s | Δ vs fuse | Log |
|------|---------|-----------|-----|
| safe_fuse | **28.6272** | baseline | `T1L_eager_safe_fuse.txt` |
| kv8 | **29.0367** | **+1.43%** | `T1L_eager_safe_kv8.txt` |
| kv4 | **28.9138** | **+1.00%** | `T1L_eager_safe_kv4.txt` |

Kill bar ≥5% → need ≥**30.059** t/s; **max 29.037** → **KILL/park**.

### Decision

1. Close long-ctx KV lever on this stack (`LONGCTX_KV.md`, `RESULTS.md`, `BRANCH_MAP`).  
2. Advance one residual: **`docs/experiments/mtp-h1-dgpu/PROTOCOL.md`** (notes only).  
3. Remaining: H1 hardware day, optional #77 hygiene.  
4. Not plateau-STOPPED (H1 still funded).  

### Next step

- H1 dGPU measure when hardware present **or** product cherry-pick hygiene on #77.  
- Do **not** re-open S4/C11/KV without new evidence.  

### Confidence

**0.93** — clean three-cell single-turn logs; arithmetic kill clear.

### Supervisor honesty

| Claim | Verdict | Path |
|-------|---------|------|
| fuse 28.6272 / kv8 29.0367 / kv4 28.9138 | **OK** | `T1L_eager_*.txt` |
| ≥5% long-ctx KV win | **FAIL / KILL** | same |
| r1 multi-turn numbers | **VOID** | `void_multiturn_r1/` |
| 35B ≥100 | **NEVER** | — |

---

## Fire 2026-08-02T02:19Z — PROGRESS (protocol fix; r2 restarted)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** — voided invalid r1; fixed single-turn feed; **r2 matrix restarted** (no TPS verdict) |
| **Branch** | `exp/mtp-t1-attack` @ tip after this commit |
| **GPU at decision** | was **99%** (void r1 thrash) → killed → r2 launched |
| **Product tip** | `fix/mtp-product` @ `777f398` (#77) |

### Clear Thought

- `sequentialthinking` — harvest blocked; diagnose multi-turn; fix protocol  
- `metacognitivemonitoring` — no kill/pass from void logs  
- `scientificmethod` — H-T1L-KV still **testing**; r1 invalidates measure not hypothesis  
- `collaborativereasoning` (qa/perf) — archive void; restart r2; no fake TPS  

### Reviewed

- `T1L_eager_safe_fuse.txt` mid-run: many Prompt/Generation pairs (~82 gen tok), prompt 29→2k+ across turns  
- `examples/chat.cpp` L307–309: `std::getline` = one user message per line  
- Prior fire: measure started multi-line prompt (protocol bug)  

### Tested

- **Did not harvest** r1 for KV bar (VOID)  
- **Killed** void chat/matrix (invalid thrash)  
- **Fixed** `longctx_prompt.txt` to **1 line**; runner collapses newlines  
- **Started r2** serial matrix background (`T1L_STATUS.txt` date 19:19)  
- Quality: not re-run  
- **Skipped** new product probes / S4/C11  

### Decision

1. Document void r1 under `void_multiturn_r1/`.  
2. Protocol fix is the implement step this fire.  
3. Next fire: harvest r2 **only if** three cells have single `Generation:` (prefer 256 tok) and large single `Prompt:`; else continue wait / GPU-busy bail.  
4. Not STOPPED.  

### Next step

- Harvest r2 vs ≥5% kill bar **or** skip if still running.  
- Then H1 dGPU notes or product hygiene.  

### Confidence

**0.80** — root cause certain; r2 outcome unknown.

### Supervisor honesty

| Claim | Verdict |
|-------|---------|
| r1 long-ctx KV numbers | **VOID** — multi-turn artifact |
| r2 complete | **NOT yet** |
| Any ≥5% KV pass/fail | **NOT claimed** |

---

## Fire 2026-08-02T02:15Z — PROGRESS (measure started)

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** — T1 long-ctx KV matrix **started** (serial); **no TPS verdict this fire** |
| **Branch** | `exp/mtp-t1-attack` @ `508442e` |
| **Parent archive** | siblings @ `875a39d` / tip lineage |
| **Product tip** | `fix/mtp-product` @ `777f398` (#77) |
| **GPU at start** | use **2%**, VRAM ~1.3/8 GB; after start use **~16%** (expected load) |

### Clear Thought

- `sequentialthinking` — next funded lever is long-ctx KV (H2 formalize done prior fire)  
- `metacognitivemonitoring` — no invent TPS; results pending harvest  
- `decisionframework` — start serial background matrix; do not poll  
- `scientificmethod` — H-T1L-KV: kv4/8 ≥ +5% gen t/s vs safe fuse at ~2.5k prompt + 256 gen  

### Reviewed

- Prior MASTER_LOOP fire (H2 formalize, S4 closed)  
- `mtp-t1-attack/RESULTS.md` KV flat@256; kill bar ≥5%  
- `examples/chat.cpp` `--kv-bits` wiring  
- GPU idle; `build/chat` present; no concurrent model thrash at decision time  

### Tested

- **Started** (background, serial only): `run_t1_longctx_kv.sh`  
  - Cells: `T1L_eager_safe_fuse` → `T1L_eager_safe_kv8` → `T1L_eager_safe_kv4`  
  - Prompt: `longctx_prompt.txt` (~10k chars ≈ ~2.5k tok filler)  
  - Status: `T1L_STATUS.txt` / `T1L_nohup.out`  
- **Not harvested** this fire (no inline poll)  
- Quality: not re-run (no sampling/emit change)  
- **Skipped:** S4/C11 re-litigation  

### Decision

1. Implement long-ctx protocol docs + runner on `exp/mtp-t1-attack`.  
2. Launch measure in background; next fire **harvests** logs vs ≥5% kill bar.  
3. Do **not** claim KV win/kill until `Generation:` lines exist for all three cells.  
4. Scheduler continues (not STOPPED).  

### Next step

- **Harvest** `T1L_eager_*.txt` + fill `LONGCTX_KV.md` verdict; if matrix still running, **skip new load** (GPU busy bail).  
- After harvest: H1 dGPU notes or product hygiene.  

### Confidence

**0.72** — protocol sound and job launched; outcome unknown; ~2.5k ctx may under-stress bandwidth (note if flat).

### Supervisor honesty

| Claim | Verdict |
|-------|---------|
| Matrix started serially | **OK** — `T1L_STATUS.txt` + `build/chat` PID |
| Long-ctx KV ≥5% win | **NOT claimed** (pending) |
| Any new gen t/s number | **NONE this fire** |

---

## Fire 2026-08-02T02:09Z — PROGRESS

| Field | Value |
|-------|--------|
| **Result** | **PROGRESS** (not PLATEAU-STOPPED) |
| **Branch** | `exp/mtp-t1-attack` @ `199a804` |
| **Parent archive** | `fix/mtp-stream-p0` / siblings @ `875a39d` |
| **Product tip** | `fix/mtp-product` @ `777f398` (#77) |
| **GPU** | ROCm GPU use **2%** at fire start — idle enough, but **no new probe** (docs formalize only) |

### Clear Thought

- `sequentialthinking` — tip on T1; MASTER_LOOP missing; S4/C11 killed; residual H2/long-ctx/H1  
- `metacognitivemonitoring` — honesty pass on T1/S4/H2/Maxwell; do not invent TPS  
- `decisionframework` — pick H2 formalize + map/MASTER hygiene; defer long-ctx KV and H1 hardware  
- `collaborativereasoning` (perf/qa/pm) — consensus PROGRESS not scheduler_delete  

### Reviewed

- `docs/experiments/mtp-t1-attack/RESULTS.md` + `T1_*.txt` (fuse +2.1%/+3.1% eager; KV flat; dense_kept norms)  
- S4 via `git show exp/mtp-tps-ceiling:docs/experiments/mtp-tps-ceiling/RESULTS.md` (batch KILL)  
- H2 logs under `mtp-stream-p0/H2_*.txt` including PASS100 **100.045** and nodebug r1–r5  
- Maxwell `FIELD_MAXWELL_FULL_RS_ndraft2_temp07_think_max8k_post_residual.txt` **EXIT:0**  
- `BRANCH_MAP.md` §7 stale P0 batch/ndraft3 rows  

### Tested

- **Skipped GPU probes** — no open 35B micro-opt hypothesis; re-litigating S4/C11 banned without new evidence; H2 formalize is log-reuse.  
- Quality: **not re-run**; cite existing Maxwell EXIT:0 path only.  

### Decision

1. **Create** this `MASTER_LOOP.md`.  
2. **Advance one lever:** H2 small-model **formalize** → `docs/experiments/mtp-h2-small-model/RESULTS.md` (n≥5 protocol + honest mean≈99.7 vs single-run 100.045).  
3. **Hygiene:** mark S4 batch-verify / ndraft3 / rs-batch **closed** on `BRANCH_MAP.md`; record T1/S4/C11 experiment branches.  
4. **Do not** `scheduler_delete` — residual levers remain (long-ctx KV, H1 dGPU, optional product cherry-picks).  

### Next step (at most one for following fire)

- **Preferred measure:** T1 **long-context KV** retest (kv4/kv8 vs safe fuse) only if single 35B load free and hypothesis still funded (bandwidth-bound ctx).  
- Else: H1 dGPU notes when hardware present; or product hygiene on #77.  
- **Do not** re-open batch verify / C11–C15 / dense_kept.

### Confidence

**0.84** — log-backed formalize; mean≥100 n=5 still slightly short (honest); no new field measure this fire.

### Supervisor honesty (triple-check)

| Claim | Verdict | Path |
|-------|---------|------|
| T1 eager nofuse 28.900 / safe 29.500 | **OK** | `mtp-t1-attack/T1_eager_*.txt` |
| S4 batch 20.890 vs seq 27.216 KILL | **OK** | `exp/mtp-tps-ceiling` RESULTS + S4_*.txt |
| H2 PASS100 100.045 | **OK** | `…/H2_TPS_probe_0p8B_MTP_ndraft2_normshift_PASS100.txt` L433 |
| H2 n=5 mean ≥100 | **NOT claimed** (mean≈99.7) | nodebug r1–r5 |
| Maxwell quality green | **EXIT:0 only** this fire | `…/FIELD_MAXWELL_FULL_RS_…_post_residual.txt` |
| 35B ≥100 | **NEVER** | — |

---

## Stop criteria (for future fires)

- Set **STOPPED** + `scheduler_delete` when: no funded next lever **or** three consecutive fires with no implement/measure.  
- This fire is **not** that condition.
