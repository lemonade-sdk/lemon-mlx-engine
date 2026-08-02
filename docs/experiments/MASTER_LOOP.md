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
| Remaining funded | **Long-ctx KV r2 IN FLIGHT** (r1 VOID multi-turn); H1 dGPU; product hygiene |

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
