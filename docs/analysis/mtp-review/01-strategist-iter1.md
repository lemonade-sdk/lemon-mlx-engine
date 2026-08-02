# 01 — Strategist Analysis (Iter 1): MTP Branch `fix/mtp-stream-p0`

**Role:** planning-analysis-strategist · **Date:** 2026-08-01 · **Method:** full code read of the MTP path + raw-log verification of every TPS claim + clear-thought metacognitive audit.
**Repo:** /home/antmi/lemon-mlx-engine · **Branch:** `fix/mtp-stream-p0` · **Field model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit @ gfx1150.

---

## 1. Executive verdict

**Are there issues with MTP? YES — two P0 correctness defects, four P1 robustness gaps, and a large debt/hygiene tail.**
The branch is an impressively disciplined measurement campaign (60+ probe logs, self-falsifying docs), but the code that shipped carries:
- **P0-A:** the MTP path **silently ignores the sampler** — `--use-mtp` forces greedy decode at any `temperature`/`top_p`. This also **confounds the quality evidence** (the only MTP field SAR at temp 0.7 was actually greedy).
- **P0-B:** the **n_draft>2 draft chain drops MTP self-attention history on the final draft step** (nullptr KV cache) — degraded/wrong γ>1 draft logits, and the **server default is n_draft=3**, so the very first server step (and every adaptive pick of 3) runs the defective path.
- Quality is **not closed**: the sole multi-turn Maxwell SAR under MTP ended `EXIT:143` (terminated), mid-output.
- Test coverage of the speculative state machine is **near zero** (3 shape/map tests; zero accept/reject/rollback/sampling/thread tests).

**TPS headroom (honest):** On gfx1150/35B the single-seq ceiling is **~28–40 t/s**; C7 (27.34, raw-verified) already beats eager (26.13, raw-verified). The realistic software shortlist (draft lm_head cut + accept lift + dense-audit + C15 triage) is worth **~+15–30%** (→ ~31–35 t/s). **100 t/s single-seq on 35B is physics-ruled-out** (speedup ≤ (1+p)/(β+δ) ≤ 2× even free-draft ⇒ ≤ ~52 t/s). 100 is reachable only via H2 (measured: 0.8B MTP 100.045, 5-run confirmed) / H1 (dGPU) / H3 (aggregate batching — architecturally blocked today by the per-model mutex).

---

## 2. Issue register

| ID | Sev | Area | Finding (evidence) | Impact | Recommended action |
|----|-----|------|--------------------|--------|--------------------|
| **A** | **P0** | Sampling & processing correctness | MTP path never calls `sampler_` **or `processor_`**; uses `mx::argmax` at `generate.cpp:794,960,1017,1109,1157,1246`. Both live only in `convert_to_token` (`generate.cpp:385-388`), never called from the MTP section (751–1438). **Iter-2 calibration (reviewer AT-3):** at temp≤0 `AnySampler::from_params` returns `ArgMaxSampler` (`generate.cpp:189-191`) ⇒ **MTP ≡ eager at temp=0** (no divergence); severity confined to temp>0 — but there it is a *total* silent override of `temperature`, `top_p`, **and `repetition_penalty`** (`RepetitionProcessor::did_sample` never fires; the field SAR harness masked this by passing `--repetition-penalty 1.0`). | API contract violation at temp>0/top_p>0/rep-penalty>1 with no warning; the entire temp-0.7 quality evidence base (finding E, quant-fuse thrash history) was collected under inert sampling ⇒ confounded. | Decide product policy: (a) hard-error at temp>0 (or non-neutral processor) + `--use-mtp` — 1–2 d, recommended for v1; (b) proper rejection sampling — 8–12 d follow-up, requires plumbing draft probabilities (the chain currently argmax-drops logits at `generate.cpp:793-797`) + verify-side ratio test + processor callbacks per emitted token. Re-run SAR with the chosen contract actually active. |
| **B** | **P0** | Draft correctness γ>1 | `generate.cpp:785-798`: `mtp_cache = (use_mtp_kv && i < n_draft-1) ? &mtp_caches_[0] : nullptr` — the **final** draft step gets `nullptr`. `MTPDecoderLayer(MoE)::operator()` (`mtp_head.cpp:50-107`, `mtp_moe.cpp`) is full self-attention: `offset=0`, no cache ⇒ the last draft token attends to **no prior draft positions**. Rationale comment addresses *writes* only, ignores *reads*. | Draft logits for d_{n-1} computed without draft-history attention ⇒ wrong distribution ⇒ suppressed accept at γ>1 (consistent with measured p: 0.69@K2 → 0.43@K4 → 0.23@K6, MICROBENCH.md). Server default `--n-draft-tokens 3` (`examples/server.cpp:50`) ⇒ first step runs n=3 (`current_draft_count` returns max when history empty, `generate.cpp:1454-1456`) and hits this. C9's n_draft=3 regression (22.71) may be partly *this bug*, not physics. | Pass the cache on the final step too (read-all, write-when-consumed); add a γ=2 numerical-parity test (draft logits with vs without correct cache); re-measure n_draft=3/4 after fix before declaring deep-draft dead. |
| **C** | **P1** | Memory / lifecycle | `QuantizedWeightRegistry` is a **process-global singleton** keyed by raw `array*` (`quantized_linear.h:31-64`). `model_manager.cpp` has **zero** registry references — unload/LRU eviction never unregisters. MTP adds runtime-quant (`mtp_head.cpp:349`), ckpt-pack (`mtp_head.cpp:293`) and fused-pack (`mtp_moe.cpp:109`, `qwen35_moe.cpp:238`) entries. | (1) Certain leak: scales/biases retained per eviction (long-running multi-model server). (2) Stale keys: if the allocator reuses a freed weight's address for a *dense* weight, `linear_forward` finds stale QuantizationInfo ⇒ `quantized_matmul` on dense data ⇒ garbage/crash. (3) Thread-safety: `find()` on worker forward vs `register_weight` during another model's load — lock topology **UNVERIFIED**; unordered_map concurrent read/write = UB. | Unregister all model-owned pointers on unload (model already knows its weight arrays); add registry `clear()`/scope API; audit lock ordering (manager lock vs container mutex) and add a mutex to the registry or prove single-writer. |
| **D** | **P1** | Test coverage | `tests/test_generate.cpp:796-854`: only 3 MTP tests — weight-map keys, 4-draft-token shape smoke, KV position round-trip. Zero coverage of: accept/reject decision logic, KV rollback/trim (`generate.cpp:1326-1338`), hidden-stash alignment on reject, EOS inside speculative window, adaptive `current_draft_count`, thread/stream binding. | The 690-line speculative state machine (5 verify variants!) can silently regress; every C-number was validated only by field runs. | Golden-vector tests: feed scripted logits through a stub `call_fn`/MTPHead, assert emitted tokens + final cache positions for accept-all / reject-first / reject-mid / EOS-in-draft. CI-runnable on CPU. |
| **E** | **P1** | Quality evidence | `FIELD_SAR_35B_FULLFUSE_MTP_ndraft4_temp07_think.txt` ends `Terminated … EXIT:143` mid-Maxwell; command used `--temperature 0.7 --top-p 0.9` — inert under finding A. `CRITICAL_ANALYSIS.md` admits "quality not fully closed". | The product quality bar for MTP (multi-turn, sampling) is **unmet and partly untestable** with current code; temp-0.7 thrash history (quant-fuse docs) is confounded across MTP/non-MTP runs. | Block MTP default-on behind a real SAR matrix (temp 0 and 0.7-with-working-sampling, n≥3 seeds, 5-turn Maxwell) after A is fixed. |
| **F** | **P1** | Platform parity | `StreamGuard` skips `set_default_stream` under `__APPLE__` (`generate.cpp:122-136`) ⇒ the C4 side-stream draft (`generate.cpp:948`) **silently serializes on Metal**. Zero Metal probe logs exist (all evidence is gfx1150). | C4/C6/C7 perf claims are ROCm-only; Apple users get unknown (probably worse) MTP behavior with no measurement. README advertises Apple as primary target. | Either measure Metal MTP ladder (even one 256-tok probe per config) or document MTP as ROCm-experimental; consider applying the side stream explicitly on Apple (`mx::StreamContext` equivalent). |
| **G** | **P2** | Code debt | **19 MTP env vars** (`grep` census): MLX_MTP_{DEQUANT,KEEP_BF16,NO_NORM_SHIFT,DRAFT_TOPK,QKV_FUSE,NO_QKV_FUSE,PREFETCH,PIPELINE_V1,NO_PIPELINE_V1,NO_PARALLEL_DRAFT,FIXED_DRAFT,BATCH_VERIFY,NO_SHARED,NO_INTERMEDIATES}, MTP_{TIMING,TIMING_SYNC,DEBUG,HEAD,NO_QLMHEAD}. Six gate **measured-regression** features kept default-off in the hot path: PREFETCH (19.42), PIPELINE_V1 (25.84), BATCH_VERIFY, DRAFT_TOPK (26.94), QKV_FUSE (25.45 raw-verified), NO_SHARED (25.60 raw-verified). The prefetch branch duplicates the entire sequential-verify loop (`generate.cpp:1138-1183` vs 1097-1135). | Hot-path complexity, review burden, static-init `getenv` traps (values cached at first call — operator changing env mid-server has no effect, undocumented). | Delete the six regression paths (or move to a `experiments/` branch); keep escapes with product value (KEEP_BF16, NO_NORM_SHIFT, FIXED_DRAFT, TIMING/DEBUG behind NDEBUG). |
| **H** | **P2** | Defaults UX | chat `--n-draft` **default 1** (`examples/chat.cpp:83`) ⇒ `--use-mtp` alone = **behavioral degeneration** (zero speculation; perf tax negligible — `call_impl`'s prenorm+norm split is the same compute as `operator()`, iter-2 AT-8 verified; "scaffolding" help text is stale). Server default **3** (`examples/server.cpp:50`) ⇒ regressed γ=2 config + finding B exposure until adaptive history fills. | Users enabling MTP with defaults get no speculation (chat) or sub-optimal + buggy-path (server). | Ship default 2 on both; remove "scaffolding" wording; log effective n_draft + accept rate in completion_info (already partially there, `generate.cpp:285-289`). |
| **I** | **P2** | Metrics/diagnostics | `mtp_draft_proposed_ += 1` when n_draft<=1 (`generate.cpp:1350`) inflates proposed vs accepted; `record_acceptance(int proposed, …)` ignores `proposed` (1440); `accept_history_` initialized all-1s but unreachable (`n<=0` return at 1456 — dead init); `maybe_log_kv_offset_` never called on the MTP path (only 1573/1580). | Accept-rate telemetry misleading in degenerate cases; KV-offset diagnostics blind under MTP. | Fix counters; drop dead init; call KV-offset logger on MTP path too. |
| **J** | **P2** | Doc integrity | `C15_TPS_probe_ndraft2_device_accept.txt` = **25.33 t/s raw** — regression, **documented in no .md**. CRITICAL_ANALYSIS missing C13/C14/C15 entirely. MICROBENCH β table (β≈1.5–1.7) is pre-C2 **batch**-verify; sequential T=1 β (≈1.0/token by construction) never tabulated — the theoretical section quotes stale β. | Future engineers will reason from stale β and miss that device-accept (C15) already failed. | Update CRITICAL_ANALYSIS/MICROBENCH with C13/C14/C15 rows and a sequential-β section; keep the "one doc, all rows" discipline that made this branch trustworthy. |
| **K** | **P2** | Memory pressure | MTP fuse (`fuse_quant_projections_mtp`) registers the fused pack but never `unregister`s originals nor frees them (members stay resident) ⇒ QKV weight bytes doubled when opt-in. Runtime quant adds ~2.1 GB (19.8→21.9, CRITICAL_ANALYSIS) on a ~22 GB-resident iGPU budget. | VRAM headroom erosion on the memory-constrained field device. | After fuse: unregister + release originals (the registry API already has `unregister` — unused on this path). |
| **L** | **P2** | Cross-model parity *(iter-2 N-2)* | Pre-norm-return + norm-for-logits discipline verified **only** for `qwen35_moe.cpp` (`call_impl` 1073-1085). `qwen3_next.cpp` (+18 lines on this branch) MTP support **UNVERIFIED** for the same invariant; a model that returns pre-norm hidden *and* computes logits from it would emit garbage silently. | Latent correctness risk on the second MTP-capable architecture. | Read-audit `qwen3_next.cpp` call path; add a load-time self-check (logits-with vs without state must match on a 1-token probe). |
| **M** | **P2** | Upstream coupling *(iter-2 N-3)* | `MTPHead::operator()` concat order `[e_norm, h_norm]` justified only by a comment citing upstream `qwen3_5.py:357` (`mtp_head.cpp:171-172`) — file not in repo; upstream reorder ⇒ silent accept collapse. | Fragile cross-repo contract. | Add sanity assertion (e.g., accept smoke on load for known heads) + pinned upstream permalink in the comment. |

---

## 3. Measurement-integrity audit (C-ladder)

| Config | Doc claim | Raw-log check | Verdict |
|--------|-----------|---------------|---------|
| Eager 26.13 | baseline | `TPS_probe_no_mtp.txt`: 26.1317 ✓ | **ROBUST** (single run; no stddev reported) |
| C1 15.87 | draft 157→24ms | `C1_…`: 15.8665 ✓ + microbench draft table | **ROBUST** (mechanism microbenched) |
| C2-seq 19.72 | verify 86→66ms | `C2_seq_…`: 19.7171 ✓ | **ROBUST** |
| C4 20.64 | parallel draft‖verify | `C4_…`: 20.6388 ✓ | **PLAUSIBLE** — single run; timer-slot semantics changed (draft= now includes joint window) ⇒ not comparable timers |
| C6 22.39 | barrier order | `C6_…`: 22.3887 ✓ | **SUSPECT attribution** — accept 0.88 vs C4's 0.62 confound *admitted in doc*; real Δ probably smaller |
| C7 27.34 | skip γ=1 KV + lazy hidden | `C7_…`: 27.3409 ✓ | **ROBUST vs C6** (same prompt, accept 0.85≈0.88) — but vs eager uses different runs; no multi-run stddev on 35B (H2 has 5 runs; 35B has 1) |
| C8–C14 | flat/regress | C13 25.4488 ✓, C14 25.6034 ✓ (raw) | **ROBUST as "not wins"** (single runs suffice to reject) |
| C15 25.33 | *undocumented* | raw ✓ | **UNDOCUMENTED regression** — finding J |
| H2 100.045 | 0.8B + RMSNorm+1 | `…PASS100`: 100.045 ✓; docs: 5 runs 100.0/99.9/99.7… | **ROBUST** (best-evidenced number on the branch; different model) |

Systemic weaknesses: (1) **single-run culture on 35B** — accept rate varies with prompt (0.62→0.88 observed), so ±0.5–1.5 t/s run-to-run noise is plausible; "C8 flat −0.05" is inside noise. (2) **Timer-slot definitions drifted** between configs — the team caught this for C12 ("step timers look faster only because v1 left the timed region"), which implies earlier rows could hide the same artifact. (3) The honest hardware-ceiling analysis (MTP_OPTIMALITY_PLAN §0.2–0.4) is sound and matches measured β/T₁ — **the "100 on 35B" stop bar is correctly declared UNMET**; do not renegotiate via accounting tricks (HARD BAN respected in code & docs — good).

---

## 4. TPS opportunity register

Derivation basis: C7 field — 1.85 tok/step ÷ 67.8 ms/step = 27.3 t/s; joint draft‖v0 ≈ 37.8 ms ≈ T₁ (38.3); accept p≈0.85 @ γ=1; eager T₁=38.3 ms.

| ID | Lever | Mechanism | Expected Δ (derivation) | Conf. | Evidence needed | Cost | Kill-criterion |
|----|-------|-----------|--------------------------|-------|-----------------|------|----------------|
| **T1** | Cheaper draft lm_head (C5) | Vocab-sliced / two-stage top-k head for the draft argmax; keep argmax on device | Draft side of the 37.8ms joint is ~½ (head MoE + full-vocab lm_head). lm_head 151k-vocab matmul ≈ est. 8–15ms ⇒ joint →28–30ms ⇒ step 67.8→~58 ⇒ **~31–33 t/s (+15–22%) — ILLUSTRATIVE (iter-2 D2: the 8–15ms share is asserted, not microbenched)** | **Low-Med** | **lm_head-only microbench FIRST (gate the whole item on it)**; accept must hold | 3–5 d | lm_head < 4ms of joint ⇒ abort |
| **T2** | Accept lift 0.85→0.93 | Fix B (γ>1 cache), keep fc BF16 (measured ~neutral), draft-quality audit | tok/step 1.85→1.93 ⇒ **+4–5%** directly; compounds with T1 ⇒ **~33–35** | Med-High | A/B accept histogram, same prompt, n≥3 runs | 1–2 d (after B) | accept Δ < 0.03 ⇒ abort |
| **T3** | C15 device-accept triage | Device-resident accept comparison avoiding host draft round-trip — **already prototyped, measured 25.33 = regress, undocumented** | Currently negative; triage whether the regression is accounting (like C12) or real | Low | Re-read C15 diff intent vs log; redo with C7 base | 0.5 d | If confirmed real regress ⇒ delete branch code |
| **T4** | Draft dense audit | `dense_kept=7` linears in head (C7 log) — force group alignment / gather_qmm everywhere | 2–5% of draft cost ⇒ **~+1–2%** | Med | Per-linear dtype dump at load | 1 d | All 7 are norms/tiny ⇒ accept as-is |
| **T5** | MTP × pure-graph decode | M6 XORs MTP with `MLX_DECODE_GRAPH_PURE`; unifying would cut T₁ via graph replay | gfx1150: pure was *slower* than eager (comment @1533) ⇒ **~0% here**; dGPU (R9700) potentially large | Low (gfx1150) | One pure+MTP probe | 5–10 d | gfx1150 flat ⇒ defer to H1 workstream |
| **T6** | Quant-fuse default-on | Fewer packs ⇒ bandwidth/mem | Unknown; thrash rate at n≥3 seeds **open** (ACTUAL_ISSUE_ANALYSIS) | Blocked | 3-seed SAR matrix | — | Blocked on quality gate E |
| **T7** | Prefill/TTFT overlap | Chunked prefill under decode | Not a gen-TPS lever | n/a | — | — | Out of scope for TPS bar |
| **H3** | Continuous batching (aggregate ≥100) | N seqs × ~27 t/s | 4 concurrent ⇒ ~108 aggregate | Med | Architecture: `ModelContainer` single mutex (`model_container.h:158-162`), single KV arena, thread-local streams | Large (multi-week) | If product bar stays single-seq Generation: line ⇒ not applicable |
| **H1** | dGPU (R9700 class) | Bandwidth ⇒ eager ≥60–100; MTP rides | Procurement lever | — | One measured probe on documented device | Hardware | — |
| **H2** | Smaller model | **Done & measured**: 0.8B MTP 100.045 (5 runs) | Meets bar on 0.8B | **High** | Already in tree | Done | — |

**Ranked shortlist (gfx1150/35B):** fix **B** → **T2** → **T1** → **T4** → **T3** triage. Realistic destination **~31–35 t/s**; declare software plateau at ~35 and escalate ≥100 to H1/H2/H3. Do **not** re-litigate C8–C14 regressions.

---

## 5. Correctness deep-dives

**5.1 Streams/threads.** P0 fix (StreamGuard + `ensure_thread_cpu_stream_encoders`, `generate.cpp:52-66,119-139`) is sound: TLS CPU encoders re-bound via `cpu::new_stream`, idempotent with `last_n` cache (minor: deleted+recreated streams at equal count would skip rebind — theoretical). `generation_stream()` TLS own-stream default-on for ROCm is correct. Residual risks: (a) Apple no-op (finding F); (b) `static thread_local mx::Stream mtp_draft_stream` (865) leaks per worker thread lifetime (bounded by httplib pool — acceptable); (c) `gpu_set_graph_decode_mode(true)` set 5×/step, reset only by next `prepare()`/`step()` — safe because every request starts with prefill (`generate.cpp:575` sets false), **verified no cross-request leak**.

**5.2 KV/rollback (sequential path).** Traced accept-all / reject-first / reject-mid: cache position advances exactly once per fed token; the substitute `y_` is always the last forward's argmax; `stash_hidden_from(last_st)` keeps hidden↔token alignment (hidden at pos of last fed token, embed of y_ = next token ⇒ MTP recursion consistent). **Sequential path KV logic is CORRECT.** Batch path (default-off): trim via `set_position(trunk_cache_pos+accepted+1)` + mamba `rollback_spec(accepted+1)` looks correct, but quantizes KV *before* trim (wasted work, not corruption) — and is untested (finding D).

**5.3 Norm-shift heuristic.** `pre_fc_norm_hidden` mean < 0.2 ⇒ +1 to **all** dense `*norm*.weight` (`mtp_head.cpp:374-419`). Correct for guru87-style heads (accept 0→0.31, H2). Risks: (a) assumes *every* norm follows the (γ−1) convention — if a future family stores q_norm/k_norm/layers norms as standard γ while only pre_fc_norm is shifted, this **double-shifts** them ⇒ silent accept collapse; (b) threshold 0.2 is magic (35B=0.4937 safe; a legitimately-converted head with mean 0.15 would false-positive); (c) detection runs on the already-loaded (possibly auto-quant-excluded) tensor — OK because norms are never quantized. Mitigation: per-model override + shift-q_norm-only-if-its-mean<0.2 too (per-tensor test instead of global decision).

**5.4 Sampling.** See finding A — the defect is total, not partial: no sampler, no rejection sampling, no temp=0 tie-breaking parity guarantee vs eager path (eager samples at temp=0 too — `AnySampler` — so MTP and eager can even disagree at temp=0 if the sampler does more than argmax). **Highest-credibility gap on the branch.**

**5.5 pending_v1 (C12, default-off).** `finish_pending_v1_` (1407-1438) correctly re-binds stream, stashes hidden, quantizes KV. `next()` finishes it at max_tokens (1495) and before buffer drain (1504). If generation breaks on EOS with v1 pending and the *external cache is reused* (chat_session `take_cache`, `chat_session.cpp:226`), the trunk KV was already advanced by the v1 `call_fn` inside the step — only hidden-stash/quant is skipped; next turn re-prefills from messages (`chat_session.cpp:225` comment) so no corruption — **safe by re-prefill, not by construction**; if anyone later makes turns cache-continuous, this becomes a P0. Document the invariant.

---

## 6. Code-debt inventory

- **Env vars (19):** see finding G. Classification: *product escapes* (KEEP_BF16, NO_NORM_SHIFT, FIXED_DRAFT, LOAD_MTP_HEAD, GEN_OWN_STREAM) keep; *debug* (MTP_DEBUG/TIMING/TIMING_SYNC) keep behind build flag; *dead regressions* (PREFETCH, PIPELINE_V1(+NO_), BATCH_VERIFY, DRAFT_TOPK, QKV_FUSE(+NO_), NO_SHARED, NO_INTERMEDIATES, DEQUANT) delete.
- **Duplicated code:** prefetch verify loop ≈ serial verify loop (55 lines, `generate.cpp:1097-1183`). Batch path (1184-1343) is legacy default-off — candidate for removal after porting mamba-rollback knowledge to a test.
- **`static` getenv caching** throughout ⇒ server operators cannot toggle at runtime; undocumented.
- **`MTPHead` sentinel ctor + optional layers**, **`void*` get_mtp_head_fn** (`model_container.h:40`) type-erasure requiring `static_cast<MTPHead*>` at call sites (generate.cpp:756,832) — typed accessor would remove a class of casts.
- **Instrumentation cost:** MTP_DEBUG fprintf per step interleaves with streamed stdout (visible corruption in FIELD_SAR logs) — route to stderr-only with token boundary guard or disable in field harness.

---

## 7. Open questions for the Program Manager

1. **Sampling policy (A):** implement rejection sampling (real cost, ~1–2 d) or declare MTP greedy-only with a hard error at temp>0? (Product call.)
2. **Server default n_draft 3→2 (H):** ship change now? (Low risk, measured.)
3. **Regression-path deletion (G):** approve removal of 6 opt-in paths + prefetch loop, or freeze behind `MLX_MTP_EXPERIMENTS=1`?
4. **Multi-model server (C):** is LRU eviction + MTP a supported combination? If yes, registry lifecycle is P0-upgrade.
5. **Quality bar (E):** define PASS matrix (turns × temps × seeds) and gate MTP default-on behind it.
6. **Platform claims (F):** is Apple MTP advertised? If yes, fund a Metal measurement day; if no, doc-gate it.
7. **Stop-bar reconciliation:** H2 (0.8B) is MET and documented; 35B is FAIL-with-ceiling. Does the scheduler accept the documented-target wording (MTP_OPTIMALITY_PLAN §0.6 says yes)?
8. **n_draft>2 after B is fixed:** re-open deep draft as a measured option, or freeze γ=1 as product config?

---

## 8. Appendix — evidence index

| Evidence | Location |
|----------|----------|
| Speculative step core | `src/common/generate.cpp:751-1438` |
| next()/drain/pending_v1 | `src/common/generate.cpp:1491-1526,1407-1438` |
| Sampler bypass proof | argmax: 794/960/1017/1109/1157/1246 vs sampler: 385 |
| γ>1 cache-null | `src/common/generate.cpp:785-798`; attention w/ cache `mtp_head.cpp:84-90`, `mtp_moe.cpp` (sdpa + cache->update) |
| Registry singleton | `include/mlx-lm/common/quantized_linear.h:31-64`; absent from `src/common/model_manager.cpp` |
| Stream fix | `generate.cpp:40-66,97-139`; gates `docs/experiments/mtp-stream-p0/gates/RESULTS.md` |
| Norm glue | `mtp_head.cpp:362-424`; 35B mean=0.4937 (`C15_…` log line) |
| Raw TPS | `TPS_probe_no_mtp.txt` 26.1317; `C1` 15.8665; `C2_seq` 19.7171; `C4` 20.6388; `C6` 22.3887; `C7` 27.3409; `C13` 25.4488; `C14` 25.6034; `C15` 25.3275 (all under `docs/experiments/mtp-stream-p0/`) |
| Quality gap | `FIELD_SAR_35B_FULLFUSE_MTP_ndraft4_temp07_think.txt` → `EXIT:143` |
| Defaults | `examples/chat.cpp:83`; `examples/server.cpp:50` |
| Tests | `tests/test_generate.cpp:796-854` |
| Physics | `MTP_OPTIMALITY_PLAN.md §0.2-0.4`; `MICROBENCH.md` β/K table |

*Findings marked UNVERIFIED: registry lock topology; Metal behavior; dense_kept=7 identities; qwen3_next MTP parity (L). Confidence per claim recorded in clear-thought metacognitive audit `mtp-audit-iter1`.*

**Iter-2 reviewer-verified facts (03-quality-review-iter1.md):** (AT-1) `call_impl` applies final norm to logits — pre-norm return does NOT corrupt output (REFUTED the strongest counter-hypothesis); (AT-2) full ladder reproduced from raw logs; (AT-3) temp=0 MTP≡eager via ArgMaxSampler.
