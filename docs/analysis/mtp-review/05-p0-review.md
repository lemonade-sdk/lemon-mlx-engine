# 05 — P0-A / P0-B Code Review (CONVERGED, MTP-only scope)

**Scope:** MTP portions only — sampling contract (P0-A), final-draft KV (P0-B), and the rejection-sampling path that superseded P0-A's hard-error. Explicitly out of scope: prefill HIP-graph F1–F3, non-MTP engine code (except where MTP touches it: sampler, registry).
**Reviewed HEAD:** `df1f199` (+ one untracked field log). **Method:** commit-chain audit + line-level trace of `mtp_speculative_step_sampled` / `mtp_run_draft_chain` / emit protocol / registry lifecycle; Clear Thought MCP (sequentialthinking ×3, metacognitivemonitoring; overall confidence 0.86).
**Process note:** mid-review the branch advanced under us (5 commits in ~90 min); every read was re-verified against the current tree. One pre-compaction finding was **refuted** by re-read (see V-6) — recorded, not silently dropped.

---

## 1. Commit chain under review

| Commit | What | Review verdict |
|--------|------|----------------|
| `38fa6e7` | P0-A greedy-only hard-error + P0-B final-draft KV + n_draft=2 defaults + unit test | P0-B **lands correct**; P0-A **superseded** (below) |
| `eba422d` | Rejection sampling for temp>0; removes hard-error/HTTP 400; chat/server keep operator flags | **Contract upgrade, sound** — strictly dominates the error path |
| `f3c10a8` | Fill `draft_buffer_` on sampled path (the CRITICAL found in review) | **Correct**; regression-locked by golden test |
| `147a319` | Golden emit/accept/adaptive protocol tests (15 cases / 68 asserts) | Good; gaps noted (R-5) |
| `90320a1` | Full Maxwell multi-turn SAR, temp0.7 RS, max_tokens=8192 | Quality bar **MET** (EXIT:0, 5/5 turns) |
| `df1f199` | Leviathan residual max(0,q−p) + quant-registry unload lifecycle | Residual dist **nearly exact** (R-1); registry leak-half fixed (R-2/R-3 remain) |

## 2. Verdicts: P0-A and P0-B

### P0-A — sampling contract — **CLOSED (superseded, better)**
38fa6e7 shipped fail-closed (throw on temp>0/top_p<1/rep≠1 under `--use-mtp`). eba422d replaced it with a working Leviathan rejection-sampling path; the greedy argmax fast path remains at temp=0 (`mtp_uses_greedy_spec`, generate.cpp:201-210; dispatch generate.cpp:1176-1178). The original W1.1 exit criterion ("temp=0.7+`--use-mtp` errors cleanly") is moot — the contract is now *"MTP honors your sampling params"*, proven by full Maxwell SAR at temp 0.7 / top_p 0.9, EXIT:0, 5/5 coherent turns, pre- and post-residual (90320a1 + `..._post_residual.txt`, mean ~25.4 t/s). This also **closes Finding E** (quality bar unmet) at temp>0 with inert-sampling removed.

**Protocol traced correct (file:line):**
- Ratio: `mtp_accept_ratio(log_q, log_p)` = min(1, q/p), NaN-safe (generate.cpp:245-263; golden-tested).
- Alignment: verify step `i` proposes `draft_tokens[i+1]` (generate.cpp:1083-1084) against trunk `log_q` (1085) and draft `log_p = draft_lps[i]` (1086-1089) — `draft_lps[i]` is the logprob recorded when the draft chain sampled `x_{i+1}` (mtp_run_draft_chain:989-992). Same alignment for `draft_logit_rows[i]` (1028-1031, df1f199). ✔
- KV-cache consistency: after reject at `i`, trunk cache holds `draft_tokens[0..i]` == emitted prefix `[d0, x1..xi]`; bonus case holds all `n_draft`; residual/bonus carried as next step's `d0` via `y_`. Traced, consistent. ✔
- Emit protocol: `mtp_make_emit_plan` (225-243) shared by greedy (1716-1723) and sampled (1130-1136); `next()` drains buffer before stepping (1834-1850). The CRITICAL accepted-draft drop is fixed and locked: `TEST_CASE("mtp_make_emit_plan: accept γ=1 → buffer holds d1 (fire-1 regression)")`.
- Repetition penalty: `processor_->process` applied to trunk logits before `log_q` and residual (1072-1081); `did_sample` on accept/reject/bonus (1055-1057, 1097, 1103, 1111). Draft side intentionally penalty-free — consistent with its recorded `p`. ✔
- EOS: per-emitted-token in the outer `generate()` loop (generate.cpp:2029) — buffered EOS stops cleanly. ✔

### P0-B — final-draft KV — **CLOSED (structural)**
`KVCache* mtp_cache = use_mtp_kv ? &mtp_caches_[0] : nullptr;` on **every** chain step including the final one (generate.cpp:980), gated on `use_mtp_kv = (n_draft > 2) && !mtp_caches_.empty()` (953), with `set_position(0)` per chain (954-958). Shared by greedy and sampled paths. The `i < n_draft-1` nullptr starvation is gone. Residual risk is test coverage, not code (R-5).

## 3. Refuted / verified non-issues (do not chase)

| ID | Claim | Outcome |
|----|-------|---------|
| V-6 | "Draft q(x) via full softmax vs TopPSampler truncation → biased ratio" (pre-compaction P1) | **REFUTED** — `TopPSampler::sample_impl` does no top-p filtering ("disabled; falls back to temperature-scaled categorical", generate.cpp:147-162). Draft proposal ≡ recorded logprob. Ratio is correct. |
| V-5 | `mtp_accept_ratio` edge cases | NaN/inf → 0, diff≥0 → 1, clamp ≤1 (245-263); golden-tested. |
| V-7 | EOS under buffered drafts | Outer loop breaks on first emitted EOS; post-EOS buffered tokens never emitted; cache surplus harmless at stop. |
| V-8 | Residual mass-collapse fallback is a seatbelt | Not a HARD-BAN violation: reject implies q(x)<p(x) strictly (q≥p ⇒ ratio=1 ⇒ accept a.s.), so mass=0 only via fp32 underflow; masking the rejected token is a numerically-necessary degenerate handler. |

## 4. Residual register at HEAD (open work, not blockers)

| ID | Sev | Finding | Evidence | Fix sketch |
|----|-----|---------|----------|------------|
| **R-1** | P2 | **Residual double-temperature-scaling.** `mtp_residual_logits` computes r=max(0,q−p) from **temperature-scaled** softmax and returns `log(r+1e-10)`; `sampler_.sample()` then multiplies by 1/t **again** ⇒ samples r^(1/t), not r. At chat default t=0.7 the reject branch is over-sharpened (~r^1.43). Bounded (reject tokens only) but a genuine Leviathan deviation. | generate.cpp df1f199 `mtp_residual_logits` + call site `sampler_.sample(sample_logits)`; CategoricalSampler::sample_impl ×1/t (generate.cpp:169-176) | Sample residual with a bare `mx::random::categorical(log_r)` (already log-domain, already at temp), or return pre-divided logits. Golden test: two-temperature residual histogram parity. |
| **R-2** | P2 | **Registry lock half still open.** Leak fixed (LoadScope + `~ModelContainer` unregister), but `QuantizedWeightRegistry` has **no mutex**: concurrent `unload` (manager thread) + `find` (request thread, another model) = unordered_map UB. AT-5/W2.1 TSAN exit criteria unmet. | quantized_linear.h (registry members; no lock); model_manager.cpp unload | shared_mutex around register/find/erase, or prove-and-document single-model invariant; TSAN cycle test. |
| **R-3** | P2 | **Double-unregister hazard for MTP delta merge.** If `load_mtp_delta_model` reuses base-model arrays also tracked by a separately-loaded base container, first container death erases live registry entries → `find` returns null mid-generation. Ownership semantics untraced. | model_container.h `~ModelContainer` erases tracked ptrs unconditionally | Refcount registry entries, or verify delta merge always allocates fresh arrays (then document). |
| **R-4** | P2 | **Stale help text:** `--repetition-penalty F ... disallowed with --use-mtp` is false since eba422d (rep≠1 routes to RS via `mtp_uses_greedy_spec`). | examples/server.cpp:100 | Rewrite: "routes MTP to rejection-sampling path". |
| **R-5** | P2 | **W1.2 numerical parity test for P0-B never written.** Fix is mechanics-verified only; n_draft=3 remains measured-regressed (22.71) and now also test-uncovered. Golden suite covers helpers, not the draft cache. | tests/test_generate.cpp [mtp] list; plan 02 W1.2 exit criteria | Stub-head parity test: logits(n_draft=3, cache-on-final) vs reference expectation. |
| **R-6** | P2/P3 | **top_p is inert engine-wide** (sampler filter disabled) yet `mtp_uses_greedy_spec` sends temp=0+top_p<1 to the slow serial RS path for argmax-identical output; product-honesty gap beyond MTP. | generate.cpp:147-148 vs 204 | Either implement top_p or drop it from params/help; simplify greedy-spec gate to `temp==0 && rep-ok`. |
| R-7 | P3 | Dead code: `temperature_set`/`top_p_set` (chat.cpp:86-87,123,126 — set, never read); `track_quant_ptr` (model_container.h — no callers). | grep | Delete or wire into logs. |
| R-8 | note | **RS TPS cost −7%** (~25.4 sampled vs 27.34 greedy) exceeds risk-register R1's ≤5% budget. Quality justifies; budget should be **explicitly ratified**, and serial T=1 verify is the obvious reclaim lever (batched verify for sampled). | MASTER_WORKLOG post-residual rows vs C7 | Stakeholder ratify; batched-verify follow-up. |
| R-9 | P3 | No in-step EOS early-exit → wasted trunk verifies after a drafted EOS. | step loops lack eos check | Cheap: break chain on EOS draft. |
| R-10 | P3 | Sampled path field-proven only at n_draft=2; n_draft=3 sampled untested. | field logs all `--n-draft 2` | One probe log or doc-gate. |

## 5. Bottom line

**Zero P0 and zero P1 remain in MTP code at `df1f199`.** P0-A closed *better than specified* (working RS instead of an error gate, quality-proven); P0-B closed structurally with the mid-review CRITICAL (accepted-draft drop) fixed and regression-locked. Six P2 / four P3 tracked above. Branch discipline remains exemplary: every fix shipped with a probe log + worklog row, HARD BAN intact.

**Recommendation:** ship `df1f199` as Fire 1+2 complete. Route **R-1/R-2/R-3 → Fire 4 (robustness)**, **R-5 → Fire 2 test backfill**, **R-4/R-6/R-7 → Fire 4 hygiene**; escalate **R-8** as the risk-register R1 decision it is.

## 6. Confidence & limits (metacognitive record)

Overall 0.86. Residual uncertainty: R-3 depends on `load_mtp_delta_model` array-ownership (untraced); R-2 is UB-by-inspection without TSAN either way; P0-B's accept-rate effect at n_draft=3 is inferred from attention mechanics, not measured (R-5 closes this).
