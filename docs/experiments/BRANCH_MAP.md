# lemonade-sdk/lemon-mlx-engine — Complete Branch Map

**Repo:** https://github.com/lemonade-sdk/lemon-mlx-engine  
**As of:** 2026-08-02  
**Working tip (this document lives on):** `fix/mtp-stream-p0` @ `6a59066`  
**Canonical product PR for MTP stream work:** [#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77) (`fix/mtp-product`)  
**Method:** full `git branch -r` / local inventory + open/merged PR scan + Clear Thought (sequentialthinking, first-principles, decisionframework, visualreasoning, metacognitivemonitoring).

---

## 0. Purpose & rules

| Rule | Meaning |
|------|---------|
| **Target** | All PRs go to **lemonade-sdk/lemon-mlx-engine** (`origin`), base **`main`**. |
| **Product** | `fix/*` or `feat/*` — reviewable code, lean docs, open PR. |
| **Experiment** | `exp/*` — measured programs, probe logs, kill criteria; **no mega-PR of .txt bulk**. |
| **Archive tip** | `fix/mtp-stream-p0` ≡ full history + all C-ladder probes (alias `exp/mtp-stream-full`). |
| **Historical** | `split/*`, `merge/*` — #63-era stacks; do not reopen. |
| **HARD BAN** | No LoopBrake / auto-disable MTP / dual-load / fake TPS. |

### Naming

```
fix/<topic>     → product fix, expected PR
feat/<topic>    → product feature, expected PR
exp/<topic>     → experiment archive or active field program
docs/<topic>    → docs-only PR
split/<…>       → historical PR stack slices (dead)
merge/prN-…     → historical merge-bridge branches
geramy/|ochafik/ → collaborator remotes (not our day-to-day tip)
```

### Topology (mental model)

```
origin/main  ──┬── fix/mtp-product          (#77 OPEN)     lean product
               ├── fix/quant-fuse-selective-gdn (#76 OPEN)
               ├── exp/prefill-hip-graph    (docs F1–F3)
               ├── fix/mtp-stream-p0        FULL tip + probes  ──► exp/mtp-stream-full (same SHA)
               └── [potential] exp/mtp-batch-verify-reprobe, exp/mtp-ndraft3-p0b, …
```

---

## 1. Active product branches (open PRs)

| Branch | Tip (short) | PR | Base | Role | Status |
|--------|-------------|-----|------|------|--------|
| **`fix/mtp-product`** | `777f398` | **[#77](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/77)** | `main` | ROCm StreamGuard, RS @ temp0.7, P0 gates, registry lifecycle, selective fuse, goldens, Maxwell PASS (lean) | **OPEN — primary MTP product** |
| **`fix/quant-fuse-selective-gdn`** | `4634f2b` | **[#76](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/76)** | `main` | Quant-fuse opt-in + GDN `in_proj` skip; no probe bulk | **OPEN** — #77 **overlaps** fuse policy + adds registry |
| **`merge/pr67-into-main`** | (remote) | **[#73](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/73)** | `main` | CI dual-lane simple-math smoke | **OPEN** |
| **`docs/agents-rocm-env`** | (remote) | **[#69](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/69)** | `main` | AGENTS.md ROCm gfx1151 env | **OPEN** |
| **`geramy/enable_more_archs`** | (remote) | **[#49](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/49)** | `main` | CI fatbin RDNA arches | **OPEN** (collaborator) |
| fork `bong-water…:setup/pr-agent` | — | **[#48](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/48)** | `main` | PR-Agent CI | **OPEN** (external) |
| fork `bong-water…:main` | — | **[#47](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/47)** | `main` | Alternate MTP delta/probabilistic acceptance + MLX sync | **OPEN** (external — **not** our product tip) |

**Merge order preference:** #76 (fuse) → rebase/restack #77 if maintainers want smaller MTP diff; else #77 is self-contained.

---

## 2. Experiment / archive branches (ours, exist on origin)

| Branch | Tip | vs `origin/main` | Role | Product PR? |
|--------|-----|------------------|------|-------------|
| **`fix/mtp-stream-p0`** | `6a59066` | **+45** (diverged base) | **Full WIP tip:** StreamGuard → C1–C15 ladder → RS → residuals → Maxwell → review docs; **all probe .txt** | No — use #77 for review |
| **`exp/mtp-stream-full`** | `6a59066` (same as above) | +45 | Explicit **experiment archive alias** of full tip | No |
| **`exp/prefill-hip-graph`** | `b99cb2f` | +1 | Prefill F1–F3 docs/A-B only (missed ≥10% pp/s bar); mlx `use_hip_graphs` opt-in patch note | Optional draft docs PR only |
| **`exp/mtp-tps-ceiling`** | child of `875a39d` | S4 | Batch-verify re-probe — **KILL**; plateau ~27 t/s | evidence only |
| **`exp/mtp-c11-topk-close`** | **sibling** of S4 / parent `875a39d` | hygiene | C11 top_k **closed** (three-way kill); R-11 comment + static env | optional tiny cherry-pick |

### Experiment doc trees on full tip

| Path | Content |
|------|---------|
| `docs/experiments/mtp-stream-p0/` | MASTER_WORKLOG, CRITICAL_ANALYSIS, C*/H* probes, Maxwell SAR, P0 gates |
| `docs/experiments/prefill-hip-graph/` | F1–F3 RESULTS, PREFILL_ARENA_DESIGN, mlx patch |
| `docs/experiments/rocm-decode-degeneration/` | Fuse thrash isolation notes/logs (partial on tip) |
| `docs/analysis/mtp-review/` | Review series 01–06 (+ this map pointer as 07) |

---

## 3. Related product lineage (merged or closed — keep for archaeology)

### Merged (still may exist as remote branches)

| Branch | PR | Outcome | What it was |
|--------|-----|---------|-------------|
| `fix/rocm-gdn-fused2-optin` | [#74](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/74) | **MERGED** | GDN decode degeneration; fused2 opt-in |
| `merge/pr66-into-main` | [#72](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/72) | **MERGED** | Load hygiene (MTP head skip, quant fuse, GDN) from #66 |
| `merge/pr65-into-main` | [#71](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/71) | **MERGED** | Server stop/thinking/tool 400 |
| `merge/pr64-into-main` | [#70](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/70) | **MERGED** | ChatSession re-prefill + EOS |
| `split/pr63-01-chatsession-eos-pure` | [#64](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/64) | **MERGED** | First #63 split |
| `feat/openai-tools-server` | [#62](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/62) | **MERGED** | OpenAI tools + thinking policy |
| `fix/rocm-hip-arch-gfx1150` | [#61](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/61) | **MERGED** | HIP fatbin RDNA3/3.5/4 |
| `fix/mtp-cleanup-merged` | [#35](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/35) | **MERGED** | MTP MoE + benchmarks (earlier generation) |
| `fix/repetition-processor-scatter` | [#33](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/33) | **MERGED** | Scatter axis |
| `fix/qwen3-smoke-test-reliability` | [#29](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/29) | **MERGED** | CI smoke |

### Closed without merge (superseded)

| Branch | PR | Note |
|--------|-----|------|
| `fix/eager-no-mtp-correctness` | [#63](https://github.com/lemonade-sdk/lemon-mlx-engine/pull/63) CLOSED | Split into #64–#68 / merge bridges |
| `split/pr63-02` … `05` | #65–#68 CLOSED | Superseded by merge/pr* |
| `docs/resolution-fuse-temp07` | #75 CLOSED | Isolation docs; fuse policy → #76 |
| `fix/mtp-review-cleanup` | #34 CLOSED | Historical MTP port cleanup |

Local still present (stale tracking OK): `fix/eager-no-mtp-correctness`, `fix/mtp-cleanup-merged`, `fix/rocm-gdn-fused2-optin`, `fix/rocm-hip-arch-gfx1150`, `feat/openai-tools-server`, all `split/*` and `merge/pr64|65|66*`.

---

## 4. Historical #63 split stack (do not reopen)

Two parallel naming series exist on origin — **both are dead stacks**.

| Series | Branches |
|--------|----------|
| `split/63-0N-*` | `01-chatsession-eos-pure` … `05-docs-ops` |
| `split/pr63-0N-*` | same five topics (PR #64–#68 era) |
| `merge/pr6N-into-main` | `pr64`…`pr67` bridges |

Local-only: `pr-63-tmp` @ `7e5b605` (scratch; not on origin).

---

## 5. Collaborator & adjacent remotes (related, not our tip)

### Prefill / graph (geramy) — relevant to prefill experiments

| Branch | Why related |
|--------|-------------|
| `geramy/graph-prefill` | Prefill graph workstream |
| `geramy/graph_replay_fix` | Graph replay |
| `geramy/diagnose-prefill-ops` | Prefill op diagnosis |
| `geramy/wmma_prefill_grouping_opti` | WMMA prefill grouping |
| `geramy/optimizations` | General opts |
| `geramy/update-release` | Release plumbing |
| `geramy/enable_more_archs` | Open PR #49 |

### Historical MTP ports

| Branch | Note |
|--------|------|
| `ochafik/qwen35-mtp-port` | Early Qwen3.5 MTP port |
| `feat/mtp-port` | Engine MTP port + TPS logging era |
| `feat/phase-memory-pools` | Memory pools (adjacent perf) |
| `feat/fastokens-tokenizer` | Tokenizer (unrelated to MTP decode) |

### ROCm packaging / CI remotes

| Branch | Note |
|--------|------|
| `rocm-7.2.1-apt`, `rocm-preview-stable-apt`, `rocm-preview-stable-tar`, `rocm-preview-stable-tar1` | ROCm install flavors |
| `fix/rocm-ccl-dnn-dev`, `fix/rocm-core-dev-package-check`, `fix/rocm-runtime-dev-deps`, `fix/rocm-tarball-hybrid` | Package/CI deps |
| `fix/mlx-pin-gfx1152-hipblaslt` | hipBLASLt pin |
| `fix/actions-queue-ubuntu-latest`, `fix/prepare-matrix-self-hosted` | CI matrix |
| `fix/cancel-decode-on-client-disconnect` | Server cancel |
| `fix/qwen35-default-enable-thinking`, `fix/qwen36-vlm-quant-prefix` | Model defaults |

Treat these as **do not merge into mtp-stream-p0 blindly**; open separate product PRs if cherry-picking.

---

## 6. Local-only branches & stashes

| Ref | Tip / id | Action |
|-----|----------|--------|
| `docs/local-fuse-temp07-resolution` | `388fb7b` | Local docs polish; #75 closed — **archive or delete when safe** |
| `pr-63-tmp` | `7e5b605` | Scratch; safe to delete after confirming no unique commits |
| `main` (local) | **behind origin/main by ~61** | Run `git fetch && git checkout main && git reset --hard origin/main` when refreshing |
| **stash@{0}** `prefill-arena-wip-isolate-from-mtp-p0` | PrefillArena + graph_decode fixed buffers (~331 lines) | Promote to **`exp/prefill-arena`** (see §7) — do not drop on product tip without isolation |
| stash@{1}…@{4} | older WIP | Review before `stash drop` |

---

## 7. POTENTIAL experiment branches (not created yet)

Create with: `git checkout -b exp/<name> origin/main` **or** from `fix/mtp-stream-p0` when the experiment needs the full stack. Always put field logs under `docs/experiments/<name>/`.

| Proposed branch | Create-from | Hypothesis | Kill / exit criteria | Priority |
|-----------------|-------------|------------|----------------------|----------|
| **`exp/mtp-batch-verify-reprobe`** | `fix/mtp-stream-p0` | Post-fuse stack may amortize batch T=2 verify (C1-era was 86 ms = 2.26×T₁) | `MLX_MTP_BATCH_VERIFY=1`, n≥3 pinned Fourier; **kill if T₂ > 67.7 ms**; reopen WS if ≤60 ms | **P0 — next probe day** |
| **`exp/mtp-ndraft3-p0b`** | `fix/mtp-stream-p0` | Pre-P0-B n_draft=3 22.71 t/s is **invalid** (final-draft KV starve) | Measure n_draft=3 greedy + RS after P0-B; kill deep-draft if still < C7 | **P0 — same day as batch** |
| **`exp/mtp-rs-batch-verify`** | after batch probe if live | RS −7% vs greedy is serial T=1 structure; batch verify reclaims | Only if batch probe reopens WS; else N/A | P1 |
| **`exp/mtp-h1-dgpu`** | product or tip | MTP relative win scales on launch-bound dGPU vs 890M 8 CU | Hardware day; A/B eager vs MTP batch/seq; decide product surface | **P1 strategic** |
| **`exp/mtp-h2-small-model`** | tip (H2 logs already exist) | 0.8B ~100 t/s is the real MTP home if 35B plateaus | Formalize n≥5 protocol, document product claim | P2 (docs formalize) |
| **`exp/mtp-h3-batching`** | main | Multi-stream throughput ≠ single-stream t/s | Server concurrency matrix; do not credit as “gen t/s” | P2 |
| **`exp/mtp-dense-kept-audit`** | main or tip | dense_kept / T₁ attack lifts **eager and MTP equally** | ≥10% T₁ cut or kill; **do not credit MTP** | P2 |
| **`exp/mtp-kv-quant`** | main | KV quant lowers bandwidth | Quality bar + t/s; same “don’t credit MTP-only” | P3 |
| **`exp/mtp-tsan-registry`** | `fix/mtp-product` or tip | Registry mutex/refcount under unload race | TSAN clean load/unload×generate; or document single-model invariant | P2 robustness |
| **`exp/prefill-arena`** | **stash@{0}** → new branch off tip | PrefillArena fixed-address `[1,T]` reduces OOM/hang history | Isolate from MTP product; kill if hang/OOM returns or <10% pp/s | P2 (WIP ready) |
| **`exp/mlx-hip-graph-upstream`** | **mlx tree**, not engine | Local `use_hip_graphs` opt-in patch (see prefill docs) | PR to MLX/ROCm host only if maintainers want; engine keeps env gate | P3 external |
| **`exp/mtp-c-ladder-archive`** | already inside tip | Optional thin branch with **only** C11–C15 REGRESS docs for citation | Optional; tip already holds probes | P3 optional |
| **`exp/mtp-maxwell-multiseed`** | tip | Multi-seed Maxwell for quality CI claim | N seeds PASS; no TPS game | P3 |
| **`docs/mtp-branch-map-sync`** | — | If map needs docs-only PR without code | Prefer keep map on tip + link from #77 | optional |

### Explicitly **not** potential (banned / dead)

| Idea | Why not a branch |
|------|------------------|
| LoopBrake / auto-disable MTP | HARD BAN |
| Dual-load base+MTP containers | HARD BAN / SEGV history |
| More C11–C15-class draft micro-opts without batch-verify reopen | Ceiling proof: draft not on critical path under seq verify |
| Fake accept-rate TPS without wall-clock | HARD BAN |
| Reopen mega `fix/mtp-stream-p0` as sole PR | Use #77 lean product |

---

## 8. How full tip maps to product PR

| Concern | Full tip `fix/mtp-stream-p0` | Lean `fix/mtp-product` (#77) |
|---------|-------------------------------|--------------------------------|
| StreamGuard / server TLS | yes | yes |
| C1–C15 code that stayed (seq verify, C7, adaptive, …) | yes | yes (as current code) |
| RS + residual + draft_buffer + registry | yes | yes |
| Selective quant-fuse | yes | yes (overlap #76) |
| F2 `MLX_PREFILL_ONE_GRAPH` | yes | yes (opt-in) |
| All C*/H* probe .txt | **yes** | **no** |
| Prefill F1–F3 tree | yes | no (→ `exp/prefill-hip-graph`) |
| mtp-review 01–06 | yes | yes (included lean) |
| This BRANCH_MAP | **yes (canonical)** | link only if desired |

---

## 9. Recommended workflow (lemonade-sdk)

1. **Daily product work** for review: branch from `origin/main` or restack on open PR base; open/update **#77** / **#76**.  
2. **Field experiments:**  
   `git checkout fix/mtp-stream-p0 && git checkout -b exp/<topic>`  
   log under `docs/experiments/<topic>/`, push `exp/<topic>`, update this map.  
3. **Never** force product reviewers to digest 20k lines of probes — archive on `exp/*`.  
4. **After experiment kill/pass:** one-line row in this file + MASTER_WORKLOG; promote code only via `fix/*` PR.  
5. **Stash promotion:** `git checkout -b exp/prefill-arena fix/mtp-stream-p0 && git stash apply stash@{0}` then isolate.

---

## 10. SHA snapshot (2026-08-02)

| Ref | SHA |
|-----|-----|
| `origin/main` | `a63692e` (PR #74 merge) |
| `origin/fix/mtp-stream-p0` | `6a59066` |
| `origin/exp/mtp-stream-full` | `6a59066` |
| `origin/fix/mtp-product` | `777f398` |
| `origin/exp/prefill-hip-graph` | `b99cb2f` |
| `origin/fix/quant-fuse-selective-gdn` | `4634f2b` |

Refresh with:

```bash
git fetch origin --prune
git log -1 --oneline origin/main origin/fix/mtp-stream-p0 origin/fix/mtp-product
gh pr list --repo lemonade-sdk/lemon-mlx-engine --state open
```

---

## 11. Related docs

| Doc | Role |
|-----|------|
| `docs/experiments/mtp-stream-p0/MASTER_WORKLOG.md` | Chronological field rows |
| `docs/experiments/mtp-stream-p0/CRITICAL_ANALYSIS.md` | C-ladder decisions |
| `docs/experiments/mtp-stream-p0/MTP_OPTIMALITY_PLAN.md` | Path-to-100 / ceilings |
| `docs/analysis/mtp-review/05-p0-review.md` | P0/P2 residual register |
| `docs/analysis/mtp-review/06-tps-ceiling.md` | Sequential-verify arithmetic + batch probe plan |
| `docs/analysis/mtp-review/07-branch-map.md` | Short pointer → **this file** |
| PR #77 body | Product summary + abbreviated branch map |

---

## 12. Maintenance

- **Owner branch for this file:** `fix/mtp-stream-p0` (and alias `exp/mtp-stream-full`).  
- When creating any new `exp/*` or opening a product PR, **add a row** in §1/§2/§7 and bump “As of”.  
- Do not delete historical §3–§5 rows; mark status only.
