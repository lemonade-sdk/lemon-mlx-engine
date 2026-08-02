# Redline ROCm dispatch research (pwilkin vs warpfront)

**Date:** 2026-08-02  
**Local trees:** `/tmp/redline-pwilkin`, `/tmp/redline-warpfront`  
**Remotes:** [github.com/warpfront/redline](https://github.com/warpfront/redline) (upstream), [github.com/pwilkin/redline](https://github.com/pwilkin/redline) (fork)  
**Author / trademark:** Kaden Schutt (`kaden@hipfire.dev`); workspace `repository` field still points at `Kaden-Schutt/redline` in `Cargo.toml`.  
**Audience:** lemonade-sdk / lemon-mlx-engine C++ MLX on ROCm (gfx1150/1151 decode, MoE T=1 launch-bound).

---

## 1. Which repo is the real project?

| Criterion | **warpfront/redline** | **pwilkin/redline** |
| --- | --- | --- |
| GitHub relationship | **Upstream** (`warpfront/redline`) | **Fork of** `warpfront/redline` |
| Commit count (GitHub UI) | **44** on `master` | **36** on `master` |
| Product identity | Same README brand: “Lightning-fast kernel dispatch for ROCm” | Same content at fork tip |
| Local checkout delta | Has `crates/redline-hipgraph/src/shims/*` and `docs/investigations/2026-08-01-hipgraph-handle-ownership-redesign.md` | **No** `shims/` tree, **no** hipgraph ownership investigation doc |
| NOTICE / copyright | Kaden Schutt / hipfire | Same |

**Verdict:** The real “fast kernel launch for ROCm” project is **warpfront/redline** (author Kaden Schutt / Hipfire lineage). **pwilkin/redline** is a personal fork that is currently **behind** upstream on hipGraph interposer completeness (missing shim surface + ownership redesign write-up). For integration work, pin **warpfront** (or Kaden’s canonical remote once crates are published), not the pwilkin fork.

Both local checkouts implement the same core story: retained PM4 / AQL replay over public ROCr/HSA queues to beat HIP’s per-launch fence+submit floor ([ROCm/ROCm#6409](https://github.com/ROCm/ROCm/issues/6409)).

---

## 2. Architecture

### 2.1 Crate map

From `/tmp/redline-warpfront/README.md` and workspace `Cargo.toml`:

| Crate | Path | Role |
| --- | --- | --- |
| **radiowave** | `crates/radiowave/` | Compiler **policy** (not a compiler fork): drives installed hipcc/LLVM, inspects CO, certifies VMEM-only mutable reads for narrow RMW, hashed manifests |
| **redline-dispatch** | `crates/redline-dispatch/` | Record/replay core, hazard/visibility, plan fingerprint, HIP multistream backend, **AQL + retained PM4**, Rust hipGraph adapter |
| **redline-rocr** | `crates/redline-rocr/` | Public ROCr/HSA ABI bindings, AQL packets, queue publication, **GFX10/11 and GFX12 PM4 encoders** (no vendored ROCm sources; see `PROVENANCE.md`) |
| **redline-capi** | `crates/redline-capi/` | Stable C ABI (`include/redline_dispatch.h`) for engines |
| **redline-py** | `crates/redline-py/` | PyO3 bindings (workspace-excluded; maturin) |
| **redline-hipgraph** | `crates/redline-hipgraph/` | `hipGraph*` / launch interposer + optional Python control; `LD_PRELOAD` |
| **redline-observe** | `crates/redline-observe/` | Optional roctx / amd-smi / rocprof hooks (ROCm 7.14) |

Workspace members: `radiowave`, `redline-dispatch`, `redline-rocr`, `redline-observe`, `redline-capi`, `redline-hipgraph` (`Cargo.toml`). Rust ≥ 1.85, edition 2024.

### 2.2 Retained PM4

Flow (engine-level, from `docs/INTEGRATION.md`):

```text
select GPU → load HSACO → pack kernargs → record dispatches + waits
         → finalize once → patch changed kernargs → replay + wait
         → free IB → free module → free GPU
```

- **Record:** `RlPm4Builder` / Rust `SingleQueuePm4Ib` / multi-queue variants accumulate family-specific PM4 into one (or N) **indirect buffer(s)**.
- **Submit:** one vendor AQL PM4-IB packet (or multi-queue set) over **public ROCr queues**, not N HIP launches.
- **Encoders:**
  - GFX12: `redline-rocr/src/pm4.rs` (`Gfx12Pm4CommandBuffer`, `DISPATCH_DIRECT`, cache acquire policies including `HipLlvmVmemL1`).
  - GFX10/11: `pm4_gfx10.rs` + type alias / path for Gfx11 (`Gfx11Pm4CommandBuffer` re-exported from dispatch AQL mod).
- **Family dispatch:** `Pm4Family::from_name` in `redline-capi/src/gpu.rs`:
  - `gfx10*` → Gfx10
  - `gfx11*` → Gfx11 (covers **gfx1100, gfx1150, gfx1151**, …)
  - `gfx12*` → Gfx12
  - else **fail closed** (`None`)
- **Decode-friendly leading acquire:** builders inject a leading same-agent acquire so **in-place kernarg mutation** between replays is not read stale from scalar cache (`gpu.rs` comments on `rl_pm4_ib_set_kernargs`).
- **Constraints (README / INTEGRATION):** zero-scratch HSA kernels preferred for legacy direct PM4; unsupported scratch / queue / dispatch / flat-scratch contracts **fail closed**. Wrong encoder family for device is rejected.

### 2.3 ROCr / HSA queues

- `redline-rocr`: dynamic load of `libhsa-runtime64`, transcribed public headers (audited ROCm 7.14, `HSA_AMD_INTERFACE_VERSION` 1.26). Missing ≥7.14 symbols → hard error.
- `QueueSet` / multi-queue: `redline-dispatch/src/aql/queue_policy.rs`
  - **Auto:** Q2 for `gfx12*` and `gfx1100`; **Q4 for other `gfx11*`** (includes gfx1151/likely gfx1150); **Q1** for unmeasured gfx10 / unknown.
  - Serial RMW / independent_width=1 stays single-queue.
- gfx1100 Q3/Q4: documented **retained-PM4 timeouts** on adjacent memory-waitcnt rows — auto capped at Q2 (`README.md`, hipfire three-way notes).
- Fault / wait: finite timeouts; queue info attrs include VM-fault related constants in ABI expansion (`PROVENANCE.md`).

### 2.4 HIP Graph relation

Three layers, not mutually exclusive:

1. **Native Redline graph IR** (`Recorder` / `rl_graph_*`): explicit buffer accesses → hazard compile → instantiate → launch (can lower to PM4).
2. **Rust `hipgraph` module** in `redline-dispatch`: HIP-shaped Graph/GraphExec for migration examples.
3. **`redline-hipgraph` preload:** intercepts `hipGraph*`, `__hipRegister*`, and kernel launches; supported module-loaded / fatbin captures can resolve to **retained PM4**; unsupported ops fall through to real HIP. **Not** a full HIP runtime.

**warpfront-only evolution:** `docs/investigations/2026-08-01-hipgraph-handle-ownership-redesign.md` + `src/shims/*` — redesign so apps hold **native** HIP handles (side-table for Redline plan) instead of fabricated heap pointers, avoiding silent corruption on unshimmed entry points. pwilkin checkout lacks this surface.

### 2.5 API surface for launching kernels

#### C (preferred for C++ engines) — `crates/redline-capi/include/redline_dispatch.h`

| Stage | APIs |
| --- | --- |
| GPU / module | `rl_gpu_new`, `rl_gpu_load_module`, `rl_gpu_load_module_radiowave`, `rl_module_kernarg_size`, free helpers |
| Low-level retained PM4 | `rl_pm4_builder_new` → `rl_pm4_dispatch` → `rl_pm4_wait_rmw` / `rl_pm4_wait_idle` → `rl_pm4_finalize` → `rl_pm4_ib_set_kernargs` → `rl_pm4_replay` |
| Multi-queue | `rl_gpu_pm4_queue_count(..., RlQueueAuto, width)`, `rl_pm4_finalize_multi`, `rl_pm4_replay_multi`, multi kernarg set |
| Graph | `rl_graph_new[_tuned]`, `rl_graph_buffer`, `rl_graph_kernel[_ex]`, `rl_graph_add_dependency`, `rl_graph_instantiate`, `rl_graphexec_launch` |
| Errors | `RL_OK`, `RL_ERR_NULL/UTF8/RECORD/COMPILE/REPLAY/HANDLE/CERTIFICATION` |

Decode pattern (documented + `examples/decode_chain_ab.c`): finalize once; per token `rl_pm4_ib_set_kernargs` + `rl_pm4_replay` (replay waits completion).

#### Python — `redline_dispatch` / `Gpu.build` (GFX12-only for direct PM4 in Python today; C/Rust for GFX10/11)

#### Preload — `LD_PRELOAD=libredline_hipgraph.so`

#### Rust — path dep on `redline-dispatch` (`Recorder`, `aql::SingleQueuePm4Ib`, etc.)

**Not in surface:** drop-in replacement for every `hipLaunchKernelGGL` without either C API wiring, graph capture, or interposer. Per-launch HIP remains the fallback for dynamic topology / unsupported kernels.

---

## 3. Benchmark claims vs HIP / hipGraph

### 3.1 Current product scorecard (ROCm 7.14) — Hipfire three-backend

Source: `examples/hipfire-6409/README.md` + per-arch `REPORT.md` at commit `bb612d14…`.

| GPU | Arch | Redline 1st | RL > HIP | Median RL/HIP | RL > Vulkan | Median RL/Vulkan | Report |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RX 9070 XT | gfx1201 | **187/240 (77.9%)** | 222/240 | 0.284× | 187/240 | 0.758× | `…/gfx1201/2026-07-23-rocm714-three-way/REPORT.md` |
| RX 7900 XTX | gfx1100 | **182/240 (75.8%)** | 227/240 | 0.427× | 192/240 | 0.701× | `…/gfx1100/2026-07-23-rocm714-three-way/REPORT.md` |
| Radeon 8060S (Strix Halo) | **gfx1151** | **185/240 (77.1%)** | 233/240 | **0.373×** | 185/240 | 0.772× | `…/gfx1151/2026-07-23-rocm714-three-way/REPORT.md` |

All 720 rows matched oracles (zero rejects) on the clean three-card set.  
**Note:** these Hipfire three-way reports **do not measure HipGraph** as a fourth backend (verdict text: “does not measure HipGraph”). HIP column is direct HIP launch, not graph replay.

Earlier gfx1201 four-backend control: `…/2026-07-22-rocm7.14-retest/REPORT.md` — 192/240 Redline firsts; RL beat Vulkan and HipGraph/HIP in 192 and 222 rows (README index).

### 3.2 HipEngine pristine harness (ROCm 7.14)

`examples/hipengine-6409/results/*/2026-07-22-714-bench/REPORT.md`:

| Arch | RL > Vulkan | RL > HIP | Notes |
| --- | ---: | ---: | --- |
| gfx1201 | 197/224 | (see REPORT) | README: 197/224 firsts vs Vulkan |
| **gfx1151** | **164/224 (73.2%)** | **182/224 (81.2%)** | Serial RMW strong; independent mixed; packed-dot often loses (codegen) |
| gfx1100 | 127/224 firsts (README) | — | Weaker relative to Vulkan |

### 3.3 Historical dispatch-floor µs (ROCm 7.2, R9700 / gfx1201) — methodology only

`docs/DISPATCH-FLOOR.md` **explicitly warns**: not current product scorecard.

| Comparison | Claim |
| --- | --- |
| Fence policy only (`BoundarySerialized` vs system-every-dispatch) | ~**1.8×** (decode-safe) |
| Literal `hipGraphLaunch` vs Redline BoundarySerialized | ~**1.6×** (N=256/512 no-op) |
| PM4 IB champion (host µs/disp, counter kernel) vs hipGraph | ~**10–12×** conservative / up to ~17× aggressive independent |

Ratios shrink toward 1× as real compute grows. Residual Vulkan wins on product matrices are attributed mostly to **codegen** (packed-dot / VOPD), not transport.

### 3.4 Decode-chain micro A/B (host us/token)

`crates/redline-capi/examples/decode_chain_ab.c` + `bench/decode_chain.hip`:

- hipGraph: 2-node graph, `hipGraphExecKernelNodeSetParams` + `hipGraphLaunch` per token  
- Redline: stage1 → `rl_pm4_wait_rmw` → stage2, `set_kernargs` + `rl_pm4_replay`  
- Correctness: `acc == T*(T+1)`  

Use this as the **integration smoke** for lemon, not the Hipfire matrix.

---

## 4. Hardware support (gfx1150 / gfx1151 / RDNA3.5 / Strix)

| Path | gfx1151 (Strix Halo / 8060S) | gfx1150 (Strix Point class / lemon builds) | RDNA3 dGPU (gfx1100) | RDNA4 (gfx1201) |
| --- | --- | --- | --- | --- |
| Public AQL replay | Exercised | Same family expected (not separately listed) | Exercised | Exercised |
| Retained PM4 | **Gfx11 encoder** (`starts_with("gfx11")`) | **Same Gfx11 encoder** by prefix | Gfx11 | Gfx12 |
| Radiowave `ArchProfile` | Exact **Gfx1151** | **`from_arch("gfx1150")` is None** (no profile bleed) | Gfx1100 profile | gfx1201 campaigns |
| Auto multi-queue | Q4 (`gfx11*` not gfx1100) | Q4 by same rule | Q2 (capped) | Q2 |
| Product benches | Full three-way + HipEngine | **No dedicated retained REPORT** in tree | Full | Full |
| lemon-mlx-engine | Related APU family | **Active product target** (`build/*gfx1150*`) | — | — |

**Implications for lemon (gfx1150):**

1. **PM4 family matching should work** (gfx1150 → Gfx11).  
2. **Radiowave exact-arch recipes for gfx1151 do not automatically apply** as certified gfx1150 bundles — fail-closed identity checks treat architectures as distinct (`radiowave/src/arch.rs` tests reject `gfx1150` as Gfx1151).  
3. Certify on **gfx1150** with the same harnesses; do not assume gfx1151 medians.  
4. Also exercised historically: gfx1010, gfx1030. Requires **ROCm Core SDK ≥ 7.14** TheRock layout (`/opt/rocm/core`).

---

## 5. Integration path options for C++ MLX / lemon-mlx-engine

lemon already has:

- `src/common/graph_decode.cpp` — opt-in pure decode graph (`MLX_DECODE_GRAPH`), fixed device pos/input buffers  
- Prefill HIP graph experiment (`docs/experiments/prefill-hip-graph/`) — ~**+2.7%** pp/s, **&lt;10% bar**, extra ~2.5 GB; pure decode graph often **regresses** on APU  
- Product decode: launch-heavy MoE T=1 on **8 CU gfx1150** (`docs/analysis/mtp-review/06-tps-ceiling.md`, `docs/ROCM_TPS_OPTIMIZATION_OPERATORS_KV.md`)

### Path A — Explicit C ABI (recommended first)

| | |
| --- | --- |
| **What** | Link `libredline_dispatch` + `redline_dispatch.h`; own HSACO (or extract CO from hipModule), fixed per-token dispatch skeleton for hottest serial chain |
| **Fit** | Engine owns allocations; decode has **fixed L=1 topology** with mutable scalars/pointers (pos, token id, expert indices if packed) |
| **Hot path** | `rl_pm4_finalize` once at capture; each token `rl_pm4_ib_set_kernargs` + `rl_pm4_replay` |
| **Refs** | `docs/INTEGRATION.md` §C/C++ engine API; `decode_chain_ab.c`, `decode_kernargs` examples |
| **Pros** | Explicit ownership; no interposer; matches Redline’s design center |
| **Cons** | Must re-express MLX’s eval graph as a **fixed** PM4 sequence; dynamic expert set / variable launches need multi-IB or fall back to HIP |

### Path B — hipGraph ABI / `LD_PRELOAD` interposer

| | |
| --- | --- |
| **What** | Build `redline-hipgraph`; `LD_PRELOAD` into `./build/chat` while using existing stream-capture / `use_hip_graphs` paths |
| **Fit** | Lowest code churn if MLX already builds hipGraphExec for decode/prefill |
| **Pros** | No rewrite of kernel launch sites |
| **Cons** | Incomplete HIP coverage; warpfront ownership redesign is load-bearing for safety; lemon pure-graph currently **not a win on APU**; unsupported nodes silently (or via fallthrough) lose acceleration; ABI/shim churn |

### Path C — Selective replacement of `hipLaunchKernel` / module launch for fixed ops

| | |
| --- | --- |
| **What** | For known MoE expert matmul / router / fused GDN T=1, bypass MLX stream commit and call Redline replay for that sub-DAG |
| **Fit** | When full forward capture is too dynamic but a **subset** is launch-bound and shape-stable |
| **Cons** | Stream ordering vs rest of MLX runtime (events, async_eval one-behind) must be designed carefully |

### Path D — Command-buffer replay without hipGraph (Redline-native graph)

| | |
| --- | --- |
| **What** | `rl_graph_kernel_ex` + resource accesses for whole L=1 forward, or multi-queue independent expert antichains |
| **Fit** | MoE experts that touch **disjoint** weight tiles can use multi-queue (`RlQueueAuto` → Q4 on gfx11) only if independence is real |
| **Decode serial chain** | Must stay **serialized** with `wait_rmw` / BoundarySerialized — aggressive independent fences are **wrong** for shared activations |

### Path E — Do **not** replace all of HIP

Redline is **not** a HIP runtime. Keep HIP for:

- Allocations, memcpy, events, random launches  
- Prefill (compute-bound WMMA; graph already small gain)  
- Any kernel with scratch / unsupported user-SGPR contract until certified  

**Practical lemon ranking:** A (C ABI micro-bench on extractable HSACO) → C (hot sub-DAG) → B only after A proves µs/token on gfx1150 → avoid whole-model pure-graph unless dGPU launch-bound evidence appears.

---

## 6. Risks

| Risk | Evidence / mechanism | Mitigation |
| --- | --- | --- |
| **ROCm version** | Hard requires **≥ 7.14** TheRock (`/opt/rocm/core`); older stack fails symbol load | Align lemonade ROCm with 7.14+ before integrate |
| **VM faults / device hang** | Direct PM4 + wrong fences / wrong family / multi-queue oversubscription | Fail-closed family checks; start Q1 serial; finite wait timeouts; dmesg + `hsa_amd` queue VM-fault attrs |
| **gfx1100-class multi-queue timeouts** | Explicit Q3/Q4 retained-PM4 timeouts in waitcnt rows | Auto Q2 on gfx1100; on gfx115x start Auto but A/B Q1 |
| **Scratch / implicit SGPR kernels** | Unsupported contracts rejected or unsafe if forced | Prefer zero-scratch; radiowave inspect; fall back HIP |
| **Stale kernarg scalar cache** | Decode must see per-token patches | Leading acquire on builder (already in C path); never patch during in-flight replay |
| **hipGraph interposer corruption** | Fabricated handles vs native (pre-redesign) | Prefer **warpfront** with shims/ownership redesign; prefer C ABI over preload |
| **ABI instability** | C API versioned (`abi=1` smoke); crates not on crates.io/PyPI yet | Vendor pin commit; static link or rpath; track header |
| **Correctness vs HIP ordering** | Minimal fences from declared accesses; wrong access list → races | Declare all RMW; `wait_rmw` for consumers; oracle tests |
| **gfx1150 ≠ gfx1151 certification** | Radiowave exact arch; no gfx1150 report artifacts | Re-run harnesses on product APU |
| **Codegen residual** | Packed-dot / VOPD Vulkan wins | Redline fixes **launch/fence**, not ACO quality |
| **Memory / lifetime** | IB, module, device pointers must outlive replays | Follow INTEGRATION lifetime rules |
| **Trademark / license** | Apache-2.0; “Redline” name trademark of Kaden Schutt | NOTICE retention; don’t brand forks as Redline product |
| **Process isolation** | Mixing HIP streams and Redline queues without barriers | Explicit device sync at integration boundaries initially |

---

## 7. Concrete next experiments (lemon decode / MoE T=1 launch-bound)

Goal: quantify whether **retained submission** recovers tokens/s on **gfx1150** MoE T=1 where pure hipGraph already failed or regressed.

### E0 — Environment gate (½ day)

1. Confirm host ROCm Core **≥ 7.14** and `/opt/rocm/core` layout.  
2. `cargo build --release -p redline-capi -p redline-dispatch` from **warpfront** tree.  
3. Run C smokes: `smoke.c` (no GPU), `gpu_smoke.c` with counter HSACO for **gfx1150** (`hipcc --genco --offload-arch=gfx1150`).  
4. Run `decode_chain_ab` for T=256/512; record hipGraph vs Redline **us/token** and PASS/FAIL.

**Pass:** counter certification + decode_chain PASS on gfx1150.

### E1 — Dispatch floor on product APU (½ day)

Reproduce `docs/DISPATCH-FLOOR.md` method with gfx1150 CO:

```bash
hipcc --genco --offload-arch=gfx1150 bench/floor_kernel.hip -o /tmp/floor_kernel.co
ROCR_VISIBLE_DEVICES=0 REDLINE_FLOOR_HSACO=/tmp/floor_kernel.co \
  cargo run --release --example dispatch_floor -p redline-dispatch
```

Also `floor_hipgraph.hip` baseline if present.  
**Metric:** µs/dispatch BoundarySerialized vs hipGraph.  
**Pass:** ≥1.3× host or GPU-span advantage on no-op batch (directional only).

### E2 — MoE-shaped multi-kernel chain microbench (1–2 days)

Build a **synthetic** chain mimicking one MoE layer T=1:

- router (tiny) → top-k scatter → **K expert matmuls** (independent weights) → combine  
- Arms: (1) HIP stream launches, (2) single hipGraphExec, (3) Redline serial PM4, (4) Redline multi-queue only for independent experts  

Kernels can start as Hipfire-style dense-q8 / selected-dual shapes from matrix families (see gfx1151 REPORT losses/wins).  
**Metric:** us/layer, correctness vs CPU oracle.  
**Hypothesis:** serial decode chain wins on fences; expert antichain may gain from Q2–Q4 **only if** memory independent.

### E3 — HSACO extract from lemon/MLX (1–2 days)

1. Identify top launch-count kernels in one decode step (rocprof / env launch counters).  
2. Dump HSACO for those symbols (module load path or fatbin extract).  
3. Measure kernarg size via `rl_module_kernarg_size` / code object metadata; map pointer slots.  
4. Replay **subset** under Redline with frozen addresses (arena-style fixed buffers, same idea as `graph_decode_pos` / `graph_decode_input`).

**Pass:** bit-identical (or tol) vs HIP for 64 tokens.

### E4 — lemon C++ FFI spike (2–3 days)

1. CMake option `LEMON_REDLINE=ON`: find `libredline_dispatch`, include header.  
2. Behind `MLX_REDLINE_DECODE=1`: for a **single** fixed subgraph (e.g. fused residual+norm or GDN fused2 alone), call finalize once + per-token replay.  
3. Keep full model on HIP; barrier before/after Redline region.  
4. A/B gen t/s on Qwen3.x MoE 35B-class, temp=0, max-tokens=128, same as existing chat probes.

**Success bar (product):** ≥5% gen t/s on gfx1150 35B **or** ≥10% on a deliberately tiny launch-bound model; zero quality regress.  
If &lt;2%, close the Redline track for APU and keep for dGPU-only hypothesis (`ROCM_TPS…` already notes pure-graph may help R9700-class).

### E5 — Optional preload A/B (only if E3/E4 green)

`LD_PRELOAD=libredline_hipgraph.so` with existing `MLX_DECODE_GRAPH` / prefill graph flags.  
Use **warpfront** shims build.  
**Abort criteria:** any SEGV, VM fault, or numeric collapse → abandon preload for product.

### E6 — Report artifacts

Write under this directory:

- `E0_env.txt`, `E1_floor.json`, `E2_moe_chain.md`, `E4_tps_ab.md`  
- Pin redline git SHA and `rocminfo` arch string  

---

## 8. File index (evidence)

| Topic | Paths |
| --- | --- |
| Product README / claims | `/tmp/redline-warpfront/README.md` |
| Integration guide | `/tmp/redline-warpfront/docs/INTEGRATION.md` |
| Dispatch floor methodology | `/tmp/redline-warpfront/docs/DISPATCH-FLOOR.md` |
| C ABI | `/tmp/redline-warpfront/crates/redline-capi/include/redline_dispatch.h` |
| Decode A/B | `/tmp/redline-warpfront/crates/redline-capi/examples/decode_chain_ab.c` |
| PM4 GFX12 | `/tmp/redline-warpfront/crates/redline-rocr/src/pm4.rs` |
| Queue policy | `/tmp/redline-warpfront/crates/redline-dispatch/src/aql/queue_policy.rs` |
| PM4 family | `/tmp/redline-warpfront/crates/redline-capi/src/gpu.rs` |
| ROCr provenance | `/tmp/redline-warpfront/crates/redline-rocr/PROVENANCE.md` |
| hipGraph interposer | `/tmp/redline-warpfront/crates/redline-hipgraph/README.md` |
| Ownership redesign | `/tmp/redline-warpfront/docs/investigations/2026-08-01-hipgraph-handle-ownership-redesign.md` |
| gfx1151 three-way | `/tmp/redline-warpfront/examples/hipfire-6409/results/gfx1151/2026-07-23-rocm714-three-way/REPORT.md` |
| gfx1151 HipEngine | `/tmp/redline-warpfront/examples/hipengine-6409/results/gfx1151/2026-07-22-714-bench/REPORT.md` |
| lemon graph decode | `/home/antmi/lemon-mlx-engine/src/common/graph_decode.cpp` |
| lemon prefill graphs | `/home/antmi/lemon-mlx-engine/docs/experiments/prefill-hip-graph/README.md` |
| lemon TPS / launch-bound context | `/home/antmi/lemon-mlx-engine/docs/ROCM_TPS_OPTIMIZATION_OPERATORS_KV.md`, `docs/analysis/mtp-review/06-tps-ceiling.md` |
| pwilkin (fork, lagging hipgraph) | `/tmp/redline-pwilkin` |

---

## 9. Bottom line for lemonade / lemon-mlx-engine

1. **Use warpfront/redline**, not the lagging pwilkin fork.  
2. Redline’s lever is **fence + submission batching** via retained PM4 over public ROCr — highly relevant to **MoE T=1 launch-bound** decode, less so to prefill WMMA.  
3. On **Strix-class APUs**, gfx1151 data shows large median wins vs raw HIP on micro matrices (**~0.37× RL/HIP**), but lemon’s **whole-forward pure hipGraph already fails the product bar**; expect wins only if a **stable multi-kernel IB** is constructed with correct RMW boundaries.  
4. **gfx1150** is PM4-compatible by prefix but **not radiowave-certified** as gfx1151 — re-measure.  
5. Prefer **C ABI + finalize/replay** over `LD_PRELOAD` for a production-grade integration.  
6. First prove E0–E2 on the product APU; only then touch lemon `generate` / MLX backend.

---

*Generated by domain-oriented ROCm/HIP kernel-dispatch research pass. Evidence-based; no live GPU re-run in this pass.*
