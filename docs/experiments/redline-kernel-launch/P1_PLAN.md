# P1 — Out-of-process AQL load of one HSACO (plan + harness)

**Date:** 2026-08-08  
**Branch:** `exp/redline-kernel-launch`  
**Status:** **PLAN + HARNESS SCAFFOLD** (no product wire)  
**Depends on:** E0 BUILD_OK · E1 AQL floor · E2 multi wall · E3 HSACO inventory · **P0 stub**

---

## 1. Goal (E4 §5)

> Out-of-process or tool path: load one MLX JIT `.hsaco` or toy CO in Redline.  
> **Success bar:** correctness gate only (load + symbol + one replay), **not** gen t/s.

**Hard bans:** no product CMake ON by default; no claiming E1/E2 speedups as model TPS; no LoopBrake.

---

## 2. Evidence constraints (critical)

| Source | Constraint |
|--------|------------|
| E3 | Product **qmm** is pointer/`hipLaunchKernel` — **not** drop-in Redline load |
| E3 | JIT path: `/tmp/mlx/<ver>/hsaco/gfx1150/*.hsaco` format-feasible |
| E1 | Prefer **AQL** `SingleQueueBatchGraph` + **BoundarySerialized** on gfx1150; PM4 example was gfx12-only FAIL |
| E2 | Host wall harness pattern in `harness/e2_aql_host_wall.rs` |

**Implication:** P1 must **not** start with qmm. Start with:

1. **Toy CO** (already: `logs/floor_kernel-gfx1150.co` from E0/E1) — smoke continuity, or  
2. **One small JIT elementwise/fused** `.hsaco` from MLX cache if present after a short run.

---

## 3. Deliverables

| ID | Artifact | Done when |
|----|----------|-----------|
| P1.a | `harness/p1_load_hsaco.rs` | Compiles against warpfront redline_dispatch |
| P1.b | Env contract | `REDLINE_P1_HSACO`, `REDLINE_P1_SYMBOL`, optional `REDLINE_P1_N` |
| P1.c | Log under `logs/p1-*.log` | Exit 0: load + kernel() + 1× replay_and_wait |
| P1.d | Doc `P1_LOAD.md` | Measured correctness (PASS/FAIL) on gfx1150 |

**Not P1:** engine link, `generate.cpp` session, MoE multipath.

---

## 4. Env contract (harness only — not product)

| Env | Default | Meaning |
|-----|---------|---------|
| `REDLINE_P1_HSACO` | required | Path to `.co` / `.hsaco` |
| `REDLINE_P1_SYMBOL` | `floor_k.kd` | Kernel symbol (E1 floor) |
| `REDLINE_P1_WARMUP` | `5` | Warmup replays |
| `REDLINE_P1_ITERS` | `20` | Timed replays (optional; report host µs, not gen t/s) |
| `REDLINE_P1_POLICY` | `BoundarySerialized` | Fence policy |

---

## 5. Procedure (host)

```text
1. Ensure Redline warpfront target still builds (E0 path / CARGO_TARGET_DIR).
2. Prefer existing floor CO:
     REDLINE_P1_HSACO=docs/experiments/redline-kernel-launch/logs/floor_kernel-gfx1150.co \
     REDLINE_P1_SYMBOL=floor_k.kd \
     cargo run --release -p ... --manifest-path ... --bin p1_load_hsaco
3. If JIT available: after any short mlx-lm run, pick one small non-qmm .hsaco;
   discover .kd name via redline load_symbols / llvm-objdump — document in P1_LOAD.md.
4. PASS: exit 0 + log "P1_OK load+replay".
5. FAIL: document hard blocker (symbol missing, ISA mismatch, runtime error).
```

---

## 6. Critical review / kill

| Failure | Action |
|---------|--------|
| Cannot load floor CO that E1 loaded | **Hard blocker** — Redline/runtime regression; re-run E0 |
| Can load floor but no MLX JIT symbol usable | **Partial PASS** — P1 correctness on toy; P3 needs recompile/shim |
| JIT loads but wrong grid for T=1 later | Defer to P3; P1 only needs one valid launch |

---

## 7. Relation to continuous loop

Each fire must advance one of: P1.a code, P1.c measurement, P1.d write-up, or documented hard blocker ×1.  
Empty fires count toward stop-after-3-empty.

After P1 green → P2 engine init smoke (still default OFF).
