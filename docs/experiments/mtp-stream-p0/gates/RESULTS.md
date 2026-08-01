# P0-MTP gate results (PR #63 → `fix/mtp-stream-p0`)

- **When:** 2026-08-01
- **Engine:** `fix/mtp-stream-p0` (post CPU-encoder TLS bind + M6 XOR log)
- **Model:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`
- **Device:** gfx1150 / ROCm · single process only
- **Flags:** `MLX_LOAD_MTP_HEAD=1` `MLX_ENABLE_QUANT_FUSE=1` `MLX_ENABLE_QUANT_FUSE_GDN=1` · `--use-mtp --n-draft-tokens 2` (server) / `--n-draft 2` (CLI)
- **Pure-graph:** default OFF (M6 sets `MLX_DECODE_GRAPH_PURE=1` only for XOR probe)

## Summary

| Gate | Status | Notes |
|------|--------|-------|
| **M1** short no-think MTP HTTP | **PASS** | HTTP 200 · `"Hi there, how are you today?"` · gen 23.4 t/s · MTP accept 5/5 · **no Stream(cpu)** |
| **M2** long no-think MTP HTTP | **PASS** | HTTP 200 · wave equation coherent · 612 tok · gen 25.4 t/s · accept 74% · **no Stream** |
| **M3** thinking + MTP HTTP | **PASS** | short+long HTTP 200 · thinking=on · 2+2→4 · Maxwell para · **no Stream** |
| **M4** CLI single-turn MTP | **PASS** | EXIT 0 · same short reply · gen 29.5 t/s · **no Stream** |
| **M6** pure XOR | **PASS** | log: `M6 XOR: MLX_DECODE_GRAPH_PURE=1 ignored while --use-mtp is active` · completes · **no Stream** |

**P0-MTP CLOSE eligible:** M1+M2+M3 **PASS**.

## Root cause (server-only failure before this fix)

CLI M1 was already green with `StreamGuard` + own gen stream (commit `69154ad`).  
Server M1 still returned:

```text
Generation error: There is no Stream(cpu, 0) in current thread.
```

**Cause:** mlx CPU `CommandEncoder`s are **thread_local**, registered only on the thread that called `cpu::new_stream`. Model load creates `Stream(cpu, 0)` on the main thread; httplib workers eval graphs that still reference that stream index → throw.

**Fix:** `ensure_thread_cpu_stream_encoders()` in `StreamGuard` re-binds all known CPU streams into the **current** thread's encoder map via `mlx::core::cpu::new_stream(s)` (`try_emplace`).

## Artifacts

| Path | Gate |
|------|------|
| `logs/M1-server-v2.log` + `raw/M1-short-v2.json` | M1 |
| `raw/M2-long.json` (same server as M1) | M2 |
| `logs/M3-server.log` + `raw/M3-short.json` + `raw/M3-long.json` | M3 |
| `logs/M4-chat.log` | M4 |
| `logs/M6-xor.log` | M6 |

## Harness

```bash
bash docs/experiments/mtp-stream-p0/run_p0_mtp_gates.sh
```

See also `../P0_MTP_GATES.md`.
