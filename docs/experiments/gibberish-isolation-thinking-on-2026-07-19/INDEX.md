# Thinking-ON isolation + main proof — index

**Model only:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`  
**Thinking:** **ON**

## Start here

| Doc | Purpose |
|-----|---------|
| [`CLARIFICATION_NOT_CLEAN.md`](CLARIFICATION_NOT_CLEAN.md) | We are **not** saying feat has no issues |
| [`COMPARISON_FEAT_VS_MAIN.md`](COMPARISON_FEAT_VS_MAIN.md) | feat vs main proof (content-identical) |
| [`feat/RESULTS.md`](feat/RESULTS.md) | Full responses — **feat** binary |
| [`main/RESULTS.md`](main/RESULTS.md) | Full responses — **main-equiv** binary |
| [`feat/raw/`](feat/raw/) | Per-step JSON feat |
| [`main/raw/`](main/raw/) | Per-step JSON main |
| [`ALL_THINKING_ON_RESPONSES.json`](ALL_THINKING_ON_RESPONSES.json) | Aggregate |

## Headline results (thinking ON)

| Step | Config | feat | main |
|------|--------|------|------|
| T0 short | pure OFF + SYNC | stop, coherent | **same content** |
| T0 long | pure OFF + SYNC | **length** @1600 | **same** |
| T2 short | pure ON default | stop, coherent | **same content** |
| T2 long | pure ON default | stop | **same content** |
| T4 | MTP | **500 Stream(cpu,0)** | **same 500** |

## Issues remaining (shared unless noted)

1. **MTP broken** (P0) — both  
2. **Thinking length cut** — both  
3. **CLI ChatSession double-prefill** — both (code)  
4. **Tools surface** — feat only  
5. Mid-stream soup **not reproduced** on this single-turn HTTP ladder (thinking on)
