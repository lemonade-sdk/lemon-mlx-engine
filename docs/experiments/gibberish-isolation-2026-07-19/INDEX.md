# Isolation index — manual verification

**Canonical model only:** `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`

## Where to look

| Artifact | Path |
|----------|------|
| **Full responses (markdown, human-readable)** | [`RESULTS.md`](RESULTS.md) |
| **All responses (one JSON blob)** | [`ALL_RESPONSES.json`](ALL_RESPONSES.json) |
| **Per-step raw API JSON** | [`raw/`](raw/) |
| **Server logs per step** | [`logs/`](logs/) |
| **Runner log** | [`logs/runner.log`](logs/runner.log) |

## Step outcomes

| file | finish_reason / status | usage | content_chars | preview |
|------|------------------------|-------|---------------|---------|
| [`raw/L0-long-wave.json`](raw/L0-long-wave.json) | `length` | `{'completion_tokens': 800, 'prompt_tokens': 45, 'total_tokens': 845}` | 2182 |  derive the wave equation for the electric field $\mathbf{E}$ in a vacuum, we start with Maxwell's e |
| [`raw/L0-short-maxwell.json`](raw/L0-short-maxwell.json) | `stop` | `{'completion_tokens': 265, 'prompt_tokens': 36, 'total_tokens': 301}` | 866 | . **Gauss’s Law for Electricity**: $\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$      *Mean |
| [`raw/L1-long-wave.json`](raw/L1-long-wave.json) | `length` | `{'completion_tokens': 800, 'prompt_tokens': 45, 'total_tokens': 845}` | 2183 | To derive the wave equation for the electric field $\mathbf{E}$ in a vacuum, we start with Maxwell's |
| [`raw/L1-short-maxwell.json`](raw/L1-short-maxwell.json) | `stop` | `{'completion_tokens': 266, 'prompt_tokens': 36, 'total_tokens': 302}` | 867 | 1. **Gauss’s Law for Electricity**: $\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$      *Mea |
| [`raw/L2-long-wave.json`](raw/L2-long-wave.json) | `length` | `{'completion_tokens': 800, 'prompt_tokens': 45, 'total_tokens': 845}` | 2175 |  derive the wave equation for the electric field $\mathbf{E}$ in a vacuum, we start with Maxwell's e |
| [`raw/L2-short-maxwell.json`](raw/L2-short-maxwell.json) | `stop` | `{'completion_tokens': 250, 'prompt_tokens': 36, 'total_tokens': 286}` | 826 | . **Gauss’s Law for Electricity**: $\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}$    *Meanin |
| [`raw/L3-long-thinking.json`](raw/L3-long-thinking.json) | `length` | `{'completion_tokens': 800, 'prompt_tokens': 43, 'total_tokens': 843}` | 2003 | 's a thinking process:  1.  **Understand User Request:**    - Derive the wave equation for the elect |
| [`raw/L3-short-thinking.json`](raw/L3-short-thinking.json) | `length` | `{'completion_tokens': 512, 'prompt_tokens': 34, 'total_tokens': 546}` | 1993 | 's a thinking process:  1.  **Analyze User Request:**    - **Topic:** Maxwell's equations in vacuum  |
| [`raw/L4-long-mtp.json`](raw/L4-long-mtp.json) | `ERROR` | `{'code': 'internal_error', 'message': 'Generation error: There is no Stream(cpu, 0) in current thread.', 'type': 'server_error'}` | 0 | {'error': {'code': 'internal_error', 'message': 'Generation error: There is no Stream(cpu, 0) in cur |
| [`raw/L4-short-mtp.json`](raw/L4-short-mtp.json) | `ERROR` | `{'code': 'internal_error', 'message': 'Generation error: There is no Stream(cpu, 0) in current thread.', 'type': 'server_error'}` | 0 | {'error': {'code': 'internal_error', 'message': 'Generation error: There is no Stream(cpu, 0) in cur |

## Config map

| Step | Config |
|------|--------|
| L0 | `MLX_DECODE_GRAPH_PURE_OFF=1` + `MLX_SYNC_DECODE=1` + `--no-think` + no MTP |
| L1 | pure OFF, no SYNC, `--no-think`, no MTP |
| L2 | pure **default ON**, no SYNC, `--no-think`, no MTP |
| L3 | pure OFF + SYNC, **thinking ON**, no MTP |
| L4 | pure OFF + SYNC + `--use-mtp` + `--no-think` |

Open **RESULTS.md** for full inlined model text under each heading.
