# GDN decode constants: load-time materialize

**Date:** 2026-07-19  
**Branch:** `fix/eager-no-mtp-correctness`  
**Why:** Explore supervisor pinned intermittent chat SIGSEGV to mid-forward  
`mx::eval(astype(a_log/dt_bias → f32))` on first GDN T=1 step  
(`qwen35_moe` decode path → mlx `copy_contiguous` → `hipLaunchKernel`).

## Change

- `Qwen35MoEGatedDeltaNet::materialize_decode_constants()` builds and `eval`s  
  `q_norm_w_`, `k_norm_w_`, `a_log_f32_`, `dt_bias_f32_` **at `load_weights`**.
- T=1 path uses them if already set; fallback materialize if missing.

**Not an NripeshN/mlx PR** — engine call-site hygiene (see `mlx-need-confirm-…`).

## Verify

| Check | Result |
|-------|--------|
| Units chat_session / thinking_budget / stop | PASS |
| Exclusive chat load to prompt | PASS (`logs/chat-load.log`) |
