# Loop4 empirical smoke attempt

**Date:** 2026-07-19  
**Branch:** fix/eager-no-mtp-correctness @ bda8f21+  
**Model:** LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit  
**Flags:** no `--use-mtp`, pure-graph default off (eager)

## Result

| Step | Outcome |
|------|---------|
| Unit tests (`test_chat_session`, `test_thinking_budget`, `test_stop_sequences`) | **PASS** |
| `./build/server` load canonical 35B | **SEGFAULT** during/after MTP head weight discovery log (see `server.log`) |

Segfault occurs **before** listening / health — load-path instability, not a failed thinking-floor assertion.  
**Does not block** unit-tested ChatSession/stop/thinking-floor commits. Follow-up: bisect load crash separately (out of MTP *decode* scope; load still probes mtp.* weights).

## Next

- Re-run load after clean rebuild / HIP env check  
- Or use smaller model only for server process smoke of floor logs (not acceptance for 35B)
