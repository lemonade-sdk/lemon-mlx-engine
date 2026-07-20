# Loop5 / continue — 35B eager load fix

## Root cause (gdb)

After optional MTP head skip, warmup still SIGSEGV'd in:

`fuse_quant_projections` → `mx::concatenate` → `hipLaunchKernel` (ROCm)

on GDN `ensure_in_proj_fused` during ModelManager warmup.

## Fixes

1. **Skip MTP head by default** (`MLX_LOAD_MTP_HEAD=1` to enable) — qwen35 / qwen35_moe / qwen3_next  
2. **Disable quant projection fuse by default** (`MLX_ENABLE_QUANT_FUSE=1` to opt in) + shape guards  
3. **Not** MTP decode / Stream(cpu,0) work

## Smoke (canonical model)

| Check | Result |
|-------|--------|
| Health | ok |
| Chat no-think `2+2` | content `4`, finish `stop` |
| Log | `skipping optional MTP head build` |

```text
finish stop
content 4
usage completion_tokens=2 prompt_tokens=23
SMOKE_OK
```
