# VOID — multi-turn stdin bug (not a KV result)

**Do not use these logs for the ≥5% long-ctx KV kill bar.**

- Cause: `longctx_prompt.txt` had many newlines; `chat.cpp` reads `std::getline` → one turn per line.
- Symptom: many ~82-token generations; multi-turn context growth, not single long prefill + 256 gen.
- Superseded by fixed single-line protocol in parent dir (`run_t1_longctx_kv.sh` r2+).
