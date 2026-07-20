# Main binary construction method

Full worktree rebuild failed (HIP/clang `__AMDGCN_WAVEFRONT_SIZE` / env).  
**Main-equivalent binary** built by:

1. Working tree = `feat/openai-tools-server` CMake build (same mlx `0dadb703`, same hipcc).
2. Temporarily restore **main** versions of all files that differ from `origin/main`.
3. `cmake --build build --target server`.
4. Install as `build/server-main-equiv`.
5. Restore feat sources and rebuild feat `build/server`.

Files replaced from main (engine delta only):
- src/common/server.cpp
- src/common/chat_session.cpp
- src/common/chat_template.cpp
- src/common/tool_calling.cpp
- src/llm/llm_factory.cpp
- include/mlx-lm/common/openai_types.h
- include/mlx-lm/common/chat_template.h
- include/mlx-lm/common/model_container.h

`generate.cpp` / pure-graph / GDN / qwen35_moe / CMake mlx pin are **identical** on main — not swapped.
