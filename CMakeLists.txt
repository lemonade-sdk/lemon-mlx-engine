cmake_minimum_required(VERSION 3.20)
project(mlx-cpp-lm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MLX_LM_BUILD_TESTS "Build tests" ON)
option(MLX_LM_BUILD_EXAMPLES "Build example programs" ON)

# Third-party dependencies
include(FetchContent)

# Enable ROCm/AMD GPU backend on non-Apple platforms (default ON, can be overridden)
if(NOT APPLE)
    if(NOT DEFINED MLX_BUILD_ROCM)
        set(MLX_BUILD_ROCM ON CACHE BOOL "Build ROCm backend")
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES
            "gfx908;gfx90a;gfx942;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1200;gfx1201"
            CACHE STRING "HIP architectures" FORCE)
    endif()
endif()

# Enable AMD XDNA NPU backend (requires XRT)
option(MLX_LM_BUILD_NPU "Build AMD XDNA NPU backend via IRON" OFF)
if(MLX_LM_BUILD_NPU)
    set(MLX_BUILD_NPU ON CACHE BOOL "Build NPU backend" FORCE)
endif()

# MLX C++ (lemonade-sdk fork with ROCm/AMD GPU + NPU support)
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/NripeshN/mlx.git
    # Pin MLX to post-PR#7 commit (2026-05-20).
    # Includes soloish90's 8-bit QMV fix (PR #6) AND the uint4 miscompile
    # fix (PR #7 — 9e768e4) where global_load_b128 miscompiles on RDNA 3.5
    # with hipcc 7.13 / LLVM 23. Still avoids clear_streams() which the
    # ROCm backend doesn't implement yet.
    GIT_TAG        9e768e453ab1e1906b10c004ecf588200dbbe968
    GIT_SHALLOW    FALSE
)
FetchContent_MakeAvailable(mlx)

# Patch: add clear_streams() to ROCm backend (missing in NripeshN/mlx fork)
# Upstream MLX PR #3395 added clear_streams() but ROCm backend lacks it.
# We need: (1) clear_encoders() method on Device, (2) get_devices() accessor,
# (3) clear_all_encoders() in rocm namespace, (4) clear_streams() in gpu namespace.
if(MLX_BUILD_ROCM)
    # 1) Patch device.h: add clear_encoders() to Device class,
    #    and clear_all_encoders() to rocm namespace
    file(READ "${mlx_SOURCE_DIR}/mlx/backend/rocm/device.h" ROCM_H)
    if(NOT ROCM_H MATCHES "clear_encoders")
        # Add clear_encoders() after get_command_encoder in Device class
        string(REPLACE
            "  CommandEncoder& get_command_encoder(Stream s);\n\n  int hip_device() const {"
            "  CommandEncoder& get_command_encoder(Stream s);\n  void clear_encoders();\n\n  int hip_device() const {"
            ROCM_H
            ROCM_H
        )
        # Add clear_all_encoders() after get_command_encoder free function
        string(REPLACE
            "CommandEncoder& get_command_encoder(Stream s);\n\n"
            "CommandEncoder& get_command_encoder(Stream s);\nvoid clear_all_encoders();\n\n"
            ROCM_H
            ROCM_H
        )
        file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/rocm/device.h" ROCM_H)
    endif()

    # 2) Patch device.cpp: extract devices map to get_devices(), add clear functions
    file(READ "${mlx_SOURCE_DIR}/mlx/backend/rocm/device.cpp" ROCM_CPP)
    if(NOT ROCM_CPP MATCHES "clear_all_encoders")
        # Extract devices to a named accessor function
        string(REPLACE
            "Device& device(mlx::core::Device device) {\n  static std::unordered_map<int, Device> devices;\n  static bool flags_set = false;"
            "std::unordered_map<int, Device>& get_devices() {\n  static std::unordered_map<int, Device> devices;\n  return devices;\n}\n\nDevice& device(mlx::core::Device device) {\n  auto& devices = get_devices();\n  static bool flags_set = false;"
            ROCM_CPP
            ROCM_CPP
        )
        # Add clear_all_encoders() before namespace close
        string(REPLACE
            "} // namespace mlx::core::rocm"
            "void clear_all_encoders() {\n  auto& devices = get_devices();\n  for (auto& [idx, dev] : devices) {\n    dev.clear_encoders();\n  }\n}\n\n} // namespace mlx::core::rocm"
            ROCM_CPP
            ROCM_CPP
        )
        # Add Device::clear_encoders() implementation (before CommandEncoder ctor)
        string(REPLACE
            "CommandEncoder::CommandEncoder(Device& d)"
            "void Device::clear_encoders() {\n  encoders_.clear();\n}\n\nCommandEncoder::CommandEncoder(Device& d)"
            ROCM_CPP
            ROCM_CPP
        )
        file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/rocm/device.cpp" ROCM_CPP)
    endif()

    # 3) Patch eval.cpp: add clear_streams() in gpu namespace
    file(READ "${mlx_SOURCE_DIR}/mlx/backend/rocm/eval.cpp" ROCM_EVAL)
    if(NOT ROCM_EVAL MATCHES "clear_streams")
        string(REPLACE
            "} // namespace mlx::core::gpu",
            "\nvoid clear_streams() {\n  rocm::clear_all_encoders();\n}\n\n} // namespace mlx::core::gpu"
            ROCM_EVAL
            ROCM_EVAL
        )
        file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/rocm/eval.cpp" ROCM_EVAL)
    endif()
endif()

# nlohmann/json (MLX may already provide this)
if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# stb (image loading)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(stb)

# Find libcurl for Hub client
find_package(CURL REQUIRED)

# tokenizers-cpp (HuggingFace tokenizers via Rust + C++ wrapper)
set(MSGPACK_USE_BOOST OFF CACHE BOOL "" FORCE)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp)

# minja — Jinja2 template engine for HF chat templates (header-only)
set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    minja
    GIT_REPOSITORY https://github.com/google/minja.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(minja)

# cpp-httplib — header-only HTTP server (OpenAI-compatible API)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# Global include path (enables #include <mlx-lm/...>)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core library (MLX wrappers and helpers)
add_library(mlx-lm-core
    src/core/module.cpp
)
target_link_libraries(mlx-lm-core PUBLIC mlx)
target_include_directories(mlx-lm-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Common library (shared infrastructure)
add_library(mlx-lm-common
    src/common/kv_cache.cpp
    src/common/attention_utils.cpp
    src/common/generate.cpp
    src/common/base_config.cpp
    src/common/hub_api.cpp
    src/common/safetensors.cpp
    src/common/switch_layers.cpp
    src/common/ssm_utils.cpp
    src/common/rope_utils.cpp
    src/common/chat_session.cpp
    src/common/tool_calling.cpp
    src/common/tokenizer.cpp
    src/common/quantize_utils.cpp
    src/common/chat_template.cpp
    src/common/gated_delta.cpp
)
target_link_libraries(mlx-lm-common PUBLIC
    mlx-lm-core
    nlohmann_json::nlohmann_json
    CURL::libcurl
    tokenizers_cpp
)
target_include_directories(mlx-lm-common PUBLIC ${minja_SOURCE_DIR}/include)

# Propagate ROCm flag as compile definition so C++ code can use #if defined(MLX_BUILD_ROCM)
if(MLX_BUILD_ROCM)
    target_compile_definitions(mlx-lm-common PUBLIC MLX_BUILD_ROCM)
endif()

# LLM library
add_library(mlx-lm-llm
    src/llm/llm_factory.cpp
    src/llm/models/llama.cpp
    src/llm/models/qwen2.cpp
    src/llm/models/gemma.cpp
    src/llm/models/phi.cpp
    src/llm/models/phi3.cpp
    src/llm/models/qwen3.cpp
    src/llm/models/gemma2.cpp
    src/llm/models/cohere.cpp
    src/llm/models/starcoder2.cpp
    src/llm/models/qwen3_moe.cpp
    src/llm/models/qwen3_next.cpp
    src/llm/models/qwen35_moe.cpp
    src/llm/models/mistral3_text.cpp
    src/llm/models/deepseek_v3.cpp
    src/llm/models/mimo.cpp
    src/llm/models/granite.cpp
    src/llm/models/glm4.cpp
    src/llm/models/ernie4_5.cpp
    src/llm/models/smollm3.cpp
    src/llm/models/minicpm.cpp
    src/llm/models/olmo2.cpp
    src/llm/models/olmo3.cpp
    src/llm/models/nanochat.cpp
    src/llm/models/lille130m.cpp
    src/llm/models/internlm2.cpp
    src/llm/models/exaone4.cpp
    src/llm/models/gemma3_text.cpp
    src/llm/models/apertus.cpp
    src/llm/models/openelm.cpp
    src/llm/models/phimoe.cpp
    src/llm/models/olmoe.cpp
    src/llm/models/glm4_moe.cpp
    src/llm/models/bailing_moe.cpp
    src/llm/models/afmoe.cpp
    src/llm/models/glm4_moe_lite.cpp
    src/llm/models/gptoss.cpp
    src/llm/models/lfm2_moe.cpp
    src/llm/models/gemma3n_text.cpp
    src/llm/models/jamba.cpp
    src/llm/models/baichuan_m1.cpp
    src/llm/models/falcon_h1.cpp
    src/llm/models/lfm2.cpp
    src/llm/models/nemotron_h.cpp
    src/llm/models/granite_moe_hybrid.cpp
)
target_link_libraries(mlx-lm-llm PUBLIC mlx-lm-common)

# Embedders library
add_library(mlx-lm-embedders
    src/embedders/pooling.cpp
    src/embedders/bert.cpp
    src/embedders/nomic_bert.cpp
    src/embedders/qwen3_embed.cpp
    src/embedders/embedder_factory.cpp
)
target_link_libraries(mlx-lm-embedders PUBLIC mlx-lm-common)

# VLM library
add_library(mlx-lm-vlm
    src/vlm/media_processing.cpp
    src/vlm/qwen_vl_utils.cpp
    src/vlm/models/qwen2_vl.cpp
    src/vlm/models/paligemma.cpp
    src/vlm/models/idefics3.cpp
    src/vlm/models/qwen25_vl.cpp
    src/vlm/models/gemma3.cpp
    src/vlm/models/qwen3_vl.cpp
    src/vlm/models/pixtral.cpp
    src/vlm/models/mistral3.cpp
    src/vlm/models/lfm2_vl.cpp
    src/vlm/models/fastvlm.cpp
    src/vlm/vlm_factory.cpp
)
target_link_libraries(mlx-lm-vlm PUBLIC mlx-lm-common)

# stb include path (header-only)
target_include_directories(mlx-lm-common PUBLIC ${stb_SOURCE_DIR})

if(MLX_LM_BUILD_EXAMPLES)
    add_executable(chat examples/chat.cpp)
    target_link_libraries(chat PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core)

    add_executable(diagnose examples/diagnose.cpp)
    target_link_libraries(diagnose PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core)

    add_executable(test_gdn examples/test_gdn.cpp)
    target_link_libraries(test_gdn PRIVATE mlx)

    add_executable(server
        examples/server.cpp
        src/common/server.cpp
        src/common/model_manager.cpp
    )
    target_link_libraries(server PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core httplib::httplib)
endif()

if(MLX_LM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
