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
    # Repo + branch — always build against the latest ROCm backend work.
    GIT_REPOSITORY https://github.com/NripeshN/mlx.git
    GIT_TAG 6abf0b7e # rocm-support (pinned working ExecUpdate commit)
    GIT_SHALLOW    FALSE
)
# Fetch MLX, apply local patches, then add it. Patching must happen before
# add_subdirectory()/FetchContent_MakeAvailable so CMakeLists.txt changes (for
# example removing unsupported ROCm clang flags) affect generated build files.
FetchContent_GetProperties(mlx)
if(NOT mlx_POPULATED)
    FetchContent_Populate(mlx)
endif()
set(MLX_SOURCE_DIR "${mlx_SOURCE_DIR}")

if(MLX_BUILD_ROCM AND MLX_SOURCE_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx-rocm-build.patch")
        execute_process(
            COMMAND git apply --check "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx-rocm-build.patch"
            WORKING_DIRECTORY "${MLX_SOURCE_DIR}"
            RESULT_VARIABLE PATCH_CHECK_RESULT
            ERROR_QUIET
            OUTPUT_QUIET
        )
        if(PATCH_CHECK_RESULT EQUAL 0)
            message(STATUS "Applying mlx-rocm-build.patch...")
            execute_process(
                COMMAND git apply "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx-rocm-build.patch"
                WORKING_DIRECTORY "${MLX_SOURCE_DIR}"
                RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_RESULT EQUAL 0)
                message(STATUS "Patch applied successfully")
            else()
                message(FATAL_ERROR "Failed to apply mlx-rocm-build.patch")
            endif()
        else()
            execute_process(
                COMMAND git apply --reverse --check "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx-rocm-build.patch"
                WORKING_DIRECTORY "${MLX_SOURCE_DIR}"
                RESULT_VARIABLE PATCH_REVERSE_CHECK_RESULT
                ERROR_QUIET
                OUTPUT_QUIET
            )
            if(PATCH_REVERSE_CHECK_RESULT EQUAL 0)
                message(STATUS "mlx-rocm-build.patch already applied, skipping")
            else()
                message(FATAL_ERROR "mlx-rocm-build.patch does not apply to fetched MLX source")
            endif()
        endif()
endif()

add_subdirectory("${mlx_SOURCE_DIR}" "${mlx_BINARY_DIR}")

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
    src/common/gguf_loader.cpp
    src/common/registry.cpp
    src/common/switch_layers.cpp
    src/common/ssm_utils.cpp
    src/common/rope_utils.cpp
    src/common/chat_session.cpp
    src/common/tool_calling.cpp
    src/common/tokenizer.cpp
    src/common/quantize_utils.cpp
    src/common/chat_template.cpp
    src/common/gated_delta.cpp
    src/common/graph_decode.cpp
    src/common/gpu_stubs.cpp
    src/llm/models/mtp_head.cpp
    src/llm/models/mtp_moe.cpp
)
# NOTE: src/common/mtp_delta_kernel.cpp intentionally excluded from build.
# Contains dead code (mtp_delta_fused / mtp_draft_forward) not wired into
# generate.cpp. Left on disk for future ROCm optimization work.
target_link_libraries(mlx-lm-common PUBLIC
    mlx-lm-core
    nlohmann_json::nlohmann_json
    CURL::libcurl
    tokenizers_cpp
)
target_include_directories(mlx-lm-common PUBLIC ${minja_SOURCE_DIR}/include)
# Patched minja headers (capitalize filter, etc.) take precedence over the
# upstream minja version fetched by FetchContent.
target_include_directories(mlx-lm-common BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/patched
)

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
    src/llm/models/gemma4.cpp
    src/llm/models/jamba.cpp
    src/llm/models/baichuan_m1.cpp
    src/llm/models/falcon_h1.cpp
    src/llm/models/lfm2.cpp
    src/llm/models/nemotron_h.cpp
    src/llm/models/granite_moe_hybrid.cpp
    src/llm/models/bitnet.cpp
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
    src/vlm/models/qwen35_vl.cpp
    src/vlm/models/pixtral.cpp
    src/vlm/models/mistral3.cpp
    src/vlm/models/lfm2_vl.cpp
    src/vlm/models/fastvlm.cpp
    src/vlm/vlm_factory.cpp
)
target_link_libraries(mlx-lm-vlm PUBLIC mlx-lm-common)

# stb include path (header-only)
target_include_directories(mlx-lm-common PUBLIC ${stb_SOURCE_DIR})

# NPU backend (optional, requires XRT)
# NPU backend (optional, requires IRON Python stack + XRT)
if(MLX_LM_BUILD_NPU)
    # The NPU backend uses the IRON JIT via Python subprocess.
    # Install IRON: pip install mlir-aie
    
    # MLIR-AIE venv path for IRON JIT
    set(NPU_VENV_DIR "${CMAKE_SOURCE_DIR}/../mlir-aie/.venv")
    
    # Copy JIT helpers to build directory
    configure_file(
        src/npu/kernels/ternary_gemv.py
        ${CMAKE_BINARY_DIR}/bin/ternary_gemv.py
        COPYONLY
    )
    
    # Find LLVM-AIE compiler
    find_program(AIE2_CLANG clang++
        PATHS "${NPU_VENV_DIR}/lib/python3.14/site-packages/llvm-aie/bin"
        NO_DEFAULT_PATH
    )
    if(NOT AIE2_CLANG)
        message(STATUS "NPU: LLVM-AIE clang not found, kernel will be JIT-compiled at runtime")
    else()
        # Compile the AIE kernel at build time
        set(AIE_KERNEL_SRC "${CMAKE_SOURCE_DIR}/src/npu/kernels/ternary_gemv_aie.cpp")
        set(AIE_KERNEL_OBJ "${CMAKE_BINARY_DIR}/kernels/ternary_gemv_aie.o")
        add_custom_command(
            OUTPUT ${AIE_KERNEL_OBJ}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/kernels"
            COMMAND ${AIE2_CLANG} --target=aie2-none-unknown-elf -O2 -std=c++20
                -c "${AIE_KERNEL_SRC}" -o "${AIE_KERNEL_OBJ}"
            DEPENDS ${AIE_KERNEL_SRC}
            COMMENT "Compiling AIE2 kernel: ternary_gemv_aie"
        )
        add_custom_target(aie_kernels ALL DEPENDS ${AIE_KERNEL_OBJ})
    endif()
    
    add_library(mlx-lm-npu STATIC
        src/npu/npu_backend.cpp
    )
    target_include_directories(mlx-lm-npu PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_definitions(mlx-lm-npu PUBLIC
        MLX_BUILD_NPU
        NPU_INSTALL_DIR="${CMAKE_BINARY_DIR}"
    )
    if(AIE2_CLANG)
        add_dependencies(mlx-lm-npu aie_kernels)
    endif()
    message(STATUS "NPU backend enabled (JIT path)")
    if(AIE2_CLANG)
        message(STATUS "  AIE2 compiler: ${AIE2_CLANG}")
    endif()
endif()

if(MLX_LM_BUILD_EXAMPLES)
    # Compile the float32 strided_scan force-instantiation stub with hipcc so
    # the HIP fat binary registration constructor is linked properly.
    if(MLX_BUILD_ROCM)
        set(FORCE_F32_SCAN_SRC "${CMAKE_SOURCE_DIR}/src/npu/force_f32_scan.hip")
        set(FORCE_F32_SCAN_OBJ "${CMAKE_BINARY_DIR}/force_f32_scan.o")
        add_custom_command(
            OUTPUT ${FORCE_F32_SCAN_OBJ}
            COMMAND ${CMAKE_HIP_COMPILER} -c --offload-arch=gfx1151 -O3 -fPIC
                -I"${CMAKE_SOURCE_DIR}/build/_deps/mlx-src"
                -I/opt/rocm-7.2.4/include
                -o ${FORCE_F32_SCAN_OBJ}
                ${FORCE_F32_SCAN_SRC}
            DEPENDS ${FORCE_F32_SCAN_SRC}
            COMMENT "Compiling force_f32_scan.hip")
        add_custom_target(force_f32_scan DEPENDS ${FORCE_F32_SCAN_OBJ})
        set(FORCE_F32_SCAN_OBJ_REF ${FORCE_F32_SCAN_OBJ})
    endif()

    add_executable(chat examples/chat.cpp)
    target_link_libraries(chat PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core)
    if(MLX_BUILD_ROCM)
        find_package(hip QUIET)
        target_compile_definitions(chat PRIVATE MLX_BUILD_ROCM)
        target_link_libraries(chat PRIVATE hip::host ${FORCE_F32_SCAN_OBJ_REF})
        add_dependencies(chat force_f32_scan)
    endif()
    if(MLX_LM_BUILD_NPU AND TARGET mlx-lm-npu)
        target_link_libraries(chat PRIVATE mlx-lm-npu)
        target_compile_definitions(chat PRIVATE MLX_BUILD_NPU)
    endif()

    add_executable(diagnose examples/diagnose.cpp)
    target_link_libraries(diagnose PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core)
    if(MLX_BUILD_ROCM)
        target_link_libraries(diagnose PRIVATE ${FORCE_F32_SCAN_OBJ_REF})
        add_dependencies(diagnose force_f32_scan)
    endif()

    # convert: quantize a bf16 Qwen3.6-A3B (+ inline MTP head) to a combined MLX
    # model with per-component mixed precision. Uses the engine's own MLX build.
    add_executable(convert examples/convert.cpp)
    target_link_libraries(convert PRIVATE mlx-lm-llm mlx-lm-common mlx-lm-core)

    add_executable(test_gdn examples/test_gdn.cpp)
    target_link_libraries(test_gdn PRIVATE mlx)

    add_executable(test_device_kv examples/test_device_kv.cpp)
    target_link_libraries(test_device_kv PRIVATE mlx)

    add_executable(test_device_rope examples/test_device_rope.cpp)
    target_link_libraries(test_device_rope PRIVATE mlx)

    add_executable(test_qbits examples/test_qbits.cpp)
    target_link_libraries(test_qbits PRIVATE mlx)

    add_executable(test_dynslice examples/test_dynslice.cpp)
    target_link_libraries(test_dynslice PRIVATE mlx)

    if(MLX_BUILD_ROCM)
        # ROCm-only: link the decode-arena bridge (undefined on CPU/Metal).
        add_executable(test_donate examples/test_donate.cpp)
        target_link_libraries(test_donate PRIVATE mlx)

        add_executable(test_arena examples/test_arena.cpp)
        target_link_libraries(test_arena PRIVATE mlx-lm-common)
    endif()

    add_executable(test_sdpa_ref examples/test_sdpa_ref.cpp)
    target_link_libraries(test_sdpa_ref PRIVATE mlx)

    if(MLX_LM_BUILD_NPU AND TARGET mlx-lm-npu)
        add_executable(test_npu examples/test_npu.cpp)
        target_link_libraries(test_npu PRIVATE mlx-lm-npu)
    endif()

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
