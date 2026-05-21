include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# Core tests
add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_types COMMAND test_types)

# KV Cache tests
add_executable(test_kv_cache test_kv_cache.cpp)
target_link_libraries(test_kv_cache PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_kv_cache COMMAND test_kv_cache)

# Config parsing tests
add_executable(test_config test_config.cpp)
target_link_libraries(test_config PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_config COMMAND test_config)

# Sampler tests (+ MTPHead smoke -- I7 sub-task 6, needs the LLM lib).
add_executable(test_generate test_generate.cpp)
target_link_libraries(test_generate PRIVATE mlx-lm-llm mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_generate COMMAND test_generate)

# Tool calling tests
add_executable(test_tool_calling test_tool_calling.cpp)
target_link_libraries(test_tool_calling PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_tool_calling COMMAND test_tool_calling)

# Chat session tests
add_executable(test_chat_session test_chat_session.cpp)
target_link_libraries(test_chat_session PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_chat_session COMMAND test_chat_session)

# User input and message generator tests
add_executable(test_user_input test_user_input.cpp)
target_link_libraries(test_user_input PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_user_input COMMAND test_user_input)

# RoPE utility tests
add_executable(test_rope_utils test_rope_utils.cpp)
target_link_libraries(test_rope_utils PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_rope_utils COMMAND test_rope_utils)

# Chat template tests
add_executable(test_chat_template test_chat_template.cpp)
target_link_libraries(test_chat_template PRIVATE mlx-lm-common Catch2::Catch2WithMain)
add_test(NAME test_chat_template COMMAND test_chat_template)

# NemotronH model tests
add_executable(test_nemotron_h test_nemotron_h.cpp)
target_link_libraries(test_nemotron_h PRIVATE mlx-lm-llm Catch2::Catch2WithMain)
add_test(NAME test_nemotron_h COMMAND test_nemotron_h)

# Server API integration tests
add_executable(test_server_api
    test_server_api.cpp
    ${CMAKE_SOURCE_DIR}/src/common/server.cpp
    ${CMAKE_SOURCE_DIR}/src/common/model_manager.cpp
)
target_link_libraries(test_server_api PRIVATE
    mlx-lm-llm mlx-lm-common mlx-lm-core httplib::httplib Catch2::Catch2WithMain
)
# Discovery tests (no model required, fast)
add_test(NAME test_server_api_discovery
    COMMAND test_server_api "[discovery]"
)
# Endpoint tests (no model loaded, tests error paths + /health + /v1/models)
add_test(NAME test_server_api_endpoints
    COMMAND test_server_api "[endpoints]"
)
# Inference tests (requires cached model, slow)
add_test(NAME test_server_api_inference
    COMMAND test_server_api "[inference]"
)
# Error handling tests
add_test(NAME test_server_api_errors
    COMMAND test_server_api "[errors]"
)
