# CMake integration for crusoecloud/fastokens via a thin C FFI staticlib.
#
# Produces:
#   - fastokens_c  (INTERFACE)  → links libfastokens_c.a + system deps
#   - fastokens_cpp (INTERFACE) → public include path for fastokens_c.h

cmake_minimum_required(VERSION 3.20)
project(fastokens_cpp C CXX)

set(FASTOKENS_CPP_CARGO_TARGET "")
set(FASTOKENS_C_LINK_LIBS "")
set(CARGO_EXTRA_ENVS "")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND FASTOKENS_C_LINK_LIBS ${CMAKE_DL_LIBS} pthread m)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(FASTOKENS_CPP_CARGO_TARGET aarch64-unknown-linux-gnu)
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(FASTOKENS_CPP_CARGO_TARGET aarch64-apple-darwin)
    endif()
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CARGO_EXTRA_ENVS MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    endif()
    find_library(SECURITY_LIB Security)
    find_library(COREFOUNDATION_LIB CoreFoundation)
    find_library(ICONV_LIB iconv)
    if(SECURITY_LIB)
        list(APPEND FASTOKENS_C_LINK_LIBS ${SECURITY_LIB})
    endif()
    if(COREFOUNDATION_LIB)
        list(APPEND FASTOKENS_C_LINK_LIBS ${COREFOUNDATION_LIB})
    endif()
    if(ICONV_LIB)
        list(APPEND FASTOKENS_C_LINK_LIBS ${ICONV_LIB})
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    string(TOUPPER "${CMAKE_SYSTEM_PROCESSOR}" _proc_upper)
    if(_proc_upper STREQUAL "ARM64" OR _proc_upper STREQUAL "AARCH64")
        set(FASTOKENS_CPP_CARGO_TARGET aarch64-pc-windows-msvc)
    else()
        set(FASTOKENS_CPP_CARGO_TARGET x86_64-pc-windows-msvc)
    endif()
    unset(_proc_upper)
    list(APPEND FASTOKENS_C_LINK_LIBS
        ntdll wsock32 ws2_32 Bcrypt iphlpapi userenv psapi
    )
endif()

set(FASTOKENS_CPP_CARGO_FLAGS "")
set(FASTOKENS_CPP_CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FASTOKENS_CPP_CARGO_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(NOT FASTOKENS_CPP_CARGO_TARGET STREQUAL "")
    list(APPEND FASTOKENS_CPP_CARGO_FLAGS --target ${FASTOKENS_CPP_CARGO_TARGET})
    set(FASTOKENS_CPP_CARGO_BINARY_DIR
        "${FASTOKENS_CPP_CARGO_BINARY_DIR}/${FASTOKENS_CPP_CARGO_TARGET}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(FASTOKENS_CPP_CARGO_BINARY_DIR "${FASTOKENS_CPP_CARGO_BINARY_DIR}/debug")
else()
    list(APPEND FASTOKENS_CPP_CARGO_FLAGS --release)
    set(FASTOKENS_CPP_CARGO_BINARY_DIR "${FASTOKENS_CPP_CARGO_BINARY_DIR}/release")
endif()

get_filename_component(FASTOKENS_CPP_ROOT ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(FASTOKENS_CPP_CARGO_SOURCE_PATH ${FASTOKENS_CPP_ROOT}/rust)
set(FASTOKENS_CPP_INCLUDE ${FASTOKENS_CPP_ROOT}/include)

if(WIN32)
    set(FASTOKENS_RUST_LIB "${FASTOKENS_CPP_CARGO_BINARY_DIR}/fastokens_c.lib")
else()
    set(FASTOKENS_RUST_LIB "${FASTOKENS_CPP_CARGO_BINARY_DIR}/libfastokens_c.a")
endif()

# Locate cargo
if(CMAKE_HOST_WIN32)
    set(USER_HOME "$ENV{USERPROFILE}")
else()
    set(USER_HOME "$ENV{HOME}")
endif()
if(NOT DEFINED CARGO_HOME)
    if("$ENV{CARGO_HOME}" STREQUAL "")
        set(CARGO_HOME "${USER_HOME}/.cargo")
    else()
        set(CARGO_HOME "$ENV{CARGO_HOME}")
    endif()
endif()

find_program(CARGO_EXECUTABLE cargo
    HINTS "${CARGO_HOME}"
    PATH_SUFFIXES "bin")
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR
        "cargo not found. Install a Rust toolchain (https://rustup.rs) — "
        "required to build crusoecloud/fastokens.")
endif()
mark_as_advanced(CARGO_EXECUTABLE)

# Build the Rust staticlib (depends on all sources so edits rebuild)
file(GLOB_RECURSE FASTOKENS_RUST_SOURCES
    ${FASTOKENS_CPP_CARGO_SOURCE_PATH}/src/*.rs
    ${FASTOKENS_CPP_CARGO_SOURCE_PATH}/Cargo.toml
)

# PCRE2_SYS_STATIC=1 forces pcre2-sys to compile and archive PCRE2 into
# libfastokens_c.a so consumers do not need a system libpcre2-dev package.
add_custom_command(
    OUTPUT ${FASTOKENS_RUST_LIB}
    COMMAND
        ${CMAKE_COMMAND} -E env
        CARGO_TARGET_DIR=${FASTOKENS_CPP_CARGO_TARGET_DIR}
        PCRE2_SYS_STATIC=1
        ${CARGO_EXTRA_ENVS}
        ${CARGO_EXECUTABLE} build --locked ${FASTOKENS_CPP_CARGO_FLAGS}
    WORKING_DIRECTORY ${FASTOKENS_CPP_CARGO_SOURCE_PATH}
    DEPENDS ${FASTOKENS_RUST_SOURCES}
    COMMENT "Building fastokens C FFI (cargo)"
    VERBATIM
)

add_custom_target(fastokens_c_build DEPENDS ${FASTOKENS_RUST_LIB})

add_library(fastokens_c INTERFACE)
add_dependencies(fastokens_c fastokens_c_build)
target_link_libraries(fastokens_c INTERFACE
    ${FASTOKENS_RUST_LIB}
    ${FASTOKENS_C_LINK_LIBS}
)
target_include_directories(fastokens_c INTERFACE ${FASTOKENS_CPP_INCLUDE})

# Alias matching the previous tokenizers_cpp consumer name so call sites stay simple.
add_library(fastokens_cpp INTERFACE)
target_link_libraries(fastokens_cpp INTERFACE fastokens_c)
target_include_directories(fastokens_cpp INTERFACE ${FASTOKENS_CPP_INCLUDE})
