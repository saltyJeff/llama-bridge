cmake_minimum_required(VERSION 3.21)
project(LlamaBridge)
set(CMAKE_CXX_STANDARD 17)

# Temporary monkey patch until this is merged: https://github.com/ggml-org/llama.cpp/pull/23215/changes
if(WIN32 AND DEFINED ENV{VULKAN_SDK})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
endif()
find_package(SPIRV-Headers CONFIG REQUIRED)

include(FetchContent)
set(GGML_VULKAN ON)
set(GGML_CUDA ON)
set(GGML_CPU_ALL_VARIANTS ON)
set(BUILD_SHARED_LIBS ON)
set(GGML_BACKEND_DL ON)

FetchContent_Declare(
  llama_cpp
  URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9209.tar.gz
)
FetchContent_MakeAvailable(llama_cpp)

# Distribution settings
set(DIST_DIR "${CMAKE_BINARY_DIR}/dist")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${DIST_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${DIST_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${DIST_DIR}")

add_library(llama_bridge SHARED
    llama_bridge.h
    llama_bridge.cpp
    nlohmann/json.hpp
    minja.hpp
)
target_compile_definitions(llama_bridge PRIVATE LLAMA_BRIDGE_EXPORTS)
target_link_libraries(llama_bridge PUBLIC llama ggml)
target_include_directories(llama_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(llama_bridge_driver main.cpp)
# download Qwen 3.5 MTP to this directory
target_compile_definitions(llama_bridge_driver PRIVATE MODEL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/Qwen3.5-4B-UD-Q4_K_XL.gguf")
target_link_libraries(llama_bridge_driver PUBLIC llama_bridge)

# Find CUDA bin dir if enabled
set(CUDA_COPY_DIR "")
if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CUDA_COPY_DIR "${CUDAToolkit_BIN_DIR}")
    endif()
endif()

# Deploy files to dist folder after build
add_custom_command(TARGET llama_bridge_driver POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:llama_bridge_driver>"
    # Copy dynamic backends and CUDA DLLs via helper script (handles build-time globbing)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/llama_bridge.h" "$<TARGET_FILE_DIR:llama_bridge_driver>/"
    COMMAND ${CMAKE_COMMAND} 
        -D "DIST_DIR=$<TARGET_FILE_DIR:llama_bridge_driver>" 
        -D "BIN_DIR=$<TARGET_FILE_DIR:llama>" 
        -D "CUDA_BIN_DIR=${CUDA_COPY_DIR}" 
        -P "${CMAKE_CURRENT_SOURCE_DIR}/copy_dist.cmake"
    COMMAND_EXPAND_LISTS
    COMMENT "Syncing artifacts to $<TARGET_FILE_DIR:llama_bridge_driver>..."
)

# Configure Visual Studio to run from the dist folder
set_target_properties(llama_bridge_driver PROPERTIES 
    VS_DEBUGGER_COMMAND "$<TARGET_FILE:llama_bridge_driver>"
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:llama_bridge_driver>"
)

# Target to create a zip archive of the dist folder
add_custom_target(archive-dist
    COMMAND ${CMAKE_COMMAND} -E tar "cfv" "${CMAKE_BINARY_DIR}/llama-bridge-dist.zip" --format=zip "$<TARGET_FILE_DIR:llama_bridge_driver>"
    DEPENDS llama_bridge_driver
    COMMENT "Creating distribution archive llama-bridge-dist.zip..."
)
