cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0054 NEW)

option(DECODER_CUDA "Enabling Nvidia CUDA of LDPC decoder" OFF)
option(DECODER_HIP "Enabling Nvidia HIP of LDPC decoder" OFF)

set(PROJECT_LANGUAGES CXX)

# ---------------------------------------------------------------------------
# CUDA object library — compiled separately so NVCC flags are isolated
# ---------------------------------------------------------------------------

# Creating the object file for modules
file(GLOB_RECURSE module_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/Module/*)
add_library(obj-module OBJECT ${module_source_files})
target_include_directories(obj-module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Module/Channel)
target_include_directories(obj-module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Module/Decoder_gpu)
set_target_properties(obj-module PROPERTIES
    POSITION_INDEPENDENT_CODE ON 
)
target_link_libraries(obj-module PUBLIC aff3ct::aff3ct-static-lib)


if(DECODER_CUDA)
    list(APPEND PROJECT_LANGUAGES CUDA)
    file(GLOB_RECURSE cuda_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*)
    add_library(obj-cuda OBJECT ${cuda_source_files})
    target_include_directories(obj-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Cuda)
    set_target_properties(obj-cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON 
    )
    target_link_libraries(obj-cuda PUBLIC aff3ct::aff3ct-static-lib)

    # --- ADDED: Guard definitions for CUDA targets ---
    target_compile_definitions(obj-cuda PRIVATE DECODER_CUDA)
    target_compile_definitions(obj-module PRIVATE DECODER_CUDA)
endif()


if(DECODER_HIP)
    file(GLOB_RECURSE hip_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*)
    add_library(obj-hip OBJECT ${hip_source_files})
    target_include_directories(obj-hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(obj-hip PROPERTIES
        POSITION_INDEPENDENT_CODE ON 
    )
    target_link_libraries(obj-hip PUBLIC aff3ct::aff3ct-static-lib)
    list(APPEND PROJECT_LANGUAGES HIP)

    # --- ADDED: Guard definitions for HIP targets ---
    target_compile_definitions(obj-hip PRIVATE DECODER_HIP)
    target_compile_definitions(obj-module PRIVATE DECODER_HIP)
endif()

project(pipeline_cpu ${PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH bin/)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(cpptrace CONFIG 0.4.1)
find_package(StreamPU REQUIRED)
find_package(AFF3CT  REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
set(link-objects $<TARGET_OBJECTS:obj-module>)

if(DECODER_CUDA)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-cuda>)
endif()

if(DECODER_HIP)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-hip>)
endif()

add_executable(pipeline_cpu
    ${link-objects}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)


add_executable(pipeline_gpu
    ${link-objects}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main_gpu.cpp
)

target_include_directories(pipeline_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(pipeline_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(pipeline_cpu PRIVATE
    aff3ct::aff3ct-static-lib
    Threads::Threads
)

target_link_libraries(pipeline_gpu PRIVATE
    aff3ct::aff3ct-static-lib
    Threads::Threads
)

# --- ADDED: Guard definitions for the main executable ---
if(DECODER_CUDA)
    target_compile_definitions(pipeline_cpu PRIVATE DECODER_CUDA)
    target_compile_definitions(pipeline_gpu PRIVATE DECODER_CUDA)
endif()

if(DECODER_HIP)
    target_compile_definitions(pipeline_cpu PRIVATE DECODER_HIP)
    target_compile_definitions(pipeline_gpu PRIVATE DECODER_HIP)
endif()
