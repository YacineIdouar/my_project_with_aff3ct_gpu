cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0054 NEW)

set(PROJECT_LANGUAGES CXX CUDA)
project(my_project ${PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH bin/)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(cpptrace CONFIG 0.4.1)
find_package(StreamPU REQUIRED)
find_package(AFF3CT CONFIG 4.1.1 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# CUDA object library — compiled separately so NVCC flags are isolated
# ---------------------------------------------------------------------------
file(GLOB_RECURSE cuda_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*)
add_library(obj-cuda OBJECT ${cuda_source_files})
target_include_directories(obj-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(obj-cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON  # needed when .cu files call extern "C" across TUs
)
target_link_libraries(obj-cuda PUBLIC aff3ct::aff3ct-static-lib)

# ---------------------------------------------------------------------------
# Executable
# $<TARGET_OBJECTS:obj-cuda> injects the compiled .o files from the OBJECT lib
# ---------------------------------------------------------------------------
add_executable(my_project
    $<TARGET_OBJECTS:obj-cuda>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(my_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(my_project PRIVATE
    aff3ct::aff3ct-static-lib
)