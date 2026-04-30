cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0054 NEW)


option(DECODER_CUDA "Enabling Nvidia CUDA of LDPC decoder" OFF)
option(DECODER_HIP "Enabling Nvidia HIP of LDPC decoder" OFF)

set(PROJECT_LANGUAGES CXX)


# ---------------------------------------------------------------------------
# CUDA object library — compiled separately so NVCC flags are isolated
# ---------------------------------------------------------------------------

if(DECODER_CUDA)
	list(APPEND PROJECT_LANGUAGES CUDA)
    file(GLOB_RECURSE cuda_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*)
    add_library(obj-cuda OBJECT ${cuda_source_files})
    target_include_directories(obj-cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(obj-cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON # needed when .cu files call extern "C" across TUs
    )
    target_link_libraries(obj-cuda PUBLIC aff3ct::aff3ct-static-lib)
endif()


if(DECODER_HIP)
    file(GLOB_RECURSE hip_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/hip/*)
    add_library(obj-hip OBJECT ${hip_source_files})
    target_include_directories(obj-hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(obj-hip PROPERTIES
        POSITION_INDEPENDENT_CODE ON # needed when .cu files call extern "C" across TUs
    )
    target_link_libraries(obj-hip PUBLIC aff3ct::aff3ct-static-lib)
	list(APPEND PROJECT_LANGUAGES HIP)
endif()

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
# Executable
# $<TARGET_OBJECTS:obj-cuda> injects the compiled .o files from the OBJECT lib
# ---------------------------------------------------------------------------
set(link-objects)

if(DECODER_CUDA)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-cuda>)
endif()

if(DECODER_HIP)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-hip>)
endif()

add_executable(my_project
    ${link-objects}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(my_project PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(my_project PRIVATE
    aff3ct::aff3ct-static-lib
)
