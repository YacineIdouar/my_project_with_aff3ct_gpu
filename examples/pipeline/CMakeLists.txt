cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0054 NEW)

option(DECODER_CUDA "Enabling Nvidia CUDA of LDPC decoder" OFF)
option(DECODER_HIP "Enabling Nvidia HIP of LDPC decoder" OFF)
option(DECODER_SYCL "Enabling SYCL of LDPC decoder" OFF)
option(DECODER_VULKAN "Enabling Vulkan of LDPC decoder" OFF)

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


# ---------------------------------------------------------------------------
# SYCL object library — plain C++ compiled with a SYCL-aware compiler.
# Unlike CUDA/HIP, SYCL is not a separate CMake LANGUAGE: the whole
# CMAKE_CXX_COMPILER must already be a SYCL compiler (icpx for oneAPI, or
# acpp/syclcc for AdaptiveCpp), matching how StreamPU itself is configured
# (see SPU_ENABLE_OAPI_SYCL / SPU_ENABLE_ACPP_SYCL in streampu's CMakeLists).
# ---------------------------------------------------------------------------
if(DECODER_SYCL)
    set(SYCL_COMPILER "oneapi" CACHE STRING "SYCL compiler to target: oneapi or acpp")
    set(SYCL_BACKEND "" CACHE STRING "oneAPI -fsycl-targets backend (e.g. nvptx64-nvidia-cuda), required when SYCL_COMPILER=oneapi")

    file(GLOB_RECURSE sycl_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/sycl/*)
    add_library(obj-sycl OBJECT ${sycl_source_files})
    target_include_directories(obj-sycl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(obj-sycl PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(obj-sycl PUBLIC aff3ct::aff3ct-static-lib)

    if(SYCL_COMPILER STREQUAL "oneapi")
        if(SYCL_BACKEND STREQUAL "")
            message(FATAL_ERROR "SYCL_BACKEND is needed for the oneAPI SYCL_COMPILER (e.g. -DSYCL_BACKEND=nvptx64-nvidia-cuda)")
        endif()
        target_compile_options(obj-sycl PUBLIC -fsycl -fsycl-targets=${SYCL_BACKEND})
        target_link_options(obj-sycl PUBLIC -fsycl -fsycl-targets=${SYCL_BACKEND})
    elseif(SYCL_COMPILER STREQUAL "acpp")
        target_compile_options(obj-sycl PUBLIC --acpp-targets=generic)
        target_link_options(obj-sycl PUBLIC --acpp-targets=generic)
    else()
        message(FATAL_ERROR "Unknown SYCL_COMPILER '${SYCL_COMPILER}': expected 'oneapi' or 'acpp'")
    endif()

    # --- Guard definitions for SYCL targets ---
    target_compile_definitions(obj-sycl PRIVATE DECODER_SYCL)
    target_compile_definitions(obj-module PRIVATE DECODER_SYCL)
endif()

project(pipeline_cpu ${PROJECT_LANGUAGES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH bin/)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(DECODER_VULKAN)
    # StreamPU's exported config links spu::spu-*-lib against Vulkan::Vulkan without calling
    # find_dependency(Vulkan) itself (same issue fixed in aff3ct's own top-level CMakeLists.txt),
    # so Vulkan must be resolved here, before StreamPU/AFF3CT are, or their imported targets fail.
    find_package(Vulkan REQUIRED)
endif()

find_package(cpptrace CONFIG 0.4.1)
find_package(StreamPU REQUIRED)
find_package(AFF3CT  REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Vulkan object library — compiles the LDPC compute shaders (.comp) to SPIR-V
# with glslc and embeds them as C++ headers, same pattern StreamPU itself uses
# for the Matmul module's shader (see lib/aff3ct/lib/streampu/CMakeLists.txt).
# ---------------------------------------------------------------------------
if(DECODER_VULKAN)
    find_program(GLSLC_EXECUTABLE glslc HINTS "$ENV{VULKAN_SDK}/bin")
    if(NOT GLSLC_EXECUTABLE)
        message(FATAL_ERROR "pipeline - 'glslc' (Vulkan shader compiler) not found; install the Vulkan SDK "
                            "or make sure 'glslc' is available in the PATH")
    endif()

    set(VULKAN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
    set(VULKAN_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake")
    set(VULKAN_SHADER_NAMES ldpc_clip_channel ldpc_update_cn ldpc_update_vn ldpc_hard_decision)
    set(VULKAN_SHADER_HEADERS "")
    foreach(shader_name ${VULKAN_SHADER_NAMES})
        set(shader_src "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/Shaders/${shader_name}.comp")
        set(shader_spv "${VULKAN_GENERATED_DIR}/${shader_name}.spv")
        set(shader_header "${VULKAN_GENERATED_DIR}/${shader_name}_spv.hpp")
        add_custom_command(
            OUTPUT "${shader_header}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${VULKAN_GENERATED_DIR}"
            COMMAND ${GLSLC_EXECUTABLE} "${shader_src}" -o "${shader_spv}"
            COMMAND ${CMAKE_COMMAND} -DSPV_FILE=${shader_spv} -DOUT_FILE=${shader_header}
                    -DSYMBOL=${shader_name}_spv -P "${VULKAN_EMBED_SCRIPT}"
            DEPENDS "${shader_src}" "${VULKAN_EMBED_SCRIPT}"
            COMMENT "Compiling and embedding Vulkan shader ${shader_name}.comp"
        )
        list(APPEND VULKAN_SHADER_HEADERS "${shader_header}")
    endforeach()
    add_custom_target(pipeline-vulkan-shaders DEPENDS ${VULKAN_SHADER_HEADERS})

    file(GLOB_RECURSE vulkan_source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/*.cpp)
    add_library(obj-vulkan OBJECT ${vulkan_source_files})
    target_include_directories(obj-vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/Vulkan "${VULKAN_GENERATED_DIR}")
    add_dependencies(obj-vulkan pipeline-vulkan-shaders)
    set_target_properties(obj-vulkan PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(obj-vulkan PUBLIC aff3ct::aff3ct-static-lib Vulkan::Vulkan)

    target_compile_definitions(obj-vulkan PRIVATE DECODER_VULKAN)
    target_compile_definitions(obj-module PRIVATE DECODER_VULKAN)
endif()


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

if(DECODER_SYCL)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-sycl>)
endif()

if(DECODER_VULKAN)
    list(APPEND link-objects $<TARGET_OBJECTS:obj-vulkan>)
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

if(DECODER_SYCL)
    target_compile_definitions(pipeline_cpu PRIVATE DECODER_SYCL)
    target_compile_definitions(pipeline_gpu PRIVATE DECODER_SYCL)
    target_link_libraries(pipeline_cpu PRIVATE obj-sycl)
    target_link_libraries(pipeline_gpu PRIVATE obj-sycl)
    if(SYCL_COMPILER STREQUAL "oneapi")
        target_link_options(pipeline_cpu PRIVATE -fsycl -fsycl-targets=${SYCL_BACKEND})
        target_link_options(pipeline_gpu PRIVATE -fsycl -fsycl-targets=${SYCL_BACKEND})
    elseif(SYCL_COMPILER STREQUAL "acpp")
        target_link_options(pipeline_cpu PRIVATE --acpp-targets=generic)
        target_link_options(pipeline_gpu PRIVATE --acpp-targets=generic)
    endif()
endif()

if(DECODER_VULKAN)
    target_compile_definitions(pipeline_cpu PRIVATE DECODER_VULKAN)
    target_compile_definitions(pipeline_gpu PRIVATE DECODER_VULKAN)
    target_link_libraries(pipeline_cpu PRIVATE Vulkan::Vulkan)
    target_link_libraries(pipeline_gpu PRIVATE Vulkan::Vulkan)
endif()
