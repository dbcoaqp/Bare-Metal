cmake_minimum_required(VERSION 3.20)
project(BareMetalComputeProbe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_subdirectory(third_party/VMA)


set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw)


# Offline SPIR-V compilation (CONVENTIONS.md §2: Runtime compilation is BANNED)
find_program(GLSLC glslc REQUIRED)

set(SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/calc_depth.comp")
set(SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/calc_depth.spv")

set(RADIX_HISTOGRAM_SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/radix_histogram.comp")
set(RADIX_HISTOGRAM_SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/radix_histogram.spv")

set(RADIX_SCAN_SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/radix_scan.comp")
set(RADIX_SCAN_SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/radix_scan.spv")

set(RADIX_SCATTER_SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/radix_scatter.comp")
set(RADIX_SCATTER_SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/radix_scatter.spv")

set(PREPROCESS_SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/preprocess.comp")
set(PREPROCESS_SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/preprocess.spv")

set(RASTERIZE_SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/rasterize.comp")
set(RASTERIZE_SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/rasterize.spv")

add_custom_command(
    OUTPUT ${SHADER_BIN}
    COMMAND ${GLSLC} ${SHADER_SRC} -o ${SHADER_BIN}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader: calc_depth.comp -> calc_depth.spv"
)

add_custom_command(
    OUTPUT ${RADIX_HISTOGRAM_SHADER_BIN}
    COMMAND ${GLSLC} ${RADIX_HISTOGRAM_SHADER_SRC} -o ${RADIX_HISTOGRAM_SHADER_BIN}
    DEPENDS ${RADIX_HISTOGRAM_SHADER_SRC}
    COMMENT "Compiling compute shader: radix_histogram.comp -> radix_histogram.spv"
)

add_custom_command(
    OUTPUT ${RADIX_SCAN_SHADER_BIN}
    COMMAND ${GLSLC} ${RADIX_SCAN_SHADER_SRC} -o ${RADIX_SCAN_SHADER_BIN}
    DEPENDS ${RADIX_SCAN_SHADER_SRC}
    COMMENT "Compiling compute shader: radix_scan.comp -> radix_scan.spv"
)

add_custom_command(
    OUTPUT ${RADIX_SCATTER_SHADER_BIN}
    COMMAND ${GLSLC} ${RADIX_SCATTER_SHADER_SRC} -o ${RADIX_SCATTER_SHADER_BIN}
    DEPENDS ${RADIX_SCATTER_SHADER_SRC}
    COMMENT "Compiling compute shader: radix_scatter.comp -> radix_scatter.spv"
)

add_custom_command(
    OUTPUT ${PREPROCESS_SHADER_BIN}
    COMMAND ${GLSLC} ${PREPROCESS_SHADER_SRC} -o ${PREPROCESS_SHADER_BIN}
    DEPENDS ${PREPROCESS_SHADER_SRC}
    COMMENT "Compiling compute shader: preprocess.comp -> preprocess.spv"
)

add_custom_command(
    OUTPUT ${RASTERIZE_SHADER_BIN}
    COMMAND ${GLSLC} ${RASTERIZE_SHADER_SRC} -o ${RASTERIZE_SHADER_BIN}
    DEPENDS ${RASTERIZE_SHADER_SRC}
    COMMENT "Compiling compute shader: rasterize.comp -> rasterize.spv"
)

add_custom_target(CompileShaders ALL DEPENDS 
    ${SHADER_BIN} 
    ${RADIX_HISTOGRAM_SHADER_BIN} 
    ${RADIX_SCAN_SHADER_BIN} 
    ${RADIX_SCATTER_SHADER_BIN} 
    ${PREPROCESS_SHADER_BIN} 
    ${RASTERIZE_SHADER_BIN})

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/core)

# NVTX v3 is header-only — add its default install path for Nsight Systems profiling
include_directories("C:/Program Files/NVIDIA Corporation/NvToolsExt/include")
link_directories("C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64")
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")

add_executable(BareMetalComputeProbe main.cpp)
add_dependencies(BareMetalComputeProbe CompileShaders)

target_compile_definitions(BareMetalComputeProbe PRIVATE 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan VulkanMemoryAllocator glfw nvToolsExt64_1.lib)
