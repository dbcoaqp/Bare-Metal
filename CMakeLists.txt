cmake_minimum_required(VERSION 3.20)
project(BareMetalComputeProbe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

add_subdirectory(third_party/VMA)


# Offline SPIR-V compilation (CONVENTIONS.md §2: Runtime compilation is BANNED)
find_program(GLSLC glslc REQUIRED)

set(SHADER_SRC "${CMAKE_SOURCE_DIR}/shaders/calc_depth.comp")
set(SHADER_BIN "${CMAKE_SOURCE_DIR}/shaders/calc_depth.spv")

add_custom_command(
    OUTPUT ${SHADER_BIN}
    COMMAND ${GLSLC} ${SHADER_SRC} -o ${SHADER_BIN}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling compute shader: calc_depth.comp -> calc_depth.spv"
)

add_custom_target(CompileShaders ALL DEPENDS ${SHADER_BIN})

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/core)

add_executable(BareMetalComputeProbe main.cpp)
add_dependencies(BareMetalComputeProbe CompileShaders)

target_compile_definitions(BareMetalComputeProbe PRIVATE 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan VulkanMemoryAllocator)
