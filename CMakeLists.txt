cmake_minimum_required(VERSION 3.20)
project(BareMetalComputeProbe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

include(FetchContent)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.1.0
)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# Offline SPIR-V compilation (CONVENTIONS.md §2: Runtime compilation is BANNED)
find_program(GLSLC glslc REQUIRED)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/probe.spv
    COMMAND ${GLSLC} ${CMAKE_SOURCE_DIR}/shaders/probe.comp -o ${CMAKE_BINARY_DIR}/shaders/probe.spv
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/probe.comp
    COMMENT "Compiling compute shader: probe.comp -> probe.spv"
)

add_custom_target(Shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/shaders/probe.spv)

include_directories(${Vulkan_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/core)

add_executable(BareMetalComputeProbe main.cpp)
add_dependencies(BareMetalComputeProbe Shaders)

target_compile_definitions(BareMetalComputeProbe PRIVATE 
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan VulkanMemoryAllocator)
