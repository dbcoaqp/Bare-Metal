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

include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(BareMetalComputeProbe main.cpp)

target_compile_definitions(BareMetalComputeProbe PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan VulkanMemoryAllocator)
