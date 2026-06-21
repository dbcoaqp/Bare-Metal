cmake_minimum_required(VERSION 3.20)
project(BareMetalComputeProbe)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIRS})

add_executable(BareMetalComputeProbe main.cpp)

target_compile_definitions(BareMetalComputeProbe PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan)



