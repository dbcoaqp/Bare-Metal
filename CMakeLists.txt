cmake_minimum_required(VERSION 3.20)
project(VulkanRenderer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(VulkanRenderer main.cpp)

target_link_libraries(VulkanRenderer PRIVATE glfw Vulkan::Vulkan)
