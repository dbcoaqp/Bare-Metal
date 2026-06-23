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


# Offline Compilation

# 1. Look for the compiler comes with the Vulkan SDK (if cannot be found, CMake will directly report an error and prevent compilation).
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# 2. Define input/output paths
set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/shaders/probe.comp")
set(SHADER_BIN "${CMAKE_CURRENT_SOURCE_DIR}/shaders/probe.spv")

# 3. Establish custom compilation rules (physical mechanism: call glslc to bake .comp to .spv)
add_custom_command(
    OUTPUT ${SHADER_BIN}
    COMMAND ${GLSLC_EXECUTABLE} -O ${SHADER_SRC} -o ${SHADER_BIN}
    DEPENDS ${SHADER_SRC}
    COMMENT "[Aegis AOT] Baking Compute Shader into SPIR-V binary..."
)

# 4. Create a target and attach the rule above to the build pipeline.
add_custom_target(CompileShaders ALL DEPENDS ${SHADER_BIN})


add_executable(BareMetalComputeProbe main.cpp)

# 5. Force the main program to compile with a Shader dependency.
# C++ code compilation must not begin until the shader has been fully compiled.
add_dependencies(BareMetalComputeProbe CompileShaders)

target_compile_definitions(BareMetalComputeProbe PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
target_link_libraries(BareMetalComputeProbe PRIVATE Vulkan::Vulkan VulkanMemoryAllocator)