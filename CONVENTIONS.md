# Project Conventions

## Project Overview
This project is a renderer engine designed for the analysis and visualization of `.npy` files, specifically tailored for 3D Gaussian Splatting (3DGS) data. This project serves as a learning vehicle for mastering the Vulkan API.

## Coding Standards

### Language Standard
- All code must strictly adhere to the C++20 standard.

### Memory Management
No Raw Pointers: The use of raw pointers (`*`) for ownership or resource management is strictly prohibited.
RAII: Resource Acquisition Is Initialization (RAII) must be used for all resource management.
Smart Pointers: Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) when dynamic allocation of resources is necessary.

### Vulkan API Usage
No Raw C API: Do not use the raw C Vulkan API.
Vulkan.hpp: You must use the C++ bindings provided by `vulkan.hpp` for all Vulkan-related operations.

### Safety Net
The `VK_LAYER_KHRONOS_validation` layer must be explicitly enabled. If the validation layer cannot be detected, the program must throw a `std::runtime_error` and crash hard; silent failure is strictly prohibited.

### Build System
A modern `CMakeLists.txt` file must be generated alongside the project, automatically linking `glfw3` and `vulkan`.


Dependency Management: Absolutely prohibit the use of vcpkg/conan; third-party libraries must only use CMake FetchContent.
Language Standard: Enforce the C++20 standard. Disable raw pointers; resource lifecycles must be strictly managed by RAII (such as std::unique_ptr or custom handles).

Vulkan Pipeline: All Vulkan API calls must be wrapped in the `VK_CHECK` macro.

Memory Alignment: Gaussian sphere structs passed between Compute Shader and C++ must strictly adhere to the std430 memory layout; implicit padding is prohibited.

Code Style: Never generate "TODO" or placeholder code. If you are unsure, stop immediately and report an error; guessing is strictly prohibited.



GPU Memory Management (VRAM Management): Direct calls to `vk::Device::allocateMemory` are strictly prohibited. All memory allocations must be sub-allocated via VMA (Vulkan Memory Allocator).

Dynamic Dispatch: The global macro `#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1` is mandatory. Static linking jumps in Vulkan functions are prohibited; `vk::DispatchLoaderDynamic` must be used to handle all API calls to maximize driver layer performance.

Math & Coordinate Space: `#define GLM_FORCE_DEPTH_ZERO_TO_ONE` and `#define GLM_FORCE_LEFT_HANDED` are mandatory. Vulkan's NDC (Normalized Device Coordinate) depth is 0 to 1, with the Y-axis pointing downwards, which is the opposite of OpenGL.

The Shader Compilation pipeline prohibits dynamic compilation of GLSL/HLSL using shaderc at the C++ runtime. All Compute Shaders must be compiled offline to SPIR-V bytecode using glslc during the build-time phase via a CMake custom target. The engine is only responsible for reading .spv binary streams.