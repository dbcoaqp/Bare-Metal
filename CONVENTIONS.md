# Project Bare Metal: System Architecture & Coding Conventions

## 1. System Baseline & C++ Standards
* **Language Standard**: Strictly C++20.
* **Resource Ownership**: Raw pointers (`*`) are strictly PROHIBITED for resource management or ownership. Resource lifecycles must be entirely governed by RAII (e.g., `std::unique_ptr`, `vk::UniqueHandle`).
* **Deterministic Code Generation**: Never generate "TODO", placeholder, or pseudo-code. If the implementation is unclear, HALT and report the error immediately. Guessing is strictly forbidden.

## 2. Build Pipeline & Dependency Management
* **Build System**: Modern CMake (`CMakeLists.txt`) exclusively.
* **Dependency Injection**: Use CMake `FetchContent` for all third-party libraries. Package managers like `vcpkg` or `conan` are strictly PROHIBITED to ensure a completely self-contained build loop.
* **Shader Compilation (AOT Pipeline)**: Runtime compilation of GLSL/HLSL (e.g., using `shaderc`) is completely BANNED. All Compute Shaders must be compiled offline into `SPIR-V` (`.spv`) binaries using `glslc` via CMake `add_custom_command` during build-time. The C++ engine acts only as a binary reader.

## 3. Vulkan Execution Paradigm
* **C++ Bindings Only**: The raw C Vulkan API is strictly forbidden. You must exclusively use `vulkan.hpp` and its object-oriented encapsulations.
* **Dynamic Dispatch (ICD Direct Hit)**: 
  * The global macro `#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1` is MANDATORY. 
  * Static linking jumps are prohibited. You must use `vk::DispatchLoaderDynamic` to bypass the intermediate loader and extract physical function pointers directly from the GPU driver.
* **Execution Validation**: Every single Vulkan API call MUST be wrapped in a custom `VK_CHECK` macro. 
* **Safety Net**: The `VK_LAYER_KHRONOS_validation` layer must be explicitly activated during instance creation. If it fails to load, or if any API validation fails, throw a `std::runtime_error` and crash hard. Silent failures or fallbacks are unacceptable.

## 4. VRAM & Memory Architecture
* **Sub-Allocation Dictatorship**: Direct calls to `vk::Device::allocateMemory` or `vkAllocateMemory` are BANNED. All GPU device memory must be sub-allocated and managed through the Vulkan Memory Allocator (VMA).
* **VMA & Dynamic Dispatch Handshake**: Because dynamic dispatch is active, VMA must be initialized by manually feeding the extracted function pointers into the `VmaVulkanFunctions` struct.
* **Host-Device Memory Alignment**: Data structures (e.g., Gaussian spheres, matrices) passed between C++ Host and Compute Shaders must strictly comply with the `std430` memory layout. Implicit compiler padding is a fatal error.

## 5. Math & Coordinate Space
* **Vulkan NDC Compliance**: `#define GLM_FORCE_DEPTH_ZERO_TO_ONE` and `#define GLM_FORCE_LEFT_HANDED` are MANDATORY before including any GLM headers. This aligns GLM with Vulkan's native Normalized Device Coordinates (Y-axis down, Depth 0 to 1), breaking away from OpenGL legacy defaults.