# Project Conventions

## Project Overview
This project is a renderer engine designed for the analysis and visualization of `.npy` files, specifically tailored for 3D Gaussian Splatting (3DGS) data. This project serves as a learning vehicle for mastering the Vulkan API.

## Coding Standards

### Language Standard
- All code must strictly adhere to the **C++20** standard.

### Memory Management
- **No Raw Pointers:** The use of raw pointers (`*`) for ownership or resource management is strictly prohibited.
- **RAII:** Resource Acquisition Is Initialization (RAII) must be used for all resource management.
- **Smart Pointers:** Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) when dynamic allocation of resources is necessary.

### Vulkan API Usage
- **No Raw C API:** Do not use the raw C Vulkan API.
- **Vulkan.hpp:** You must use the C++ bindings provided by `vulkan.hpp` for all Vulkan-related operations.
