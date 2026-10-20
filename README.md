# Bare Metal 3DGS

Bare Metal 3DGS is a C++20 and Vulkan compute renderer for 3D Gaussian Splatting data. It loads Gaussian records from a NumPy `.npy` file, sorts and bins them on the GPU, rasterizes them into a swapchain image, and presents the result in a GLFW window.

The renderer also includes a Dear ImGui HUD with camera controls, kernel parameters, VRAM usage, frustum-culling counts, and per-stage Vulkan timestamp measurements.

## Features

- GPU compute pipeline for depth sorting, covariance preprocessing, tile duplication, tile sorting, tile-pointer extraction, and rasterization.
- Free-look camera controlled with the mouse and WASD.
- ImGui telemetry HUD with live low-pass filter and Gaussian scale controls.
- Vulkan timestamp query timings for the main GPU stages.
- VMA-backed GPU buffers and offline GLSL-to-SPIR-V compilation through CMake.

## Requirements

- Windows 10/11, 64-bit.
- Visual Studio 2022 C++ workload (MSVC with C++20 support).
- CMake 3.20 or newer.
- Vulkan SDK 1.3 or newer, including `glslc.exe` and `VK_LAYER_KHRONOS_validation`.
- A discrete Vulkan GPU that supports Vulkan 1.3, shader `int64`, a queue family supporting compute, graphics and presentation, and queue timestamps.
- Git, with submodule support. The first CMake configure also fetches GLM and Dear ImGui, so it needs network access.
- The NVIDIA NVTX headers and `nvToolsExt64_1.lib` for the Windows link configured by this project. CMake currently expects them under `C:\Program Files\NVIDIA Corporation\NvToolsExt`; it also adds the CUDA 11.4 include path. Adjust the corresponding paths in `CMakeLists.txt` if your installation differs.

## Get the source

Clone the repository with its GLFW and Vulkan Memory Allocator submodules:

```powershell
git clone --recurse-submodules <GitHub-repository-URL>
cd Bare-Metal
```

For an existing clone, initialize or update the submodules with:

```powershell
git submodule update --init --recursive
```

## Build on Windows

Run these commands from a Visual Studio 2022 Developer PowerShell or a terminal where CMake, MSVC, and the Vulkan SDK are available:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

CMake compiles the compute shaders in `shaders/*.comp` into SPIR-V as part of the build. You do not need to compile the shaders manually.

If CMake cannot find Vulkan or `glslc`, install the Vulkan SDK and make sure its environment is initialized. If the linker cannot find `nvToolsExt64_1.lib`, install the NVTX component or update the NVTX include and library paths in `CMakeLists.txt`.

## Run

From the repository root, launch the Release executable:

```powershell
.\build\Release\BareMetalComputeProbe.exe
```

The application opens a window and loads `shaders/test.npy`. It currently uses this fixed input path and does not accept a model filename on the command line. Keep the source `shaders` directory in place when running the executable; the build embeds its path into the application.

GPU timings and visible-Gaussian counts are empty on the first frame, then update from the previous completed frame.

## Controls and HUD

Start in camera mode with the cursor captured by the render window:

| Input | Action |
| --- | --- |
| `W` / `S` | Move forward / backward |
| `A` / `D` | Move left / right |
| Mouse | Look around |
| `Tab` | Toggle between camera mode and UI mode |
| Window close button | Exit |

In UI mode, the cursor is released so the HUD controls can be used. The HUD shows CPU frame time and FPS, device-local VRAM usage, total/visible/culled Gaussian counts, and GPU timings for stages A, B, C, E, F, F.5, and G. Its sliders control the low-pass filter epsilon and global Gaussian scale modifier.

## Input data

The default dataset is `shaders/test.npy`. Each record must match the 64-byte `GaussianPoint` layout in [`core/GaussianData.h`](core/GaussianData.h): position and opacity, scale and padding, RGB color and padding, then a quaternion. The bundled converter expects a 3DGS PLY containing the usual position, opacity, scale, DC color, and rotation fields.

To convert a compatible PLY file, install the Python dependencies and run the converter from the repository root:

```powershell
py -m pip install numpy plyfile
py convert_ply_to_npy.py path\to\model.ply shaders\test.npy
```

This replaces the default input dataset. The parser expects a NumPy `.npy` file whose binary payload consists of 64-byte Gaussian records; use the supplied converter to preserve the expected field order and representation.

## Repository layout

```text
core/       Camera, Gaussian data types, and VMA buffer wrapper
shaders/    Compute shaders and the default test dataset
utils/      .npy parser, file I/O, and Vulkan error helpers
third_party/ GLFW and Vulkan Memory Allocator submodules
docs/       Project phase prompts and implementation notes
```
