#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#define VMA_IMPLEMENTATION
#include "VmaUsage.h"

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <optional>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cmath>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <nvToolsExt.h>

#include "utils/FileIO.h"
#include "utils/VkCheck.h"
#include "core/VmaBuffer.h"
#include "core/GaussianData.h"
#include "core/Camera.h"
#include "utils/NpyParser.h"
#include "core/Gaussian2D.h"
#include "core/MathUtils.h"

// Global function pointer table memory allocation
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

// ============================================================
// Phase 10: Mouse state globals for camera control
// ============================================================
static double lastMouseX = 0.0;
static double lastMouseY = 0.0;
static float yaw = -90.0f;   // Initial yaw: looking along +Z (toward origin)
static float pitch = 0.0f;
static bool isLeftMouseDown = false;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isLeftMouseDown = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        } else if (action == GLFW_RELEASE) {
            isLeftMouseDown = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (!isLeftMouseDown) return;

    float xOffset = static_cast<float>(xpos - lastMouseX);
    float yOffset = static_cast<float>(lastMouseY - ypos); // y-axis inverted

    lastMouseX = xpos;
    lastMouseY = ypos;

    float sensitivity = 0.1f;
    yaw += xOffset * sensitivity;
    pitch += yOffset * sensitivity;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

class GlfwGuard {
public:
    GlfwGuard() {
        if (!glfwInit()) {
            throw std::runtime_error("Fatal: Failed to initialize GLFW.");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        std::cout << "[System] GLFW initialized." << std::endl;
    }
    ~GlfwGuard() { glfwTerminate(); }
    GlfwGuard(const GlfwGuard&) = delete;
    GlfwGuard& operator=(const GlfwGuard&) = delete;
};

class GlfwWindow {
public:
    GlfwWindow(uint32_t width, uint32_t height, const char* title) {
        m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!m_window) {
            throw std::runtime_error("Fatal: Failed to create GLFW window.");
        }
        std::cout << "[System] GLFW window created (" << width << "x" << height << ")." << std::endl;
    }
    ~GlfwWindow() { if (m_window) glfwDestroyWindow(m_window); }
    GlfwWindow(const GlfwWindow&) = delete;
    GlfwWindow& operator=(const GlfwWindow&) = delete;
    GLFWwindow* get() const { return m_window; }
private:
    GLFWwindow* m_window = nullptr;
};

class WindowSurface {
public:
    WindowSurface(vk::Instance instance, GLFWwindow* window) : m_instance(instance) {
        VkSurfaceKHR rawSurface;
        VkResult result = glfwCreateWindowSurface(
            static_cast<VkInstance>(instance), window, nullptr, &rawSurface);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Fatal: Failed to create window surface.");
        }
        m_surface = static_cast<vk::SurfaceKHR>(rawSurface);
        std::cout << "[System] Window Surface created." << std::endl;
    }
    ~WindowSurface() {
        if (m_surface) {
            m_instance.destroySurfaceKHR(m_surface);
        }
    }
    WindowSurface(const WindowSurface&) = delete;
    WindowSurface& operator=(const WindowSurface&) = delete;
    vk::SurfaceKHR get() const { return m_surface; }
private:
    vk::Instance m_instance;
    vk::SurfaceKHR m_surface = nullptr;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;
};

struct TargetDevice {
    vk::PhysicalDevice physical;
    QueueFamilyIndices indices;
};

class DeviceSelector {
public:
    static TargetDevice selectOptimalDevice(const vk::Instance& instance, vk::SurfaceKHR surface) {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("Fatal: No Vulkan supported GPU.");
        }

        std::multimap<int, TargetDevice> candidates;

        for (const auto& device : devices) {
            auto [score, indices] = evaluateDevice(device, surface);
            if (score > 0) {
                candidates.insert({score, TargetDevice{device, indices}});
            }
        }

        if (candidates.empty()) {
            throw std::runtime_error("Fatal: None of the GPU could pass the hard indicators of "
                "the underlying computing architecture for interception.");
        }

        TargetDevice bestDevice = candidates.rbegin()->second;

        std::cout << "[System] Locked Target Device: "
                  << bestDevice.physical.getProperties().deviceName.data() << std::endl;

        return bestDevice;
    }

private:
    static std::pair<int, QueueFamilyIndices> evaluateDevice(const vk::PhysicalDevice& device, vk::SurfaceKHR surface) {
        int score = 0;
        QueueFamilyIndices indices{};
        vk::PhysicalDeviceProperties properties = device.getProperties();

        // 1. Needs to be a Discrete GPU
        if (properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) {
            return {0, indices};
        }

        // 2. Must support Compute Queue
        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) {
                indices.computeFamily = i;
                break;
            }
        }
        if (!indices.computeFamily.has_value()) {
            return {0, indices};
        }

        // 2b. Check if compute queue also supports present on this surface
        vk::Bool32 presentSupport = device.getSurfaceSupportKHR(
            indices.computeFamily.value(), surface);
        if (presentSupport) {
            indices.presentFamily = indices.computeFamily.value();
        }
        if (!indices.presentFamily.has_value()) {
            return {0, indices};
        }

        // 3. Evaluate the VRAM, every 1MB counts as one point
        vk::PhysicalDeviceMemoryProperties memProperties = device.getMemoryProperties();
        vk::DeviceSize vramSize = 0;
        for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
            if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                vramSize += memProperties.memoryHeaps[i].size;
            }
        }
        score += static_cast<int>(vramSize / (1024 * 1024));

        // 4. Whether support at least Vulkan 1.3+
        if (properties.apiVersion >= VK_API_VERSION_1_3) {
            score += 10000;
        }

        return {score, indices};
    }
};

class VulkanDevice {
public:
    explicit VulkanDevice(const vk::Instance& instance, const TargetDevice& target) {
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo(
            vk::DeviceQueueCreateFlags(),
            target.indices.computeFamily.value(),
            1,
            &queuePriority
        );

        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.synchronization2 = VK_TRUE;

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        vk::DeviceCreateInfo deviceCreateInfo(
            vk::DeviceCreateFlags(),
            1,
            &queueCreateInfo,
            0, nullptr,
            static_cast<uint32_t>(deviceExtensions.size()),
            deviceExtensions.data()
        );
        deviceCreateInfo.pNext = &vulkan13Features;

        device = target.physical.createDeviceUnique(deviceCreateInfo);

        // Extracting device-level function pointers
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

        computeQueue = device->getQueue(target.indices.computeFamily.value(), 0);

        // Initialize VMA Allocator (Pure C Interface)
        VmaVulkanFunctions vulkanFunctions = {};
        vulkanFunctions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
        vulkanFunctions.vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
        vulkanFunctions.vkGetPhysicalDeviceProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties;
        vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties;
        vulkanFunctions.vkAllocateMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory;
        vulkanFunctions.vkFreeMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory;
        vulkanFunctions.vkMapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory;
        vulkanFunctions.vkUnmapMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkUnmapMemory;
        vulkanFunctions.vkFlushMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkFlushMappedMemoryRanges;
        vulkanFunctions.vkInvalidateMappedMemoryRanges = VULKAN_HPP_DEFAULT_DISPATCHER.vkInvalidateMappedMemoryRanges;
        vulkanFunctions.vkBindBufferMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory;
        vulkanFunctions.vkBindImageMemory = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory;
        vulkanFunctions.vkGetBufferMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements;
        vulkanFunctions.vkGetImageMemoryRequirements = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements;
        vulkanFunctions.vkCreateBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer;
        vulkanFunctions.vkDestroyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer;
        vulkanFunctions.vkCreateImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage;
        vulkanFunctions.vkDestroyImage = VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage;
        vulkanFunctions.vkCmdCopyBuffer = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer;

        vulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceMemoryProperties2;
        vulkanFunctions.vkGetBufferMemoryRequirements2KHR       = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements2;
        vulkanFunctions.vkGetImageMemoryRequirements2KHR        = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2;
        vulkanFunctions.vkBindBufferMemory2KHR                  = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory2;
        vulkanFunctions.vkBindImageMemory2KHR                   = VULKAN_HPP_DEFAULT_DISPATCHER.vkBindImageMemory2;
        
        vulkanFunctions.vkGetDeviceBufferMemoryRequirements     = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceBufferMemoryRequirements;
        vulkanFunctions.vkGetDeviceImageMemoryRequirements      = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceImageMemoryRequirements;

        VmaAllocatorCreateInfo allocatorCreateInfo = {};
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        
        allocatorCreateInfo.instance = static_cast<VkInstance>(instance);
        allocatorCreateInfo.physicalDevice = static_cast<VkPhysicalDevice>(target.physical);
        allocatorCreateInfo.device = static_cast<VkDevice>(device.get());
        
        allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;

        VK_CHECK(vmaCreateAllocator(&allocatorCreateInfo, &allocator));
    }

    ~VulkanDevice() {
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
        }
    }

    vk::Queue getComputeQueue() const { return computeQueue; }
    vk::Device getDevice() const { return *device; }
    VmaAllocator getAllocator() const { return allocator; }

private:
    vk::UniqueDevice device;
    vk::Queue computeQueue;
    VmaAllocator allocator = VK_NULL_HANDLE;
};

class VulkanInstance {
public:
    VulkanInstance() {
        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = 
            dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        if (!vkGetInstanceProcAddr) {
            throw std::runtime_error("Fatal: Unable to extract the Vulkan loader pointer from the system dynamic library.");
        }
        
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        vk::ApplicationInfo appInfo("Bare Metal 3DGS", 1, "Bare Metal", 1, VK_API_VERSION_1_3);

        const std::vector<const char*> validationLayers = {
            "VK_LAYER_KHRONOS_validation"
        };
        const std::vector<const char*> validationLayers = {};

        auto availableLayers = vk::enumerateInstanceLayerProperties();
        for (const char* layerName : validationLayers) {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }
            if (!layerFound) {
                throw std::runtime_error(std::string("Fatal: Validation Layer Lost: ") + layerName);
            }
        }

        // Get GLFW required Vulkan extensions (includes VK_KHR_surface + platform extension)
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions) {
            throw std::runtime_error("Fatal: GLFW requires Vulkan surface extensions that are not available.");
        }

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        std::cout << "[System] GLFW required " << glfwExtensionCount << " Vulkan instance extension(s):" << std::endl;
        for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
            std::cout << "  - " << glfwExtensions[i] << std::endl;
        }

        vk::InstanceCreateInfo createInfo({}, &appInfo, validationLayers, extensions);

        instance = vk::createInstanceUnique(createInfo);

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    }

    ~VulkanInstance() {
        instance.reset();
    }

    vk::Instance get() const { return *instance; }

private:
    vk::detail::DynamicLoader dl;
    vk::UniqueInstance instance;
};

int main() {
    try {
        // ===== GLFW Initialization (before Vulkan instance) =====
        GlfwGuard glfwGuard;

        std::cout << "--- Bare Metal Boot Sequence Initiated ---" << std::endl;
        
        uint32_t WIDTH = 1024;
        uint32_t HEIGHT = 768;

        VulkanInstance vkInstance;
        std::cout << "[System] Vulkan Instance initialized (Dynamic Dispatch Active)." << std::endl;

        // ===== Phase 9: Create GLFW Window & Surface =====
        GlfwWindow window(WIDTH, HEIGHT, "Bare Metal 3DGS");

        glfwSetMouseButtonCallback(window.get(), mouseButtonCallback);
        glfwSetCursorPosCallback(window.get(), cursorPosCallback);
        glfwGetCursorPos(window.get(), &lastMouseX, &lastMouseY);

        WindowSurface surface(vkInstance.get(), window.get());

        TargetDevice target = DeviceSelector::selectOptimalDevice(vkInstance.get(), surface.get());

        VulkanDevice vulkanDevice(vkInstance.get(), target);
        std::cout << "[System] Logical Device and Compute Queue successfully engaged." << std::endl;
        std::cout << "[System] VMA Allocator successfully engaged." << std::endl;
        std::cout << "--- Bare Metal Boot Sequence Completed ---" << std::endl;

        // ===== Phase 9: Build Swapchain =====
        auto surfaceCapabilities = target.physical.getSurfaceCapabilitiesKHR(surface.get());
        auto surfaceFormats = target.physical.getSurfaceFormatsKHR(surface.get());
        auto surfacePresentModes = target.physical.getSurfacePresentModesKHR(surface.get());

        // Choose format: prefer eB8G8R8A8Unorm + eSrgbNonlinear
        vk::SurfaceFormatKHR chosenFormat = surfaceFormats[0];
        for (const auto& fmt : surfaceFormats) {
            if (fmt.format == vk::Format::eB8G8R8A8Unorm &&
                fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                chosenFormat = fmt;
                break;
            }
        }
        std::cout << "[System] Swapchain format: " << vk::to_string(chosenFormat.format) << std::endl;

        // Choose present mode: prefer eMailbox, fallback to eFifo (guaranteed)
        vk::PresentModeKHR chosenPresentMode = vk::PresentModeKHR::eFifo;
        for (const auto& mode : surfacePresentModes) {
            if (mode == vk::PresentModeKHR::eMailbox) {
                chosenPresentMode = mode;
                break;
            }
        }
        std::cout << "[System] Swapchain present mode: " << vk::to_string(chosenPresentMode) << std::endl;

        // Choose extent
        vk::Extent2D swapchainExtent;
        if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
            swapchainExtent = surfaceCapabilities.currentExtent;
        } else {
            swapchainExtent.width = std::clamp(WIDTH,
                surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width);
            swapchainExtent.height = std::clamp(HEIGHT,
                surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height);
        }
        WIDTH = swapchainExtent.width;
        HEIGHT = swapchainExtent.height;
        std::cout << "[System] Swapchain extent: " << WIDTH << "x" << HEIGHT << std::endl;

        // Choose image count (prefer triple buffering)
        uint32_t minImageCount = surfaceCapabilities.minImageCount + 1;
        if (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) {
            minImageCount = surfaceCapabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR swapchainCreateInfo(
            vk::SwapchainCreateFlagsKHR(),
            surface.get(),
            minImageCount,
            chosenFormat.format,
            chosenFormat.colorSpace,
            swapchainExtent,
            1,                                          // imageArrayLayers
            vk::ImageUsageFlagBits::eTransferDst,       // Compute result copies TO this image
            vk::SharingMode::eExclusive,
            0, nullptr,
            surfaceCapabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            chosenPresentMode,
            VK_TRUE,                                    // clipped
            nullptr                                     // oldSwapchain
        );

        auto swapchain = vulkanDevice.getDevice().createSwapchainKHRUnique(swapchainCreateInfo);
        std::cout << "[System] Swapchain created (" << minImageCount << " images)." << std::endl;

        auto swapchainImages = vulkanDevice.getDevice().getSwapchainImagesKHR(*swapchain);
        std::cout << "[System] Retrieved " << swapchainImages.size() << " swapchain images." << std::endl;

        // ===== Compute Pipeline Setup =====

        // 1. Read shader binary
        std::string shaderPath = std::string(SHADER_DIR) + "/calc_depth.spv";
        auto shaderCode = FileIO::readFile(shaderPath);
        std::cout << "[System] Shader loaded: " << shaderCode.size() << " bytes." << std::endl;

        // 2. Parse Gaussian points
        std::string npyPath = std::string(SHADER_DIR) + "/test.npy";
        auto pointData = NpyParser::parse(npyPath);
        vk::DeviceSize gaussianBufferSize = pointData.size() * sizeof(GaussianPoint);
        std::cout << "[System] Loaded " << pointData.size() << " Gaussian points (" 
                  << gaussianBufferSize << " bytes)." << std::endl;

        uint32_t paddedCount = 1;
        while (paddedCount < pointData.size()) {
            paddedCount <<= 1;
        }
        std::cout << "[System] Radix sort padded count: " << paddedCount 
                  << " (original: " << pointData.size() << ")." << std::endl;

        uint32_t histogramGroupCount = (paddedCount + 255) / 256;

        // 3. Staging buffer and DMA transfer
        VmaBuffer stagingBuffer(
            vulkanDevice.getAllocator(),
            gaussianBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        void* mapped = stagingBuffer.map();
        std::memcpy(mapped, pointData.data(), gaussianBufferSize);
        stagingBuffer.flush();
        stagingBuffer.unmap();
        std::cout << "[System] Staging buffer populated with Gaussian data." << std::endl;

        VmaBuffer gaussianBuffer(
            vulkanDevice.getAllocator(),
            gaussianBufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        // 4. Command pool (with reset flag for per-frame recording)
        vk::CommandPoolCreateInfo commandPoolCreateInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            target.indices.computeFamily.value()
        );

        auto commandPool = vulkanDevice.getDevice().createCommandPoolUnique(commandPoolCreateInfo);

        // 5. DMA transfer
        vk::CommandBufferAllocateInfo tempCmdBufInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );
        auto tempCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(tempCmdBufInfo);
        auto& tempCmdBuffer = tempCmdBuffers[0];

        vk::CommandBufferBeginInfo tempBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        tempCmdBuffer->begin(tempBeginInfo);

        vk::BufferCopy copyRegion(0, 0, gaussianBufferSize);
        tempCmdBuffer->copyBuffer(stagingBuffer.getBuffer(), gaussianBuffer.getBuffer(), copyRegion);

        tempCmdBuffer->end();

        vk::SubmitInfo tempSubmitInfo(0, nullptr, nullptr, 1, &*tempCmdBuffer, 0, nullptr);
        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &tempSubmitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();
        std::cout << "[System] DMA Transfer Complete: Gaussian data injected into Device Local VRAM." << std::endl;

        // 6. Camera buffer
        constexpr float PI = 3.14159265358979f;

        CameraUniform cameraData{};

        // Compute initial forward from yaw/pitch
        float radYaw = yaw * PI / 180.0f;
        float radPitch = pitch * PI / 180.0f;
        float initFwdX = std::cos(radPitch) * std::cos(radYaw);
        float initFwdY = std::sin(radPitch);
        float initFwdZ = std::cos(radPitch) * std::sin(radYaw);

        cameraData.cameraPosition[0] = 0.0f;
        cameraData.cameraPosition[1] = 0.0f;
        cameraData.cameraPosition[2] = 5.0f;
        cameraData.cameraPosition[3] = 0.0f;
        cameraData.cameraForward[0] = initFwdX;
        cameraData.cameraForward[1] = initFwdY;
        cameraData.cameraForward[2] = initFwdZ;
        cameraData.cameraForward[3] = 0.0f;

        MathUtils::buildViewProjection(cameraData.viewProj, 0.0f, 0.0f, 5.0f,
            initFwdX, initFwdY, initFwdZ, WIDTH, HEIGHT);

        cameraData.focal_x = 1000.0f;
        cameraData.focal_y = 1000.0f;
        cameraData.tan_fovx = 0.5f;
        cameraData.tan_fovy = 0.5f;
        cameraData.viewport[0] = static_cast<float>(WIDTH);
        cameraData.viewport[1] = static_cast<float>(HEIGHT);
        cameraData.pad_cam[0] = 0.0f;
        cameraData.pad_cam[1] = 0.0f;

        VmaBuffer cameraBuffer(
            vulkanDevice.getAllocator(),
            sizeof(CameraUniform),
            vk::BufferUsageFlagBits::eUniformBuffer,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        CameraUniform* mappedCameraData = static_cast<CameraUniform*>(cameraBuffer.map());
        std::memcpy(mappedCameraData, &cameraData, sizeof(CameraUniform));
        cameraBuffer.flush();
        std::cout << "[System] Camera buffer created (persistent mapped), position (0, 0, 5), forward ("
                  << initFwdX << ", " << initFwdY << ", " << initFwdZ << "), yaw=" << yaw << std::endl;

        // 7. Sort key buffer
        vk::DeviceSize sortKeyBufferSize = static_cast<vk::DeviceSize>(paddedCount) * sizeof(SortKeyPair);
        VmaBuffer sortKeyBuffer(
            vulkanDevice.getAllocator(),
            sortKeyBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Sort key buffer created (" << sortKeyBufferSize << " bytes)." << std::endl;

        // 7b. Gaussian2D buffer
        vk::DeviceSize gaussian2DBufferSize = static_cast<vk::DeviceSize>(paddedCount) * sizeof(Gaussian2D);
        VmaBuffer gaussian2DBuffer(
            vulkanDevice.getAllocator(),
            gaussian2DBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Gaussian2D buffer created (" << gaussian2DBufferSize << " bytes)." << std::endl;

        // 7c. Image buffer (uint32_t per pixel = packed BGRA8)
        vk::DeviceSize imageBufferSize = static_cast<vk::DeviceSize>(WIDTH) * HEIGHT * sizeof(uint32_t);
        VmaBuffer imageBuffer(
            vulkanDevice.getAllocator(),
            imageBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Image buffer created (" << imageBufferSize << " bytes, " << WIDTH << "x" << HEIGHT << " BGRA8)." << std::endl;

        // 7d. Sort key buffer alt (ping-pong for radix sort)
        VmaBuffer sortKeyBufferAlt(
            vulkanDevice.getAllocator(),
            sortKeyBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Sort key buffer alt created (" << sortKeyBufferSize << " bytes, ping-pong)." << std::endl;

        // 7e. Histogram buffer for radix sort: [histogramGroupCount * 256] uint32_t
        vk::DeviceSize histogramBufferSize = static_cast<vk::DeviceSize>(histogramGroupCount) * 256 * sizeof(uint32_t);
        VmaBuffer histogramBuffer(
            vulkanDevice.getAllocator(),
            histogramBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Radix histogram buffer created (" << histogramBufferSize 
                  << " bytes, " << histogramGroupCount << " workgroups x 256 buckets)." << std::endl;

        // 7f. Global offset buffer for radix sort: [256] uint32_t
        vk::DeviceSize globalOffsetBufferSize = 256 * sizeof(uint32_t);
        VmaBuffer globalOffsetBuffer(
            vulkanDevice.getAllocator(),
            globalOffsetBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Radix global offset buffer created (" << globalOffsetBufferSize << " bytes)." << std::endl;

        // 8. Shader module
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        );
        auto shaderModule = vulkanDevice.getDevice().createShaderModuleUnique(shaderModuleCreateInfo);

        // 9. Descriptor set layout (8 bindings: original 5 + radix sort 3)
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // GaussianBuffer
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // SortKeyBuffer (src)
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute), // CameraUBO
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // Gaussian2DBuffer
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // ImageBuffer
            vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // SortKeyBufferAlt (dst)
            vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // HistogramBuffer
            vk::DescriptorSetLayoutBinding(7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)  // GlobalOffsetBuffer
        };

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(layoutBindings.size()),
            layoutBindings.data()
        );

        auto descriptorSetLayout = vulkanDevice.getDevice().createDescriptorSetLayoutUnique(layoutCreateInfo);

        // 10. Descriptor pool (2 descriptor sets, 8 bindings each)
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 14),  // 7 storage buffers x 2 sets
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2)    // 1 uniform buffer x 2 sets
        };

        vk::DescriptorPoolCreateInfo poolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            2,  // maxSets: mainDS + swapDS
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data()
        );

        auto descriptorPool = vulkanDevice.getDevice().createDescriptorPoolUnique(poolCreateInfo);

        // 11. Pipeline layout (push constants: 3 x uint32_t = 12 bytes)
        vk::PushConstantRange pushConstantRange(
            vk::ShaderStageFlagBits::eCompute,
            0,
            sizeof(uint32_t) * 3
        );

        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            vk::PipelineLayoutCreateFlags(),
            1,
            &*descriptorSetLayout,
            1,
            &pushConstantRange
        );

        auto pipelineLayout = vulkanDevice.getDevice().createPipelineLayoutUnique(pipelineLayoutCreateInfo);

        // 12a. calc_depth pipeline
        vk::PipelineShaderStageCreateInfo stageCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            *shaderModule,
            "main"
        );

        vk::ComputePipelineCreateInfo computePipelineCreateInfo(
            vk::PipelineCreateFlags(),
            stageCreateInfo,
            *pipelineLayout
        );

        auto computePipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, computePipelineCreateInfo);
        if (computePipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create compute pipeline.");
        }
        auto computePipeline = std::move(computePipelineResult.value);

        // 12b. radix_histogram pipeline
        std::string radixHistogramShaderPath = std::string(SHADER_DIR) + "/radix_histogram.spv";
        auto radixHistogramShaderCode = FileIO::readFile(radixHistogramShaderPath);

        vk::ShaderModuleCreateInfo radixHistogramSMCI(vk::ShaderModuleCreateFlags(), radixHistogramShaderCode.size(),
            reinterpret_cast<const uint32_t*>(radixHistogramShaderCode.data()));
        auto radixHistogramShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(radixHistogramSMCI);

        vk::PipelineShaderStageCreateInfo radixHistogramStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *radixHistogramShaderModule, "main");
        vk::ComputePipelineCreateInfo radixHistogramPipelineCI(vk::PipelineCreateFlags(), radixHistogramStageCI, *pipelineLayout);

        auto radixHistogramPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, radixHistogramPipelineCI);
        if (radixHistogramPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create radix histogram compute pipeline.");
        }
        auto radixHistogramPipeline = std::move(radixHistogramPipelineResult.value);
        std::cout << "[System] Radix histogram pipeline created." << std::endl;

        // 12c. radix_scan pipeline
        std::string radixScanShaderPath = std::string(SHADER_DIR) + "/radix_scan.spv";
        auto radixScanShaderCode = FileIO::readFile(radixScanShaderPath);

        vk::ShaderModuleCreateInfo radixScanSMCI(vk::ShaderModuleCreateFlags(), radixScanShaderCode.size(),
            reinterpret_cast<const uint32_t*>(radixScanShaderCode.data()));
        auto radixScanShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(radixScanSMCI);

        vk::PipelineShaderStageCreateInfo radixScanStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *radixScanShaderModule, "main");
        vk::ComputePipelineCreateInfo radixScanPipelineCI(vk::PipelineCreateFlags(), radixScanStageCI, *pipelineLayout);

        auto radixScanPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, radixScanPipelineCI);
        if (radixScanPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create radix scan compute pipeline.");
        }
        auto radixScanPipeline = std::move(radixScanPipelineResult.value);
        std::cout << "[System] Radix scan pipeline created." << std::endl;

        // 12d. radix_scatter pipeline
        std::string radixScatterShaderPath = std::string(SHADER_DIR) + "/radix_scatter.spv";
        auto radixScatterShaderCode = FileIO::readFile(radixScatterShaderPath);

        vk::ShaderModuleCreateInfo radixScatterSMCI(vk::ShaderModuleCreateFlags(), radixScatterShaderCode.size(),
            reinterpret_cast<const uint32_t*>(radixScatterShaderCode.data()));
        auto radixScatterShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(radixScatterSMCI);

        vk::PipelineShaderStageCreateInfo radixScatterStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *radixScatterShaderModule, "main");
        vk::ComputePipelineCreateInfo radixScatterPipelineCI(vk::PipelineCreateFlags(), radixScatterStageCI, *pipelineLayout);

        auto radixScatterPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, radixScatterPipelineCI);
        if (radixScatterPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create radix scatter compute pipeline.");
        }
        auto radixScatterPipeline = std::move(radixScatterPipelineResult.value);
        std::cout << "[System] Radix scatter pipeline created." << std::endl;

        // 12e. preprocess pipeline
        std::string preprocessShaderPath = std::string(SHADER_DIR) + "/preprocess.spv";
        auto preprocessShaderCode = FileIO::readFile(preprocessShaderPath);

        vk::ShaderModuleCreateInfo preprocessSMCI(vk::ShaderModuleCreateFlags(), preprocessShaderCode.size(),
            reinterpret_cast<const uint32_t*>(preprocessShaderCode.data()));
        auto preprocessShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(preprocessSMCI);

        vk::PipelineShaderStageCreateInfo preprocessStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *preprocessShaderModule, "main");
        vk::ComputePipelineCreateInfo preprocessPipelineCI(vk::PipelineCreateFlags(), preprocessStageCI, *pipelineLayout);

        auto preprocessPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, preprocessPipelineCI);
        if (preprocessPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create preprocess compute pipeline.");
        }
        auto preprocessPipeline = std::move(preprocessPipelineResult.value);
        std::cout << "[System] Preprocess pipeline created." << std::endl;

        // 12f. rasterize pipeline
        std::string rasterizeShaderPath = std::string(SHADER_DIR) + "/rasterize.spv";
        auto rasterizeShaderCode = FileIO::readFile(rasterizeShaderPath);

        vk::ShaderModuleCreateInfo rasterizeSMCI(vk::ShaderModuleCreateFlags(), rasterizeShaderCode.size(),
            reinterpret_cast<const uint32_t*>(rasterizeShaderCode.data()));
        auto rasterizeShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(rasterizeSMCI);

        vk::PipelineShaderStageCreateInfo rasterizeStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *rasterizeShaderModule, "main");
        vk::ComputePipelineCreateInfo rasterizePipelineCI(vk::PipelineCreateFlags(), rasterizeStageCI, *pipelineLayout);

        auto rasterizePipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, rasterizePipelineCI);
        if (rasterizePipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create rasterize compute pipeline.");
        }
        auto rasterizePipeline = std::move(rasterizePipelineResult.value);
        std::cout << "[System] Rasterize pipeline created." << std::endl;

        // 13. Descriptor sets (2 sets: mainDS + swapDS for ping-pong radix sort)
        std::vector<vk::DescriptorSetLayout> setLayouts = {*descriptorSetLayout, *descriptorSetLayout};
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool, static_cast<uint32_t>(setLayouts.size()), setLayouts.data());
        auto descriptorSets = vulkanDevice.getDevice().allocateDescriptorSets(descriptorSetAllocateInfo);
        auto mainDS = descriptorSets[0];
        auto swapDS = descriptorSets[1];

        vk::DescriptorBufferInfo gaussianBufferInfo(gaussianBuffer.getBuffer(), 0, gaussianBufferSize);
        vk::DescriptorBufferInfo sortKeyBufferInfo(sortKeyBuffer.getBuffer(), 0, sortKeyBufferSize);
        vk::DescriptorBufferInfo cameraBufferInfo(cameraBuffer.getBuffer(), 0, sizeof(CameraUniform));
        vk::DescriptorBufferInfo gaussian2DBufferInfo(gaussian2DBuffer.getBuffer(), 0, gaussian2DBufferSize);
        vk::DescriptorBufferInfo imageBufferInfo(imageBuffer.getBuffer(), 0, imageBufferSize);
        vk::DescriptorBufferInfo sortKeyBufferAltInfo(sortKeyBufferAlt.getBuffer(), 0, sortKeyBufferSize);
        vk::DescriptorBufferInfo histogramBufferInfo(histogramBuffer.getBuffer(), 0, histogramBufferSize);
        vk::DescriptorBufferInfo globalOffsetBufferInfo(globalOffsetBuffer.getBuffer(), 0, globalOffsetBufferSize);

        std::vector<vk::WriteDescriptorSet> allWrites = {
            // mainDS: binding 1 = sortKeyBuffer (src), binding 5 = sortKeyBufferAlt (dst)
            vk::WriteDescriptorSet(mainDS, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferAltInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histogramBufferInfo, nullptr),
            vk::WriteDescriptorSet(mainDS, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &globalOffsetBufferInfo, nullptr),
            // swapDS: binding 1 = sortKeyBufferAlt (src), binding 5 = sortKeyBuffer (dst) — SWAPPED
            vk::WriteDescriptorSet(swapDS, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferAltInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histogramBufferInfo, nullptr),
            vk::WriteDescriptorSet(swapDS, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &globalOffsetBufferInfo, nullptr),
        };

        vulkanDevice.getDevice().updateDescriptorSets(
            static_cast<uint32_t>(allWrites.size()), allWrites.data(), 0, nullptr);

        // 14. Command buffer
        const int MAX_FRAMES_IN_FLIGHT = 2;
        const uint32_t MAX_FRAMES_IN_FLIGHT = static_cast<uint32_t>(swapchainImages.size());
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
        auto commandBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo);
        //auto commandBuffer = std::move(commandBuffers[0]);

        // Phase 10: Vulkan traffic lights (semaphores)
        std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
        std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
        std::vector<vk::UniqueFence> renderFences;

        vk::SemaphoreCreateInfo semaphoreCreateInfo;
        vk::FenceCreateInfo renderFenceInfo(vk::FenceCreateFlagBits::eSignaled);

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            imageAvailableSemaphores.push_back(vulkanDevice.getDevice().createSemaphoreUnique(semaphoreCreateInfo));
            renderFinishedSemaphores.push_back(vulkanDevice.getDevice().createSemaphoreUnique(semaphoreCreateInfo));
            renderFences.push_back(vulkanDevice.getDevice().createFenceUnique(renderFenceInfo));
        }
        

        uint32_t currentFrame = 0;
        std::cout << "[System] " << MAX_FRAMES_IN_FLIGHT << " sets of Synchronization traffic lights created." << std::endl;

        std::cout << "[System] Entering render loop..." << std::endl;

        // ===== Phase 10: The Async Hot Loop (Radix Sort) =====
        while (!glfwWindowShouldClose(window.get())) {
            glfwPollEvents();

            // Wait for the specific frame's GPU work to complete
            VK_CHECK(vulkanDevice.getDevice().waitForFences(*renderFences[currentFrame], VK_TRUE, UINT64_MAX));
            vulkanDevice.getDevice().resetFences(*renderFences[currentFrame]);

            // Update camera from mouse input
            float radYaw = yaw * PI / 180.0f;
            float radPitch = pitch * PI / 180.0f;
            float fwdX = std::cos(radPitch) * std::cos(radYaw);
            float fwdY = std::sin(radPitch);
            float fwdZ = std::cos(radPitch) * std::sin(radYaw);

            mappedCameraData->cameraForward[0] = fwdX;
            mappedCameraData->cameraForward[1] = fwdY;
            mappedCameraData->cameraForward[2] = fwdZ;
            mappedCameraData->cameraForward[3] = 0.0f;
            mappedCameraData->cameraPosition[0] = 0.0f;
            mappedCameraData->cameraPosition[1] = 0.0f;
            mappedCameraData->cameraPosition[2] = 5.0f;
            mappedCameraData->cameraPosition[3] = 0.0f;

            MathUtils::buildViewProjection(mappedCameraData->viewProj, 0.0f, 0.0f, 5.0f,
                fwdX, fwdY, fwdZ, WIDTH, HEIGHT);
            cameraBuffer.flush();

            // Acquire swapchain image (semaphore signals when available)
            auto acqResult = vulkanDevice.getDevice().acquireNextImageKHR(
                *swapchain, UINT64_MAX, *imageAvailableSemaphores[currentFrame], nullptr);
            if (acqResult.result != vk::Result::eSuccess && acqResult.result != vk::Result::eSuboptimalKHR) {
                VK_CHECK(acqResult.result);
            }
            uint32_t imageIndex = acqResult.value;

            // Extract the command scroll specific to the current frame
            auto& commandBuffer = commandBuffers[currentFrame];

            // Record command buffer
            commandBuffer->reset(vk::CommandBufferResetFlags());
            vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer->begin(beginInfo);

            // ====== Stage A: Buffer Pad & Depth Calc ======
            nvtxRangePushA("1. Buffer Pad & Depth Calc");

            // Pad buffers
            commandBuffer->fillBuffer(sortKeyBuffer.getBuffer(), 0, VK_WHOLE_SIZE, ~0u);
            commandBuffer->fillBuffer(gaussian2DBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0);

            vk::MemoryBarrier fillBarrier(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &fillBarrier, 0, nullptr, 0, nullptr);

            // Dispatch calc_depth
            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
            commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &mainDS, 0, nullptr);

            uint32_t groupCountX = (static_cast<uint32_t>(pointData.size()) + 255) / 256;
            commandBuffer->dispatch(groupCountX, 1, 1);

            vk::MemoryBarrier depthBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &depthBarrier, 0, nullptr, 0, nullptr);

            nvtxRangePop();

            // ====== Stage B: Radix Sort O(N) ======
            nvtxRangePushA("2. Radix Sort O(N)");

            // ============================================================
            // Radix Sort: 4 passes over 8-bit chunks (O(N) total)
            // After 4 passes (even), sorted data resides in sortKeyBuffer.
            // ============================================================
            for (uint32_t pass = 0; pass < 4; pass++) {
                uint32_t shift = pass * 8;
                bool isEvenPass = (pass % 2 == 0);
                vk::DescriptorSet currentDS = isEvenPass ? mainDS : swapDS;

                // Bind the correct descriptor set for this pass (ping-pong)
                commandBuffer->bindDescriptorSets(
                    vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &currentDS, 0, nullptr);

                // --- Histogram Pass ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *radixHistogramPipeline);
                uint32_t histPC[3] = {shift, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, histPC);
                commandBuffer->dispatch(histogramGroupCount, 1, 1);

                vk::MemoryBarrier histBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &histBarrier, 0, nullptr, 0, nullptr);

                // --- Scan Pass (single workgroup) ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *radixScanPipeline);
                uint32_t scanPC[3] = {histogramGroupCount, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, scanPC);
                commandBuffer->dispatch(1, 1, 1);

                vk::MemoryBarrier scanBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &scanBarrier, 0, nullptr, 0, nullptr);

                // --- Scatter Pass ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *radixScatterPipeline);
                uint32_t scatterPC[3] = {shift, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, scatterPC);
                commandBuffer->dispatch(histogramGroupCount, 1, 1);

                vk::MemoryBarrier scatterBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &scatterBarrier, 0, nullptr, 0, nullptr);
            }

            // Rebind mainDS for downstream shaders (sorted data is in sortKeyBuffer after 4 passes)
            commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &mainDS, 0, nullptr);

            nvtxRangePop();

            // ====== Stage C: Covariance Preprocess ======
            nvtxRangePushA("3. Covariance Preprocess");

            vk::MemoryBarrier preprocessBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &preprocessBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *preprocessPipeline);
            uint32_t preprocessGroupCountX = (paddedCount + 255) / 256;
            commandBuffer->dispatch(preprocessGroupCountX, 1, 1);

            nvtxRangePop();

            // ====== Stage D: Compute Rasterization ======
            nvtxRangePushA("4. Compute Rasterization");

            vk::MemoryBarrier rasterizeBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &rasterizeBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *rasterizePipeline);
            //uint32_t rasterizePC[3] = {static_cast<uint32_t>(pointData.size()), WIDTH, HEIGHT};
            uint32_t rasterizePC[3] = { 30000, WIDTH, HEIGHT };
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, rasterizePC);
            uint32_t rasterizeGroupCountX = (WIDTH + 15) / 16;
            uint32_t rasterizeGroupCountY = (HEIGHT + 15) / 16;
            commandBuffer->dispatch(rasterizeGroupCountX, rasterizeGroupCountY, 1);

            // Image copy: buffer -> swapchain image
            vk::MemoryBarrier computeToTransferBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eTransferRead);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags(), 1, &computeToTransferBarrier, 0, nullptr, 0, nullptr);

            vk::ImageMemoryBarrier toTransferDst(
                vk::AccessFlagBits::eNone,
                vk::AccessFlagBits::eTransferWrite,
                vk::ImageLayout::eUndefined,
                vk::ImageLayout::eTransferDstOptimal,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                swapchainImages[imageIndex],
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toTransferDst);

            vk::BufferImageCopy bufferImageCopy(
                0, 0, 0,
                vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                vk::Offset3D(0, 0, 0),
                vk::Extent3D(WIDTH, HEIGHT, 1));
            commandBuffer->copyBufferToImage(
                imageBuffer.getBuffer(),
                swapchainImages[imageIndex],
                vk::ImageLayout::eTransferDstOptimal,
                bufferImageCopy);

            vk::ImageMemoryBarrier toPresentSrc(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eNone,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::ePresentSrcKHR,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                swapchainImages[imageIndex],
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
                vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toPresentSrc);

            nvtxRangePop();

            commandBuffer->end();

            // Single submit: wait on imageAvailableSemaphore, signal renderFinishedSemaphore + renderFence
            vk::Semaphore waitSemaphores[] = {*imageAvailableSemaphores[currentFrame]};
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eTransfer};
            vk::Semaphore signalSemaphores[] = {*renderFinishedSemaphores[currentFrame]};

            vk::SubmitInfo submitInfo(
                1, waitSemaphores, waitStages,
                1, &*commandBuffer,
                1, signalSemaphores
            );
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &submitInfo, *renderFences[currentFrame]));

            // Present: wait on renderFinishedSemaphore before displaying
            vk::PresentInfoKHR presentInfo(
                1, signalSemaphores,
                1, &*swapchain,
                &imageIndex,
                nullptr
            );
            (void)vulkanDevice.getComputeQueue().presentKHR(presentInfo);

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        std::cout << "[System] Render loop exited. Shutting down..." << std::endl;
        vulkanDevice.getDevice().waitIdle();

    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
