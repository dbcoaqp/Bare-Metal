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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
        std::cout << "[System] Bitonic sort padded count: " << paddedCount 
                  << " (original: " << pointData.size() << ")." << std::endl;

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
        CameraUniform cameraData{};
        MathUtils::buildViewProjection(cameraData.viewProj, 0.0f, 0.0f, -5.0f, WIDTH, HEIGHT);
        cameraData.cameraPosition[0] = 0.0f;
        cameraData.cameraPosition[1] = 0.0f;
        cameraData.cameraPosition[2] = -5.0f;
        cameraData.cameraPosition[3] = 0.0f;
        cameraData.cameraForward[0] = 0.0f;
        cameraData.cameraForward[1] = 0.0f;
        cameraData.cameraForward[2] = 1.0f;
        cameraData.cameraForward[3] = 0.0f;
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

        void* cameraMapped = cameraBuffer.map();
        std::memcpy(cameraMapped, &cameraData, sizeof(CameraUniform));
        cameraBuffer.flush();
        cameraBuffer.unmap();
        std::cout << "[System] Camera buffer created." << std::endl;

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
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
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

        // 8. Shader module
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        );
        auto shaderModule = vulkanDevice.getDevice().createShaderModuleUnique(shaderModuleCreateInfo);

        // 9. Descriptor set layout
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
        };

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(layoutBindings.size()),
            layoutBindings.data()
        );

        auto descriptorSetLayout = vulkanDevice.getDevice().createDescriptorSetLayoutUnique(layoutCreateInfo);

        // 10. Descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 4),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
        };

        vk::DescriptorPoolCreateInfo poolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            1,
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data()
        );

        auto descriptorPool = vulkanDevice.getDevice().createDescriptorPoolUnique(poolCreateInfo);

        // 11. Pipeline layout (push constants: 3 x uint32_t = 12 bytes for rasterize)
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

        // 12b. bitonic_sort pipeline
        std::string bitonicShaderPath = std::string(SHADER_DIR) + "/bitonic_sort.spv";
        auto bitonicShaderCode = FileIO::readFile(bitonicShaderPath);

        vk::ShaderModuleCreateInfo bitonicSMCI(vk::ShaderModuleCreateFlags(), bitonicShaderCode.size(),
            reinterpret_cast<const uint32_t*>(bitonicShaderCode.data()));
        auto bitonicShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(bitonicSMCI);

        vk::PipelineShaderStageCreateInfo bitonicStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *bitonicShaderModule, "main");
        vk::ComputePipelineCreateInfo bitonicPipelineCI(vk::PipelineCreateFlags(), bitonicStageCI, *pipelineLayout);

        auto bitonicPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, bitonicPipelineCI);
        if (bitonicPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create bitonic sort compute pipeline.");
        }
        auto sortPipeline = std::move(bitonicPipelineResult.value);
        std::cout << "[System] Bitonic sort pipeline created." << std::endl;

        // 12c. preprocess pipeline
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

        // 12d. rasterize pipeline
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

        // 13. Descriptor set
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool, 1, &*descriptorSetLayout);
        auto descriptorSets = vulkanDevice.getDevice().allocateDescriptorSets(descriptorSetAllocateInfo);
        auto descriptorSet = descriptorSets[0];

        vk::DescriptorBufferInfo gaussianBufferInfo(gaussianBuffer.getBuffer(), 0, gaussianBufferSize);
        vk::DescriptorBufferInfo sortKeyBufferInfo(sortKeyBuffer.getBuffer(), 0, sortKeyBufferSize);
        vk::DescriptorBufferInfo cameraBufferInfo(cameraBuffer.getBuffer(), 0, sizeof(CameraUniform));
        vk::DescriptorBufferInfo gaussian2DBufferInfo(gaussian2DBuffer.getBuffer(), 0, gaussian2DBufferSize);
        vk::DescriptorBufferInfo imageBufferInfo(imageBuffer.getBuffer(), 0, imageBufferSize);

        std::vector<vk::WriteDescriptorSet> descriptorWrites = {
            vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
            vk::WriteDescriptorSet(descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
            vk::WriteDescriptorSet(descriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
            vk::WriteDescriptorSet(descriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
            vk::WriteDescriptorSet(descriptorSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr)
        };

        vulkanDevice.getDevice().updateDescriptorSets(
            static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

        // 14. Command buffer
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1);
        auto commandBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo);
        auto commandBuffer = std::move(commandBuffers[0]);

        // Acquire fence for swapchain image acquisition
        vk::FenceCreateInfo acquireFenceInfo(vk::FenceCreateFlagBits::eSignaled);
        auto acquireFence = vulkanDevice.getDevice().createFenceUnique(acquireFenceInfo);

        std::cout << "[System] Entering render loop..." << std::endl;

        // ===== Phase 9: The Hot Loop =====
        while (!glfwWindowShouldClose(window.get())) {
            glfwPollEvents();

            // (b) Acquire next swapchain image
            VK_CHECK(vulkanDevice.getDevice().waitForFences(*acquireFence, VK_TRUE, UINT64_MAX));
            vulkanDevice.getDevice().resetFences(*acquireFence);

            auto acqResult = vulkanDevice.getDevice().acquireNextImageKHR(
                *swapchain, UINT64_MAX, vk::Semaphore(), *acquireFence);
            VK_CHECK(acqResult.result);
            uint32_t imageIndex = acqResult.value;

            VK_CHECK(vulkanDevice.getDevice().waitForFences(*acquireFence, VK_TRUE, UINT64_MAX));

            // (c) Reset and begin recording
            commandBuffer->reset(vk::CommandBufferResetFlags());
            vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer->begin(beginInfo);

            // Pad buffers
            commandBuffer->fillBuffer(sortKeyBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0xBF800000);
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
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

            uint32_t groupCountX = (static_cast<uint32_t>(pointData.size()) + 255) / 256;
            commandBuffer->dispatch(groupCountX, 1, 1);

            vk::MemoryBarrier depthBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &depthBarrier, 0, nullptr, 0, nullptr);

            // Dispatch bitonic sort
            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *sortPipeline);
            uint32_t sortGroupCountX = (paddedCount + 255) / 256;
            for (uint32_t k = 2; k <= paddedCount; k <<= 1) {
                for (uint32_t j = k / 2; j > 0; j >>= 1) {
                    uint32_t pcData[2] = {k, j};
                    commandBuffer->pushConstants<uint32_t>(
                        *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pcData);
                    commandBuffer->dispatch(sortGroupCountX, 1, 1);

                    vk::MemoryBarrier sortBarrier(
                        vk::AccessFlagBits::eShaderWrite,
                        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                    commandBuffer->pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlags(), 1, &sortBarrier, 0, nullptr, 0, nullptr);
                }
            }

            // Dispatch preprocess
            vk::MemoryBarrier preprocessBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &preprocessBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *preprocessPipeline);
            uint32_t preprocessGroupCountX = (paddedCount + 255) / 256;
            commandBuffer->dispatch(preprocessGroupCountX, 1, 1);

            // Dispatch rasterize
            vk::MemoryBarrier rasterizeBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &rasterizeBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *rasterizePipeline);
            uint32_t rasterizePC[3] = {static_cast<uint32_t>(pointData.size()), WIDTH, HEIGHT};
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, rasterizePC);
            uint32_t rasterizeGroupCountX = (WIDTH + 15) / 16;
            uint32_t rasterizeGroupCountY = (HEIGHT + 15) / 16;
            commandBuffer->dispatch(rasterizeGroupCountX, rasterizeGroupCountY, 1);

            // (e) Image copy: buffer -> swapchain image
            // Barrier: compute shader writes to imageBuffer must be visible for transfer read
            vk::MemoryBarrier computeToTransferBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eTransferRead);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags(), 1, &computeToTransferBarrier, 0, nullptr, 0, nullptr);

            // Image barrier: swapchain image undefined -> transfer dst
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

            // Copy buffer to image
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

            // Image barrier: transfer dst -> present src
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

            // (f) End recording
            commandBuffer->end();

            // (g) Submit and wait
            vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &*commandBuffer, 0, nullptr);
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &submitInfo, nullptr));
            vulkanDevice.getComputeQueue().waitIdle();

            // (h) Present
            vk::PresentInfoKHR presentInfo(0, nullptr, 1, &*swapchain, &imageIndex, nullptr);
            vulkanDevice.getComputeQueue().presentKHR(presentInfo);
        }

        std::cout << "[System] Render loop exited. Shutting down..." << std::endl;
        vulkanDevice.getDevice().waitIdle();

    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
