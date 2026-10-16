#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
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
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// Dear ImGui (Phase 19b)
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#ifdef _WIN32
#include <nvtx3/nvToolsExt.h>
#ifdef MemoryBarrier
#undef MemoryBarrier
#endif
#else
// NVTX stubs for non-Windows (macOS/Linux)
inline void nvtxRangePushA(const char*) {}
inline void nvtxRangePop() {}
#endif

#include "utils/FileIO.h"
#include "utils/VkCheck.h"
#include "core/VmaBuffer.h"
#include "core/GaussianData.h"
#include "core/Camera.h"
#include "utils/NpyParser.h"
#include "core/Gaussian2D.h"

// Global function pointer table memory allocation
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

// GLFW callbacks accumulate input; the render loop applies it to the camera.
static double lastMouseX = 0.0;
static double lastMouseY = 0.0;
static float pendingMouseX = 0.0f;
static float pendingMouseY = 0.0f;
static bool keyForward = false;
static bool keyBackward = false;
static bool keyLeft = false;
static bool keyRight = false;
static bool uiMode = false;  // Phase 19b: false = Camera Mode, true = UI Mode

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (!uiMode) {
        pendingMouseX += static_cast<float>(xpos - lastMouseX);
        pendingMouseY += static_cast<float>(lastMouseY - ypos);
    }
    lastMouseX = xpos;
    lastMouseY = ypos;
}

static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    const bool pressed = action != GLFW_RELEASE;
    if (key == GLFW_KEY_W) keyForward = pressed;
    if (key == GLFW_KEY_S) keyBackward = pressed;
    if (key == GLFW_KEY_A) keyLeft = pressed;
    if (key == GLFW_KEY_D) keyRight = pressed;

    // Phase 19b: TAB toggles between Camera Mode and UI Mode
    if (key == GLFW_KEY_TAB && action == GLFW_PRESS) {
        uiMode = !uiMode;
        if (uiMode) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            // Re-anchor mouse position to prevent camera snap
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
            pendingMouseX = 0.0f;
            pendingMouseY = 0.0f;
        }
    }
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

        // 2. Must support Compute AND Graphics Queue (Graphics needed for ImGui overlay)
        std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
            if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute) &&
                (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                queueFamilies[i].timestampValidBits > 0) {
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

        vk::PhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.shaderInt64 = VK_TRUE;

        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };

        vk::DeviceCreateInfo deviceCreateInfo(
            vk::DeviceCreateFlags(),
            1,
            &queueCreateInfo,
            0, nullptr,
            static_cast<uint32_t>(deviceExtensions.size()),
            deviceExtensions.data(),
            &deviceFeatures
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

        glfwSetInputMode(window.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetCursorPosCallback(window.get(), cursorPosCallback);
        glfwSetKeyCallback(window.get(), keyCallback);
        glfwGetCursorPos(window.get(), &lastMouseX, &lastMouseY);

        WindowSurface surface(vkInstance.get(), window.get());

        TargetDevice target = DeviceSelector::selectOptimalDevice(vkInstance.get(), surface.get());

        VulkanDevice vulkanDevice(vkInstance.get(), target);
        std::cout << "[System] Logical Device and Compute Queue successfully engaged." << std::endl;
        std::cout << "[System] VMA Allocator successfully engaged." << std::endl;
        constexpr uint32_t TIMESTAMP_COUNT = 8;
        const float timestampPeriod = target.physical.getProperties().limits.timestampPeriod;
        const uint32_t timestampValidBits = target.physical.getQueueFamilyProperties()
            [target.indices.computeFamily.value()].timestampValidBits;
        auto timestampQueryPool = vulkanDevice.getDevice().createQueryPoolUnique(
            vk::QueryPoolCreateInfo({}, vk::QueryType::eTimestamp, TIMESTAMP_COUNT));
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
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,  // Phase 19b: ImGui needs ColorAttachment
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

        // ===== Phase 19b: Swapchain ImageViews (needed for ImGui framebuffers) =====
        std::vector<vk::UniqueImageView> swapchainImageViews;
        for (size_t i = 0; i < swapchainImages.size(); ++i) {
            vk::ImageViewCreateInfo viewCreateInfo(
                vk::ImageViewCreateFlags(),
                swapchainImages[i],
                vk::ImageViewType::e2D,
                chosenFormat.format,
                vk::ComponentMapping(),
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
            );
            swapchainImageViews.push_back(vulkanDevice.getDevice().createImageViewUnique(viewCreateInfo));
        }
        std::cout << "[System] Created " << swapchainImageViews.size() << " swapchain image views." << std::endl;

        // ===== Phase 19b: ImGui RenderPass =====
        // loadOp = eLoad: MUST preserve the DMA-copied 3DGS image
        // finalLayout = ePresentSrcKHR: auto-transition for presentation
        vk::AttachmentDescription imguiAttachment(
            vk::AttachmentDescriptionFlags(),
            chosenFormat.format,
            vk::SampleCountFlagBits::e1,
            vk::AttachmentLoadOp::eLoad,       // Preserve compute-rasterized image
            vk::AttachmentStoreOp::eStore,
            vk::AttachmentLoadOp::eDontCare,   // stencil
            vk::AttachmentStoreOp::eDontCare,  // stencil
            vk::ImageLayout::eColorAttachmentOptimal,  // initialLayout
            vk::ImageLayout::ePresentSrcKHR            // finalLayout (auto-transition)
        );

        vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass(
            vk::SubpassDescriptionFlags(),
            vk::PipelineBindPoint::eGraphics,
            0, nullptr,       // input attachments
            1, &colorRef,     // color attachments
            nullptr,          // resolve
            nullptr,          // depth
            0, nullptr        // preserve
        );

        vk::SubpassDependency imguiDep(
            VK_SUBPASS_EXTERNAL, 0,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
            vk::DependencyFlags()
        );

        vk::RenderPassCreateInfo imguiRenderPassCI(
            vk::RenderPassCreateFlags(),
            1, &imguiAttachment,
            1, &subpass,
            1, &imguiDep
        );

        auto imguiRenderPass = vulkanDevice.getDevice().createRenderPassUnique(imguiRenderPassCI);
        std::cout << "[System] ImGui RenderPass created." << std::endl;

        // ===== Phase 19b: ImGui Framebuffers (one per swapchain image) =====
        std::vector<vk::UniqueFramebuffer> imguiFramebuffers;
        for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
            vk::ImageView attachments[] = { *swapchainImageViews[i] };
            vk::FramebufferCreateInfo fbCI(
                vk::FramebufferCreateFlags(),
                *imguiRenderPass,
                1, attachments,
                WIDTH, HEIGHT, 1
            );
            imguiFramebuffers.push_back(vulkanDevice.getDevice().createFramebufferUnique(fbCI));
        }
        std::cout << "[System] Created " << imguiFramebuffers.size() << " ImGui framebuffers." << std::endl;

        // ===== Phase 19b: ImGui Descriptor Pool =====
        vk::DescriptorPoolSize imguiPoolSizes[] = {
            { vk::DescriptorType::eCombinedImageSampler, 1 }
        };
        vk::DescriptorPoolCreateInfo imguiPoolCI(
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            1,
            1, imguiPoolSizes
        );
        auto imguiDescriptorPool = vulkanDevice.getDevice().createDescriptorPoolUnique(imguiPoolCI);

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

        // ============================================================
        // Phase 15: Tile grid parameters
        // ============================================================
        const uint32_t TILE_SIZE = 16;
        uint32_t numTilesX = (WIDTH + TILE_SIZE - 1) / TILE_SIZE;
        uint32_t numTilesY = (HEIGHT + TILE_SIZE - 1) / TILE_SIZE;
        std::cout << "[System] Tile grid: " << numTilesX << " x " << numTilesY
                  << " (" << numTilesX * numTilesY << " tiles total, tile size = "
                  << TILE_SIZE << "px)." << std::endl;

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

        Camera camera;
        const glm::vec3 initialForward = camera.getForward();

        cameraData.cameraPosition[0] = camera.position.x;
        cameraData.cameraPosition[1] = camera.position.y;
        cameraData.cameraPosition[2] = camera.position.z;
        cameraData.cameraPosition[3] = 0.0f;
        cameraData.cameraForward[0] = initialForward.x;
        cameraData.cameraForward[1] = initialForward.y;
        cameraData.cameraForward[2] = initialForward.z;
        cameraData.cameraForward[3] = 0.0f;

        const float fovY = glm::radians(57.29578f);
        const float aspect = static_cast<float>(WIDTH) / static_cast<float>(HEIGHT);
        const glm::mat4 initialProjection = glm::perspectiveRH_ZO(fovY, aspect, 0.1f, 100.0f);
        glm::mat4 initialViewProj = initialProjection * camera.getViewMatrix();
        initialViewProj[1][1] *= -1.0f;
        std::memcpy(cameraData.viewProj, &initialViewProj, sizeof(cameraData.viewProj));

        // Derive focal length and tan(half-fov) from the actual FOV — NEVER hardcode!
        // fovY = 57.29578° ≈ 1.0 rad  →  tan(0.5) ≈ 0.5463  →  focal_y ≈ HEIGHT / (2 * tan(fovY/2))
        const float tanHalfFovY = std::tan(fovY * 0.5f);
        const float tanHalfFovX = tanHalfFovY * aspect;
        cameraData.focal_y = static_cast<float>(HEIGHT) / (2.0f * tanHalfFovY);
        cameraData.focal_x = static_cast<float>(WIDTH)  / (2.0f * tanHalfFovX);
        cameraData.tan_fovx = tanHalfFovX;
        cameraData.tan_fovy = tanHalfFovY;
        cameraData.viewport[0] = static_cast<float>(WIDTH);
        cameraData.viewport[1] = static_cast<float>(HEIGHT);
        cameraData.filterDilation = 0.3f;  // Phase 19b: default low-pass filter
        cameraData.scaleModifier = 1.0f;   // Phase 19b: default scale multiplier

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
                  << initialForward.x << ", " << initialForward.y << ", " << initialForward.z << ")" << std::endl;

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

        // 7g. Touched tiles buffer for Phase 13 tile bounding box verification
        vk::DeviceSize touchedTilesBufferSize = static_cast<vk::DeviceSize>(paddedCount) * sizeof(uint32_t);
        VmaBuffer touchedTilesBuffer(
            vulkanDevice.getAllocator(),
            touchedTilesBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Touched tiles buffer created (" << touchedTilesBufferSize << " bytes)." << std::endl;

        // ============================================================
        // Phase 15: Tile prefix sum buffer
        // (paddedCount + 1) elements: [0..paddedCount-1] = exclusive prefix sum,
        // [paddedCount] = totalTileKeys
        // ============================================================
        vk::DeviceSize tilePrefixSumBufferSize = static_cast<vk::DeviceSize>(paddedCount + 1) * sizeof(uint32_t);
        VmaBuffer tilePrefixSumBuffer(
            vulkanDevice.getAllocator(),
            tilePrefixSumBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile prefix sum buffer created (" << tilePrefixSumBufferSize << " bytes)." << std::endl;

        // ============================================================
        // Phase 15: Tile sort key buffers (allocated with initial guess,
        // will be dynamically re-allocated after tile_scan readback)
        // Initial allocation: 8x paddedCount as a safe upper bound
        // ============================================================
        uint32_t tileSortKeyBufferElementCount = paddedCount * 8;  // Conservative initial guess
        vk::DeviceSize tileSortKeyBufferSize = static_cast<vk::DeviceSize>(tileSortKeyBufferElementCount) * sizeof(TileSortKeyPair);

        auto tileSortKeyBuffer = std::make_unique<VmaBuffer>(
            vulkanDevice.getAllocator(),
            tileSortKeyBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile sort key buffer allocated (" << tileSortKeyBufferSize << " bytes, "
                  << tileSortKeyBufferElementCount << " elements)." << std::endl;

        auto tileSortKeyBufferAlt = std::make_unique<VmaBuffer>(
            vulkanDevice.getAllocator(),
            tileSortKeyBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile sort key buffer alt allocated (" << tileSortKeyBufferSize << " bytes, ping-pong)." << std::endl;

        // ============================================================
        // Phase 15: Tile histogram & offset buffers for 64-bit radix sort
        // ============================================================
        uint32_t tileHistGroupCount = (tileSortKeyBufferElementCount + 255) / 256;
        vk::DeviceSize tileHistogramBufferSize = static_cast<vk::DeviceSize>(tileHistGroupCount) * 256 * sizeof(uint32_t);

        VmaBuffer tileHistogramBuffer(
            vulkanDevice.getAllocator(),
            tileHistogramBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile radix histogram buffer created (" << tileHistogramBufferSize
                  << " bytes, " << tileHistGroupCount << " workgroups x 256 buckets)." << std::endl;

        VmaBuffer tileGlobalOffsetBuffer(
            vulkanDevice.getAllocator(),
            globalOffsetBufferSize,  // 256 uint32_t
            vk::BufferUsageFlagBits::eStorageBuffer,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile radix global offset buffer created (" << globalOffsetBufferSize << " bytes)." << std::endl;

        // ============================================================
        // Phase 16: Tile pointers buffer
        // uvec2 per tile: .x = Start index, .y = End index (exclusive)
        // ============================================================
        uint32_t numTilesTotal = numTilesX * numTilesY;
        vk::DeviceSize tilePointersBufferSize = static_cast<vk::DeviceSize>(numTilesTotal) * sizeof(uint32_t) * 2;
        VmaBuffer tilePointersBuffer(
            vulkanDevice.getAllocator(),
            tilePointersBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Tile pointers buffer created (" << tilePointersBufferSize
                  << " bytes, " << numTilesTotal << " tiles x uvec2)." << std::endl;

        // 8. Shader module
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        );
        auto shaderModule = vulkanDevice.getDevice().createShaderModuleUnique(shaderModuleCreateInfo);

        // 9. Descriptor set layout (15 bindings)
        // 0:  GaussianBuffer (SSBO)
        // 1:  SortKeyBuffer (SSBO, src for 32-bit radix)
        // 2:  CameraUBO (UBO)
        // 3:  Gaussian2DBuffer (SSBO)
        // 4:  ImageBuffer (SSBO)
        // 5:  SortKeyBufferAlt (SSBO, dst for 32-bit radix)
        // 6:  HistogramBuffer (SSBO, 32-bit radix)
        // 7:  GlobalOffsetBuffer (SSBO, 32-bit radix)
        // 8:  TouchedTilesBuffer (SSBO)
        // 9:  TilePrefixSumBuffer (SSBO)
        // 10: TileSortKeyBuffer (SSBO, src for 64-bit tile radix)
        // 11: TileSortKeyBufferAlt (SSBO, dst for 64-bit tile radix)
        // 12: TileHistogramBuffer (SSBO, 64-bit tile radix)
        // 13: TileGlobalOffsetBuffer (SSBO, 64-bit tile radix)
        // 14: TilePointersBuffer (SSBO)
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(6, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(7, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(8, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(9, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(10, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(11, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(12, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(13, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(14, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),  // TilePointersBuffer
        };

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(layoutBindings.size()),
            layoutBindings.data()
        );

        auto descriptorSetLayout = vulkanDevice.getDevice().createDescriptorSetLayoutUnique(layoutCreateInfo);

        // 10. Descriptor pool (3 descriptor sets: mainDS + swapDS + tileSwapDS, 15 bindings each)
        // (tileMainDS and tilePtrDS were byte-identical clones of mainDS — eliminated)
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 45),  // 15 storage buffers x 3 sets
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 3)    // 1 uniform buffer x 3 sets
        };

        vk::DescriptorPoolCreateInfo poolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            3,  // maxSets: mainDS + swapDS + tileSwapDS
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

        // 12b. radix_histogram pipeline (32-bit, reads SortKeyPair from binding 1)
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

        // 12c. radix_scan pipeline (32-bit, reads histogram from binding 6, writes offsets to binding 7)
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

        // 12d. radix_scatter pipeline (32-bit, reads SortKeyPair from binding 1, writes to binding 5)
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

        // ============================================================
        // Phase 15: tile_scan pipeline
        // ============================================================
        std::string tileScanShaderPath = std::string(SHADER_DIR) + "/tile_scan.spv";
        auto tileScanShaderCode = FileIO::readFile(tileScanShaderPath);

        vk::ShaderModuleCreateInfo tileScanSMCI(vk::ShaderModuleCreateFlags(), tileScanShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tileScanShaderCode.data()));
        auto tileScanShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tileScanSMCI);

        vk::PipelineShaderStageCreateInfo tileScanStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tileScanShaderModule, "main");
        vk::ComputePipelineCreateInfo tileScanPipelineCI(vk::PipelineCreateFlags(), tileScanStageCI, *pipelineLayout);

        auto tileScanPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tileScanPipelineCI);
        if (tileScanPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_scan compute pipeline.");
        }
        auto tileScanPipeline = std::move(tileScanPipelineResult.value);
        std::cout << "[System] Tile scan pipeline created." << std::endl;

        // ============================================================
        // Phase 15: tile_duplicate pipeline
        // ============================================================
        std::string tileDuplicateShaderPath = std::string(SHADER_DIR) + "/tile_duplicate.spv";
        auto tileDuplicateShaderCode = FileIO::readFile(tileDuplicateShaderPath);

        vk::ShaderModuleCreateInfo tileDuplicateSMCI(vk::ShaderModuleCreateFlags(), tileDuplicateShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tileDuplicateShaderCode.data()));
        auto tileDuplicateShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tileDuplicateSMCI);

        vk::PipelineShaderStageCreateInfo tileDuplicateStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tileDuplicateShaderModule, "main");
        vk::ComputePipelineCreateInfo tileDuplicatePipelineCI(vk::PipelineCreateFlags(), tileDuplicateStageCI, *pipelineLayout);

        auto tileDuplicatePipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tileDuplicatePipelineCI);
        if (tileDuplicatePipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_duplicate compute pipeline.");
        }
        auto tileDuplicatePipeline = std::move(tileDuplicatePipelineResult.value);
        std::cout << "[System] Tile duplicate pipeline created." << std::endl;

        // ============================================================
        // Phase 15: tile_radix_histogram pipeline (64-bit, reads TileSortKeyPair from binding 10)
        // ============================================================
        std::string tileRadixHistogramShaderPath = std::string(SHADER_DIR) + "/tile_radix_histogram.spv";
        auto tileRadixHistogramShaderCode = FileIO::readFile(tileRadixHistogramShaderPath);

        vk::ShaderModuleCreateInfo tileRadixHistogramSMCI(vk::ShaderModuleCreateFlags(), tileRadixHistogramShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tileRadixHistogramShaderCode.data()));
        auto tileRadixHistogramShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tileRadixHistogramSMCI);

        vk::PipelineShaderStageCreateInfo tileRadixHistogramStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tileRadixHistogramShaderModule, "main");
        vk::ComputePipelineCreateInfo tileRadixHistogramPipelineCI(vk::PipelineCreateFlags(), tileRadixHistogramStageCI, *pipelineLayout);

        auto tileRadixHistogramPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tileRadixHistogramPipelineCI);
        if (tileRadixHistogramPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_radix_histogram compute pipeline.");
        }
        auto tileRadixHistogramPipeline = std::move(tileRadixHistogramPipelineResult.value);
        std::cout << "[System] Tile radix histogram pipeline created." << std::endl;

        // ============================================================
        // Phase 15: tile_radix_scan pipeline (64-bit, reads histogram from binding 12)
        // ============================================================
        std::string tileRadixScanShaderPath = std::string(SHADER_DIR) + "/tile_radix_scan.spv";
        auto tileRadixScanShaderCode = FileIO::readFile(tileRadixScanShaderPath);

        vk::ShaderModuleCreateInfo tileRadixScanSMCI(vk::ShaderModuleCreateFlags(), tileRadixScanShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tileRadixScanShaderCode.data()));
        auto tileRadixScanShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tileRadixScanSMCI);

        vk::PipelineShaderStageCreateInfo tileRadixScanStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tileRadixScanShaderModule, "main");
        vk::ComputePipelineCreateInfo tileRadixScanPipelineCI(vk::PipelineCreateFlags(), tileRadixScanStageCI, *pipelineLayout);

        auto tileRadixScanPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tileRadixScanPipelineCI);
        if (tileRadixScanPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_radix_scan compute pipeline.");
        }
        auto tileRadixScanPipeline = std::move(tileRadixScanPipelineResult.value);
        std::cout << "[System] Tile radix scan pipeline created." << std::endl;

        // ============================================================
        // Phase 15: tile_radix_scatter pipeline (64-bit, reads TileSortKeyPair from binding 10, writes to binding 11)
        // ============================================================
        std::string tileRadixScatterShaderPath = std::string(SHADER_DIR) + "/tile_radix_scatter.spv";
        auto tileRadixScatterShaderCode = FileIO::readFile(tileRadixScatterShaderPath);

        vk::ShaderModuleCreateInfo tileRadixScatterSMCI(vk::ShaderModuleCreateFlags(), tileRadixScatterShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tileRadixScatterShaderCode.data()));
        auto tileRadixScatterShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tileRadixScatterSMCI);

        vk::PipelineShaderStageCreateInfo tileRadixScatterStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tileRadixScatterShaderModule, "main");
        vk::ComputePipelineCreateInfo tileRadixScatterPipelineCI(vk::PipelineCreateFlags(), tileRadixScatterStageCI, *pipelineLayout);

        auto tileRadixScatterPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tileRadixScatterPipelineCI);
        if (tileRadixScatterPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_radix_scatter compute pipeline.");
        }
        auto tileRadixScatterPipeline = std::move(tileRadixScatterPipelineResult.value);
        std::cout << "[System] Tile radix scatter pipeline created." << std::endl;

        // ============================================================
        // Phase 16: tile_pointers pipeline
        // ============================================================
        std::string tilePointersShaderPath = std::string(SHADER_DIR) + "/tile_pointers.spv";
        auto tilePointersShaderCode = FileIO::readFile(tilePointersShaderPath);

        vk::ShaderModuleCreateInfo tilePointersSMCI(vk::ShaderModuleCreateFlags(), tilePointersShaderCode.size(),
            reinterpret_cast<const uint32_t*>(tilePointersShaderCode.data()));
        auto tilePointersShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(tilePointersSMCI);

        vk::PipelineShaderStageCreateInfo tilePointersStageCI(
            vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute, *tilePointersShaderModule, "main");
        vk::ComputePipelineCreateInfo tilePointersPipelineCI(vk::PipelineCreateFlags(), tilePointersStageCI, *pipelineLayout);

        auto tilePointersPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(nullptr, tilePointersPipelineCI);
        if (tilePointersPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create tile_pointers compute pipeline.");
        }
        auto tilePointersPipeline = std::move(tilePointersPipelineResult.value);
        std::cout << "[System] Tile pointers pipeline created." << std::endl;

        // 13. Descriptor sets (3 sets: mainDS + swapDS + tileSwapDS)
        std::vector<vk::DescriptorSetLayout> setLayouts = {
            *descriptorSetLayout, *descriptorSetLayout, *descriptorSetLayout};
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(*descriptorPool, static_cast<uint32_t>(setLayouts.size()), setLayouts.data());
        auto descriptorSets = vulkanDevice.getDevice().allocateDescriptorSets(descriptorSetAllocateInfo);
        auto mainDS = descriptorSets[0];
        auto swapDS = descriptorSets[1];
        auto tileSwapDS = descriptorSets[2];

        // Helper lambda to update all descriptor sets
        auto updateAllDescriptorSets = [&]() {
            vk::DescriptorBufferInfo gaussianBufferInfo(gaussianBuffer.getBuffer(), 0, gaussianBufferSize);
            vk::DescriptorBufferInfo sortKeyBufferInfo(sortKeyBuffer.getBuffer(), 0, sortKeyBufferSize);
            vk::DescriptorBufferInfo cameraBufferInfo(cameraBuffer.getBuffer(), 0, sizeof(CameraUniform));
            vk::DescriptorBufferInfo gaussian2DBufferInfo(gaussian2DBuffer.getBuffer(), 0, gaussian2DBufferSize);
            vk::DescriptorBufferInfo imageBufferInfo(imageBuffer.getBuffer(), 0, imageBufferSize);
            vk::DescriptorBufferInfo sortKeyBufferAltInfo(sortKeyBufferAlt.getBuffer(), 0, sortKeyBufferSize);
            vk::DescriptorBufferInfo histogramBufferInfo(histogramBuffer.getBuffer(), 0, histogramBufferSize);
            vk::DescriptorBufferInfo globalOffsetBufferInfo(globalOffsetBuffer.getBuffer(), 0, globalOffsetBufferSize);
            vk::DescriptorBufferInfo touchedTilesBufferInfo(touchedTilesBuffer.getBuffer(), 0, touchedTilesBufferSize);
            vk::DescriptorBufferInfo tilePrefixSumBufferInfo(tilePrefixSumBuffer.getBuffer(), 0, tilePrefixSumBufferSize);
            vk::DescriptorBufferInfo tileSortKeyBufferInfo(tileSortKeyBuffer->getBuffer(), 0, tileSortKeyBufferSize);
            vk::DescriptorBufferInfo tileSortKeyBufferAltInfo(tileSortKeyBufferAlt->getBuffer(), 0, tileSortKeyBufferSize);
            vk::DescriptorBufferInfo tileHistogramBufferInfo(tileHistogramBuffer.getBuffer(), 0, tileHistogramBufferSize);
            vk::DescriptorBufferInfo tileGlobalOffsetBufferInfo(tileGlobalOffsetBuffer.getBuffer(), 0, globalOffsetBufferSize);
            vk::DescriptorBufferInfo tilePointersBufferInfo(tilePointersBuffer.getBuffer(), 0, tilePointersBufferSize);

            std::vector<vk::WriteDescriptorSet> allWrites = {
                // mainDS: 32-bit depth sort ping-pong (binding 1=src, binding 5=dst)
                vk::WriteDescriptorSet(mainDS, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &globalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 8, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &touchedTilesBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 9, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePrefixSumBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 10, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 11, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 12, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileHistogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 13, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileGlobalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(mainDS, 14, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePointersBufferInfo, nullptr),
                // swapDS: 32-bit depth sort ping-pong (binding 1=dst, binding 5=src — SWAPPED)
                vk::WriteDescriptorSet(swapDS, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &globalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 8, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &touchedTilesBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 9, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePrefixSumBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 10, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 11, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 12, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileHistogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 13, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileGlobalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(swapDS, 14, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePointersBufferInfo, nullptr),
                // tileSwapDS: 64-bit tile radix ping-pong (binding 10=dst, binding 11=src — SWAPPED)
                vk::WriteDescriptorSet(tileSwapDS, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussianBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &cameraBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &gaussian2DBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &imageBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 5, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &sortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 6, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &histogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 7, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &globalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 8, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &touchedTilesBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 9, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePrefixSumBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 10, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferAltInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 11, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileSortKeyBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 12, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileHistogramBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 13, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tileGlobalOffsetBufferInfo, nullptr),
                vk::WriteDescriptorSet(tileSwapDS, 14, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &tilePointersBufferInfo, nullptr),
            };

            vulkanDevice.getDevice().updateDescriptorSets(
                static_cast<uint32_t>(allWrites.size()), allWrites.data(), 0, nullptr);
        };

        updateAllDescriptorSets();

        // 14. Command buffer
        // A single frame in flight protects the shared camera UBO and timestamp query pool.
        const uint32_t MAX_FRAMES_IN_FLIGHT = 1;
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, MAX_FRAMES_IN_FLIGHT);
        auto commandBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo);

        // Phase 10: Vulkan traffic lights (semaphores)
        std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
        std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
        std::vector<vk::UniqueFence> renderFences;

        vk::SemaphoreCreateInfo semaphoreCreateInfo;
        vk::FenceCreateInfo renderFenceInfo(vk::FenceCreateFlagBits::eSignaled);

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            imageAvailableSemaphores.push_back(vulkanDevice.getDevice().createSemaphoreUnique(semaphoreCreateInfo));
            renderFinishedSemaphores.push_back(vulkanDevice.getDevice().createSemaphoreUnique(semaphoreCreateInfo));
            renderFences.push_back(vulkanDevice.getDevice().createFenceUnique(renderFenceInfo));
        }
        

        uint32_t currentFrame = 0;
        std::cout << "[System] " << MAX_FRAMES_IN_FLIGHT << " sets of Synchronization traffic lights created." << std::endl;

        std::cout << "[System] Entering render loop..." << std::endl;

        // ===== Phase 19b: Initialize Dear ImGui =====
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan(window.get(), true);

        ImGui_ImplVulkan_InitInfo imguiInitInfo = {};
        imguiInitInfo.Instance = static_cast<VkInstance>(vkInstance.get());
        imguiInitInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(target.physical);
        imguiInitInfo.Device = static_cast<VkDevice>(vulkanDevice.getDevice());
        imguiInitInfo.QueueFamily = target.indices.computeFamily.value();
        imguiInitInfo.Queue = static_cast<VkQueue>(vulkanDevice.getComputeQueue());
        imguiInitInfo.PipelineCache = VK_NULL_HANDLE;
        imguiInitInfo.DescriptorPool = static_cast<VkDescriptorPool>(*imguiDescriptorPool);
        imguiInitInfo.Subpass = 0;
        imguiInitInfo.MinImageCount = minImageCount;
        imguiInitInfo.ImageCount = static_cast<uint32_t>(swapchainImages.size());
        imguiInitInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        imguiInitInfo.RenderPass = static_cast<VkRenderPass>(*imguiRenderPass);

        ImGui_ImplVulkan_Init(&imguiInitInfo);

        // Upload ImGui fonts
        {
            auto fontCmdBufs = vulkanDevice.getDevice().allocateCommandBuffersUnique(
                vk::CommandBufferAllocateInfo(*commandPool, vk::CommandBufferLevel::ePrimary, 1));
            fontCmdBufs[0]->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
            ImGui_ImplVulkan_CreateFontsTexture();
            fontCmdBufs[0]->end();
            vk::SubmitInfo fontSubmit(0, nullptr, nullptr, 1, &*fontCmdBufs[0], 0, nullptr);
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &fontSubmit, nullptr));
            vulkanDevice.getComputeQueue().waitIdle();
        }
        std::cout << "[System] Dear ImGui initialized with Vulkan + GLFW backend." << std::endl;

        // ===== Phase 19b: Visible count readback buffer (4 bytes, persistently mapped) =====
        VmaBuffer visibleCountReadbackBuffer(
            vulkanDevice.getAllocator(),
            sizeof(uint32_t),
            vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_TO_CPU
        );
        uint32_t* mappedVisibleCount = static_cast<uint32_t*>(visibleCountReadbackBuffer.map());
        *mappedVisibleCount = 0;
        uint32_t lastVisibleCount = 0;
        double stageTimesMs[7] = {};
        double totalGpuTimeMs = 0.0;
        bool isFirstFrame = true;

        // Phase 19b: HUD slider variables (host-side, written to UBO each frame)
        float filterDilation = 0.3f;
        float scaleModifier = 1.0f;

        // ===== Phase 10: The Async Hot Loop (Radix Sort) =====
        auto previousFrameTime = std::chrono::steady_clock::now();
        while (!glfwWindowShouldClose(window.get())) {
            glfwPollEvents();

            const auto currentTime = std::chrono::steady_clock::now();
            const float deltaTime = std::chrono::duration<float>(currentTime - previousFrameTime).count();
            previousFrameTime = currentTime;

            camera.processMouse(pendingMouseX, pendingMouseY);
            pendingMouseX = 0.0f;
            pendingMouseY = 0.0f;
            const glm::vec3 movement(
                uiMode ? 0.0f : static_cast<float>(keyRight) - static_cast<float>(keyLeft),
                0.0f,
                uiMode ? 0.0f : static_cast<float>(keyForward) - static_cast<float>(keyBackward));
            camera.processKeyboard(movement, deltaTime);

            // Wait for the specific frame's GPU work to complete
            VK_CHECK(vulkanDevice.getDevice().waitForFences(*renderFences[currentFrame], VK_TRUE, UINT64_MAX));
            vulkanDevice.getDevice().resetFences(*renderFences[currentFrame]);

            if (!isFirstFrame) {
                visibleCountReadbackBuffer.invalidate();
                lastVisibleCount = *mappedVisibleCount;

                uint64_t timestamps[TIMESTAMP_COUNT] = {};
                const vk::Result queryResult = vulkanDevice.getDevice().getQueryPoolResults(
                    *timestampQueryPool, 0, TIMESTAMP_COUNT,
                    sizeof(timestamps), timestamps, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
                if (queryResult == vk::Result::eSuccess) {
                    const uint64_t validMask = timestampValidBits >= 64
                        ? UINT64_MAX
                        : ((uint64_t{1} << timestampValidBits) - 1);
                    for (uint32_t stage = 0; stage < 7; ++stage) {
                        const uint64_t elapsedTicks = (timestamps[stage + 1] - timestamps[stage]) & validMask;
                        stageTimesMs[stage] = static_cast<double>(elapsedTicks) * timestampPeriod / 1.0e6;
                    }
                    const uint64_t totalTicks = (timestamps[7] - timestamps[0]) & validMask;
                    totalGpuTimeMs = static_cast<double>(totalTicks) * timestampPeriod / 1.0e6;
                }
            }

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
            ImGui::SetNextWindowBgAlpha(0.82f);
            ImGui::Begin("Bare-Metal 3DGS Telemetry [Phase 20]", nullptr,
                ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoCollapse);
            ImGui::Text("Mode: %s  [TAB] Toggle UI / Camera Mode", uiMode ? "UI" : "Camera");
            const float safeDeltaTime = std::max(deltaTime, 1.0e-6f);
            ImGui::Text("CPU Frame: %.3f ms  |  FPS: %.1f", deltaTime * 1000.0f, 1.0f / safeDeltaTime);

            VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
            vmaGetHeapBudgets(vulkanDevice.getAllocator(), budgets);
            const vk::PhysicalDeviceMemoryProperties memoryProperties = target.physical.getMemoryProperties();
            vk::DeviceSize deviceLocalUsage = 0;
            vk::DeviceSize deviceLocalBudget = 0;
            for (uint32_t heap = 0; heap < memoryProperties.memoryHeapCount; ++heap) {
                if (memoryProperties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                    deviceLocalUsage += budgets[heap].usage;
                    deviceLocalBudget += budgets[heap].budget;
                }
            }
            const float vramFraction = deviceLocalBudget > 0
                ? std::clamp(static_cast<float>(deviceLocalUsage) / static_cast<float>(deviceLocalBudget), 0.0f, 1.0f)
                : 0.0f;
            constexpr double BYTES_PER_MEGABYTE = 1024.0 * 1024.0;
            ImGui::Text("VRAM: %.2f MB / %.2f MB",
                static_cast<double>(deviceLocalUsage) / BYTES_PER_MEGABYTE,
                static_cast<double>(deviceLocalBudget) / BYTES_PER_MEGABYTE);
            ImGui::ProgressBar(vramFraction, ImVec2(280.0f, 0.0f));

            const uint32_t totalGaussianCount = static_cast<uint32_t>(pointData.size());
            const uint32_t culledCount = totalGaussianCount > lastVisibleCount
                ? totalGaussianCount - lastVisibleCount
                : 0;
            const float cullingEfficiency = totalGaussianCount > 0
                ? static_cast<float>(culledCount) / static_cast<float>(totalGaussianCount)
                : 0.0f;
            ImGui::Separator();
            ImGui::Text("Frustum Culling: Total %u  |  Visible %u  |  Culled %u (%.1f%%)",
                totalGaussianCount, lastVisibleCount, culledCount, cullingEfficiency * 100.0f);

            static constexpr const char* GPU_STAGE_NAMES[7] = {
                "Stage A (Pad + Depth)",
                "Stage B (32b Radix Sort)",
                "Stage C (Preprocess + Cov2D)",
                "Stage E (Tile Duplicate)",
                "Stage F (64b Tile Radix Sort)",
                "Stage F.5 (Tile Pointers)",
                "Stage G (Rasterize + Copy)"
            };
            ImGui::Separator();
            ImGui::Text("Phase 20 GPU Hardware Timestamps: %.3f ms", totalGpuTimeMs);
            for (uint32_t stage = 0; stage < 7; ++stage) {
                ImGui::Text("%s: %.3f ms", GPU_STAGE_NAMES[stage], stageTimesMs[stage]);
                const float stageFraction = totalGpuTimeMs > 0.0
                    ? static_cast<float>(stageTimesMs[stage] / totalGpuTimeMs)
                    : 0.0f;
                ImGui::ProgressBar(std::clamp(stageFraction, 0.0f, 1.0f), ImVec2(280.0f, 0.0f));
            }

            ImGui::Separator();
            ImGui::TextUnformatted("Kernel Sliders");
            ImGui::SliderFloat("Low-Pass Filter (+eps)", &filterDilation, 0.0f, 5.0f, "%.3f");
            ImGui::SliderFloat("Scale Modifier", &scaleModifier, 0.1f, 3.0f, "%.2f");
            ImGui::End();
            ImGui::Render();

            const glm::vec3 forward = camera.getForward();
            mappedCameraData->cameraPosition[0] = camera.position.x;
            mappedCameraData->cameraPosition[1] = camera.position.y;
            mappedCameraData->cameraPosition[2] = camera.position.z;
            mappedCameraData->cameraPosition[3] = 0.0f;
            mappedCameraData->cameraForward[0] = forward.x;
            mappedCameraData->cameraForward[1] = forward.y;
            mappedCameraData->cameraForward[2] = forward.z;
            mappedCameraData->cameraForward[3] = 0.0f;

            glm::mat4 viewProj = initialProjection * camera.getViewMatrix();
            viewProj[1][1] *= -1.0f;
            std::memcpy(mappedCameraData->viewProj, &viewProj, sizeof(mappedCameraData->viewProj));
            mappedCameraData->filterDilation = filterDilation;
            mappedCameraData->scaleModifier = scaleModifier;
            cameraBuffer.flush();
            
            uint32_t imageIndex;
            try {
                auto acqResult = vulkanDevice.getDevice().acquireNextImageKHR(
                    *swapchain, UINT64_MAX, *imageAvailableSemaphores[currentFrame], nullptr);
                if (acqResult.result != vk::Result::eSuccess && acqResult.result != vk::Result::eSuboptimalKHR) {
                    throw std::runtime_error("Acquire image failed");
                }
                imageIndex = acqResult.value;
            } catch (const vk::OutOfDateKHRError& e) {
                std::cout << "[System] Warning: Window resized/Swapchain out of date. Exiting loop safely." << std::endl;
                break; 
            }

            // Extract the command scroll specific to the current frame
            auto& commandBuffer = commandBuffers[currentFrame];

            // Record command buffer
            commandBuffer->reset(vk::CommandBufferResetFlags());
            vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffer->begin(beginInfo);

            commandBuffer->resetQueryPool(*timestampQueryPool, 0, TIMESTAMP_COUNT);
            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *timestampQueryPool, 0);

            // ====== Stage A: Buffer Pad & Depth Calc ======
            nvtxRangePushA("1. Buffer Pad & Depth Calc");

            // Pad buffers
            commandBuffer->fillBuffer(sortKeyBuffer.getBuffer(), 0, VK_WHOLE_SIZE, ~0u);
            commandBuffer->fillBuffer(gaussian2DBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0);
            commandBuffer->fillBuffer(touchedTilesBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0);
            commandBuffer->fillBuffer(tilePrefixSumBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0);
            //commandBuffer->fillBuffer(tileSortKeyBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
            commandBuffer->fillBuffer(tileSortKeyBuffer->getBuffer(), 0, VK_WHOLE_SIZE, ~0u);
            commandBuffer->fillBuffer(tileSortKeyBufferAlt->getBuffer(), 0, VK_WHOLE_SIZE, ~0u);
            commandBuffer->fillBuffer(tilePointersBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0);

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

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 1);

            nvtxRangePop();

            // ====== Stage B: 32-bit Radix Sort O(N) ======
            nvtxRangePushA("2. Radix Sort O(N)");

            for (uint32_t pass = 0; pass < 4; pass++) {
                uint32_t shift = pass * 8;
                bool isEvenPass = (pass % 2 == 0);
                vk::DescriptorSet currentDS = isEvenPass ? mainDS : swapDS;

                commandBuffer->bindDescriptorSets(
                    vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &currentDS, 0, nullptr);

                // --- Histogram Pass (32-bit: reads SortKeyPair from binding 1) ---
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

                // --- Scan Pass (32-bit: reads histogram from binding 6, writes to binding 7) ---
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

                // --- Scatter Pass (32-bit: reads SortKeyPair from binding 1, writes to binding 5) ---
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

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 2);

            // Rebind mainDS for downstream shaders (sorted data is in sortKeyBuffer after 4 passes)
            commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &mainDS, 0, nullptr);

            nvtxRangePop();

            // ====== Stage C: Covariance Preprocess ======
            nvtxRangePushA("3. Covariance Preprocess");

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *preprocessPipeline);
            uint32_t preprocessGroupCountX = (paddedCount + 255) / 256;
            commandBuffer->dispatch(preprocessGroupCountX, 1, 1);

            vk::MemoryBarrier preprocessBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &preprocessBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 3);

            nvtxRangePop();
            /*
            // ====== Stage D: Tile Scan (Exclusive Prefix Sum) ======
            nvtxRangePushA("4. Tile Scan (Prefix Sum)");

            vk::MemoryBarrier tileScanBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &tileScanBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tileScanPipeline);
            uint32_t tileScanPC[3] = {paddedCount, 0, 0};
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tileScanPC);
            uint32_t tileScanGroupCount = (paddedCount + 255) / 256;
            commandBuffer->dispatch(tileScanGroupCount, 1, 1);

            vk::MemoryBarrier tileScanWriteBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &tileScanWriteBarrier, 0, nullptr, 0, nullptr);

            nvtxRangePop();
            */
            // ====== Stage E: Tile Duplicate (Space Explosion) ======
            nvtxRangePushA("5. Tile Duplicate (Space Explosion)");

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tileDuplicatePipeline);
            uint32_t tileDupPC[3] = {static_cast<uint32_t>(pointData.size()), numTilesX, numTilesY};
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tileDupPC);
            commandBuffer->dispatch(groupCountX, 1, 1);

            vk::MemoryBarrier tileDupBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &tileDupBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 4);

            nvtxRangePop();

            // ====== Stage F: 64-bit Tile Radix Sort ======
            nvtxRangePushA("6. 64-bit Tile Radix Sort");

            // 8 passes for 64-bit key (low bytes first = depth, high bytes = tileID)
            // Uses dedicated tile radix shaders reading TileSortKeyPair from binding 10/11
            uint32_t tileHistGroupCountActual = (tileSortKeyBufferElementCount + 255) / 256;

            for (uint32_t pass = 0; pass < 8; pass++) {
                uint32_t shift = pass * 8;
                bool isEvenPass = (pass % 2 == 0);
                vk::DescriptorSet currentTileDS = isEvenPass ? mainDS : tileSwapDS;

                commandBuffer->bindDescriptorSets(
                    vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &currentTileDS, 0, nullptr);

                // --- Tile Histogram Pass (64-bit: reads TileSortKeyPair from binding 10) ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tileRadixHistogramPipeline);
                uint32_t tileHistPC[3] = {shift, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tileHistPC);
                commandBuffer->dispatch(tileHistGroupCountActual, 1, 1);

                vk::MemoryBarrier tileHistBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &tileHistBarrier, 0, nullptr, 0, nullptr);

                // --- Tile Scan Pass (64-bit: reads histogram from binding 12) ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tileRadixScanPipeline);
                uint32_t tileScanPC2[3] = {tileHistGroupCountActual, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tileScanPC2);
                commandBuffer->dispatch(1, 1, 1);

                vk::MemoryBarrier tileScanPassBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &tileScanPassBarrier, 0, nullptr, 0, nullptr);

                // --- Tile Scatter Pass (64-bit: reads TileSortKeyPair from binding 10, writes to binding 11) ---
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tileRadixScatterPipeline);
                uint32_t tileScatterPC[3] = {shift, 0, 0};
                commandBuffer->pushConstants<uint32_t>(
                    *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tileScatterPC);
                commandBuffer->dispatch(tileHistGroupCountActual, 1, 1);

                vk::MemoryBarrier tileScatterBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(), 1, &tileScatterBarrier, 0, nullptr, 0, nullptr);
            }

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 5);

            nvtxRangePop();

            // ====== Stage F.5: Tile Pointers Extraction ======
            nvtxRangePushA("6.5 Tile Pointers Extraction");

            // (Barrier already issued by last tileScatterBarrier in Stage F — no redundant double-barrier)

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *tilePointersPipeline);
            commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &mainDS, 0, nullptr);
            uint32_t tilePtrPC[3] = {paddedCount, 0, 0};
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, tilePtrPC);
            uint32_t tilePtrGroupCount = (tileSortKeyBufferElementCount + 255) / 256;
            commandBuffer->dispatch(tilePtrGroupCount, 1, 1);

            vk::MemoryBarrier tilePtrBarrier(
                vk::AccessFlagBits::eShaderWrite,
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                vk::DependencyFlags(), 1, &tilePtrBarrier, 0, nullptr, 0, nullptr);

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, *timestampQueryPool, 6);

            nvtxRangePop();

            // ====== Stage G: Compute Rasterization ======
            nvtxRangePushA("7. Compute Rasterization");

            // (Barrier already issued by tilePtrBarrier in Stage F.5 — no redundant double-barrier)

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *rasterizePipeline);
            commandBuffer->bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, 1, &mainDS, 0, nullptr);
            uint32_t rasterizePC[3] = { WIDTH, HEIGHT, numTilesX };
            commandBuffer->pushConstants<uint32_t>(
                *pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, rasterizePC);
            /*
            uint32_t rasterizeGroupCountX = (WIDTH + 15) / 16;
            uint32_t rasterizeGroupCountY = (HEIGHT + 15) / 16;
            commandBuffer->dispatch(rasterizeGroupCountX, rasterizeGroupCountY, 1);
            */
            commandBuffer->dispatch(numTilesX, numTilesY, 1);

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

            commandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eTransfer, *timestampQueryPool, 7);

            // Defer telemetry readback until the transfer stage so the compute stages stay contiguous.
            const vk::BufferCopy visibleCountCopy(
                static_cast<vk::DeviceSize>(paddedCount) * sizeof(uint32_t), 0, sizeof(uint32_t));
            commandBuffer->copyBuffer(
                tilePrefixSumBuffer.getBuffer(), visibleCountReadbackBuffer.getBuffer(), visibleCountCopy);
            vk::BufferMemoryBarrier visibleCountHostBarrier(
                vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                visibleCountReadbackBuffer.getBuffer(), 0, sizeof(uint32_t));
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                vk::DependencyFlags(), 0, nullptr, 1, &visibleCountHostBarrier, 0, nullptr);

            vk::ImageMemoryBarrier transferToColorBarrier(
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eColorAttachmentOptimal,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                swapchainImages[imageIndex],
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            commandBuffer->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eColorAttachmentOutput,
                vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &transferToColorBarrier);

            vk::RenderPassBeginInfo imguiBeginInfo(
                *imguiRenderPass,
                *imguiFramebuffers[imageIndex],
                vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(WIDTH, HEIGHT)),
                0, nullptr);
            commandBuffer->beginRenderPass(imguiBeginInfo, vk::SubpassContents::eInline);
            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), static_cast<VkCommandBuffer>(*commandBuffer));
            commandBuffer->endRenderPass();

            nvtxRangePop();

            commandBuffer->end();

            // Single submit: wait on imageAvailableSemaphore, signal renderFinishedSemaphore + renderFence
            vk::Semaphore waitSemaphores[] = {*imageAvailableSemaphores[currentFrame]};
            vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eTopOfPipe};
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

            try {
                (void)vulkanDevice.getComputeQueue().presentKHR(presentInfo);
            } catch (const vk::OutOfDateKHRError& e) {
                std::cout << "[System] Warning: Swapchain out of date on Present. Exiting loop safely." << std::endl;
                break; 
            }

            currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
            isFirstFrame = false;
        }

        std::cout << "[System] Render loop exited. Shutting down..." << std::endl;
        VK_CHECK(vulkanDevice.getDevice().waitIdle());
        visibleCountReadbackBuffer.unmap();
        cameraBuffer.unmap();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        // ============================================================
        // Phase 13: Reverse DMA - Readback touchedTilesBuffer to CPU
        // ============================================================
        {
            VmaBuffer touchedTilesReadback(
                vulkanDevice.getAllocator(),
                touchedTilesBufferSize,
                vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_GPU_TO_CPU
            );

            vk::CommandBufferAllocateInfo readbackCmdBufInfo(
                *commandPool, vk::CommandBufferLevel::ePrimary, 1);
            auto readbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(readbackCmdBufInfo);
            auto& readbackCmdBuffer = readbackCmdBuffers[0];

            vk::CommandBufferBeginInfo readbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            readbackCmdBuffer->begin(readbackBeginInfo);

            vk::BufferCopy readbackCopyRegion(0, 0, touchedTilesBufferSize);
            readbackCmdBuffer->copyBuffer(
                touchedTilesBuffer.getBuffer(),
                touchedTilesReadback.getBuffer(),
                readbackCopyRegion
            );

            readbackCmdBuffer->end();

            vk::SubmitInfo readbackSubmitInfo(0, nullptr, nullptr, 1, &*readbackCmdBuffer, 0, nullptr);
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &readbackSubmitInfo, nullptr));
            vulkanDevice.getComputeQueue().waitIdle();

            uint32_t* touchedData = static_cast<uint32_t*>(touchedTilesReadback.map());

            std::cout << "\n[System] === Touched Tiles Verification ===" << std::endl;
            std::cout << "[System] Total gaussians: " << pointData.size()
                      << ", Padded count: " << paddedCount << std::endl;

            uint32_t printCount = std::min(static_cast<uint32_t>(10), static_cast<uint32_t>(pointData.size()));
            for (uint32_t i = 0; i < printCount; ++i) {
                std::cout << "[System] Gaussian[" << i << "]: touchedTiles = " << touchedData[i] << std::endl;
            }
            std::cout << "[System] === Touched Tiles Verification Complete ===" << std::endl;

            touchedTilesReadback.unmap();
        }

        // ============================================================
        // Phase 15: Reverse DMA - Readback totalTileKeys & first 20 keys
        // ============================================================
        {
            // 1. Readback prefix sum to get totalTileKeys
            VmaBuffer prefixSumReadback(
                vulkanDevice.getAllocator(),
                tilePrefixSumBufferSize,
                vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_GPU_TO_CPU
            );

            vk::CommandBufferAllocateInfo prefixReadbackCmdBufInfo(
                *commandPool, vk::CommandBufferLevel::ePrimary, 1);
            auto prefixReadbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(prefixReadbackCmdBufInfo);
            auto& prefixReadbackCmdBuffer = prefixReadbackCmdBuffers[0];

            vk::CommandBufferBeginInfo prefixReadbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            prefixReadbackCmdBuffer->begin(prefixReadbackBeginInfo);

            vk::BufferCopy prefixCopyRegion(0, 0, tilePrefixSumBufferSize);
            prefixReadbackCmdBuffer->copyBuffer(
                tilePrefixSumBuffer.getBuffer(),
                prefixSumReadback.getBuffer(),
                prefixCopyRegion
            );

            prefixReadbackCmdBuffer->end();

            vk::SubmitInfo prefixSubmitInfo(0, nullptr, nullptr, 1, &*prefixReadbackCmdBuffer, 0, nullptr);
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &prefixSubmitInfo, nullptr));
            vulkanDevice.getComputeQueue().waitIdle();

            uint32_t* prefixData = static_cast<uint32_t*>(prefixSumReadback.map());
            //uint32_t totalTileKeys = prefixData[paddedCount];
            uint32_t totalTileKeys = prefixData[0];

            std::cout << "\n[System] === Phase 15: Tile Explosion Verification ===" << std::endl;
            std::cout << "[System] totalTileKeys (total exploded copies): " << totalTileKeys << std::endl;
            std::cout << "[System] Expansion ratio: "
                      << (pointData.size() > 0 ? static_cast<float>(totalTileKeys) / static_cast<float>(pointData.size()) : 0.0f)
                      << "x" << std::endl;

            uint32_t prefixPrintCount = std::min(static_cast<uint32_t>(10), static_cast<uint32_t>(pointData.size()));
            for (uint32_t i = 0; i < prefixPrintCount; ++i) {
                std::cout << "[System] tilePrefixSum[" << i << "] = " << prefixData[i] << std::endl;
            }

            prefixSumReadback.unmap();

            // 2. Readback tile sort keys (first 20 elements)
            if (totalTileKeys > 0) {
                uint32_t readbackKeyCount = std::min(totalTileKeys, 20u);
                vk::DeviceSize keyReadbackSize = static_cast<vk::DeviceSize>(readbackKeyCount) * sizeof(TileSortKeyPair);

                VmaBuffer tileKeysReadback(
                    vulkanDevice.getAllocator(),
                    keyReadbackSize,
                    vk::BufferUsageFlagBits::eTransferDst,
                    VMA_MEMORY_USAGE_GPU_TO_CPU
                );

                vk::CommandBufferAllocateInfo keyReadbackCmdBufInfo(
                    *commandPool, vk::CommandBufferLevel::ePrimary, 1);
                auto keyReadbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(keyReadbackCmdBufInfo);
                auto& keyReadbackCmdBuffer = keyReadbackCmdBuffers[0];

                vk::CommandBufferBeginInfo keyReadbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
                keyReadbackCmdBuffer->begin(keyReadbackBeginInfo);

                vk::BufferCopy keyCopyRegion(0, 0, keyReadbackSize);
                keyReadbackCmdBuffer->copyBuffer(
                    tileSortKeyBuffer->getBuffer(),
                    tileKeysReadback.getBuffer(),
                    keyCopyRegion
                );

                keyReadbackCmdBuffer->end();

                vk::SubmitInfo keySubmitInfo(0, nullptr, nullptr, 1, &*keyReadbackCmdBuffer, 0, nullptr);
                VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &keySubmitInfo, nullptr));
                vulkanDevice.getComputeQueue().waitIdle();

                TileSortKeyPair* keyData = static_cast<TileSortKeyPair*>(tileKeysReadback.map());

                std::cout << "\n[System] === First " << readbackKeyCount << " 64-bit Tile Sort Keys ===" << std::endl;
                for (uint32_t i = 0; i < readbackKeyCount; ++i) {
                    uint64_t k = keyData[i].key;
                    uint32_t tileID = static_cast<uint32_t>(k >> 32);
                    uint32_t depthBits = static_cast<uint32_t>(k & 0xFFFFFFFFu);
                    float depth;
                    std::memcpy(&depth, &depthBits, sizeof(float));
                    uint32_t gaussIdx = keyData[i].index;

                    uint32_t tileY = numTilesX > 0 ? tileID / numTilesX : 0;
                    uint32_t tileX = numTilesX > 0 ? tileID % numTilesX : 0;

                    std::cout << "[System] Key[" << i << "]: tileID=" << tileID
                              << " (tileX=" << tileX << ", tileY=" << tileY << ")"
                              << ", depthBits=0x" << std::hex << std::setw(8) << std::setfill('0') << depthBits << std::dec
                              << ", depth=" << depth
                              << ", gaussianIndex=" << gaussIdx << std::endl;
                }
                std::cout << "[System] === Tile Sort Key Verification Complete ===" << std::endl;

                tileKeysReadback.unmap();
            }
        }

        // ============================================================
        // Phase 16: Reverse DMA - Readback TilePointersBuffer
        // ============================================================
        {
            VmaBuffer tilePointersReadback(
                vulkanDevice.getAllocator(),
                tilePointersBufferSize,
                vk::BufferUsageFlagBits::eTransferDst,
                VMA_MEMORY_USAGE_GPU_TO_CPU
            );

            vk::CommandBufferAllocateInfo ptrReadbackCmdBufInfo(
                *commandPool, vk::CommandBufferLevel::ePrimary, 1);
            auto ptrReadbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(ptrReadbackCmdBufInfo);
            auto& ptrReadbackCmdBuffer = ptrReadbackCmdBuffers[0];

            vk::CommandBufferBeginInfo ptrReadbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            ptrReadbackCmdBuffer->begin(ptrReadbackBeginInfo);

            vk::BufferCopy ptrCopyRegion(0, 0, tilePointersBufferSize);
            ptrReadbackCmdBuffer->copyBuffer(
                tilePointersBuffer.getBuffer(),
                tilePointersReadback.getBuffer(),
                ptrCopyRegion
            );

            ptrReadbackCmdBuffer->end();

            vk::SubmitInfo ptrSubmitInfo(0, nullptr, nullptr, 1, &*ptrReadbackCmdBuffer, 0, nullptr);
            VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &ptrSubmitInfo, nullptr));
            vulkanDevice.getComputeQueue().waitIdle();

            uint32_t* ptrData = static_cast<uint32_t*>(tilePointersReadback.map());

            std::cout << "\n[System] === Phase 16: Tile Pointers Verification ===" << std::endl;
            std::cout << "[System] Tile grid: " << numTilesX << " x " << numTilesY
                      << " (" << numTilesTotal << " tiles)." << std::endl;

            // Print Tile[444] and Tile[447]
            uint32_t tileIndices[] = {444, 447};
            for (uint32_t ti = 0; ti < 2; ++ti) {
                uint32_t tileIdx = tileIndices[ti];
                if (tileIdx < numTilesTotal) {
                    uint32_t startX = ptrData[tileIdx * 2 + 0];
                    uint32_t endX   = ptrData[tileIdx * 2 + 1];
                    uint32_t tileY = numTilesX > 0 ? tileIdx / numTilesX : 0;
                    uint32_t tileX = numTilesX > 0 ? tileIdx % numTilesX : 0;
                    std::cout << "[System] Tile[" << tileIdx << "] (tileX=" << tileX
                              << ", tileY=" << tileY << "): [Start, End) = ["
                              << startX << ", " << endX << ")"
                              << ", count = " << (endX - startX) << std::endl;
                } else {
                    std::cout << "[System] Tile[" << tileIdx << "]: OUT OF RANGE (max="
                              << numTilesTotal - 1 << ")." << std::endl;
                }
            }

            // Also print a few non-zero tiles to verify correctness
            std::cout << "\n[System] --- First 10 non-empty tiles ---" << std::endl;
            uint32_t nonEmptyCount = 0;
            for (uint32_t t = 0; t < numTilesTotal && nonEmptyCount < 10; ++t) {
                uint32_t start = ptrData[t * 2 + 0];
                uint32_t end   = ptrData[t * 2 + 1];
                if (end > start) {
                    uint32_t tY = numTilesX > 0 ? t / numTilesX : 0;
                    uint32_t tX = numTilesX > 0 ? t % numTilesX : 0;
                    std::cout << "[System] Tile[" << t << "] (tileX=" << tX
                              << ", tileY=" << tY << "): [" << start << ", " << end << ")"
                              << ", count = " << (end - start) << std::endl;
                    nonEmptyCount++;
                }
            }
            if (nonEmptyCount == 0) {
                std::cout << "[System] WARNING: No non-empty tiles found! All pointers are [0, 0)." << std::endl;
            }

            std::cout << "[System] === Phase 16: Tile Pointers Verification Complete ===" << std::endl;

            tilePointersReadback.unmap();
        }

    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
