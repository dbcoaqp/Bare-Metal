#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>
#include <map>
#include <optional>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
};

struct TargetDevice {
    vk::PhysicalDevice physical;
    QueueFamilyIndices indices;
};

class DeviceSelector {
public:
    static TargetDevice selectOptimalDevice(const vk::Instance& instance) {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("Fatal: No Vulkan supported GPU.\n ");
        }

        // Use multimap to automatically sort in ascending order based on the score.
        std::multimap<int, TargetDevice> candidates;

        for (const auto& device : devices) {
            auto [score, indices] = evaluateDevice(device);
            if (score > 0) {
                candidates.insert({score, TargetDevice{device, indices}});
            }
        }

        if (candidates.empty()) {
            throw std::runtime_error("Fatal: None of the GPU could pass the hard indicators of "
                "the underlying computing architecture for interception.");
        }

        // rbegin() retrieves the device with the highest score.
        TargetDevice bestDevice = candidates.rbegin()->second;

        std::cout << "[System] Locked Target Device: "
                  << bestDevice.physical.getProperties().deviceName.data() << std::endl;

        return bestDevice;
    }

private:
    static std::pair<int, QueueFamilyIndices> evaluateDevice(const vk::PhysicalDevice& device) {
        int score = 0;
        QueueFamilyIndices indices{};
        vk::PhysicalDeviceProperties properties = device.getProperties();

        // 1. Need to be a Discrete GPU
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
            return {0, indices}; // Discard the cards without the calculation unit
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

// RAII wrapper for the Vulkan Logical Device and Compute Queue
class VulkanDevice {
public:
    explicit VulkanDevice(const TargetDevice& target) {
        // 1. Queue creation with priority 1.0f (compute task exclusive)
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo(
            vk::DeviceQueueCreateFlags(),
            target.indices.computeFamily.value(),
            1,
            &queuePriority
        );

        // 2. Activate Vulkan 1.3 core features: synchronization2
        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.synchronization2 = VK_TRUE;

        // 3. Device creation with pNext chain
        vk::DeviceCreateInfo deviceCreateInfo(
            vk::DeviceCreateFlags(),
            1,
            &queueCreateInfo
        );
        deviceCreateInfo.pNext = &vulkan13Features;

        // 4. Create unique device to manage lifecycle
        device = target.physical.createDeviceUnique(deviceCreateInfo);

        // 5. Update dynamic dispatcher for device-level function pointers
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device.get());

        // 6. Acquire compute queue handle
        computeQueue = device->getQueue(target.indices.computeFamily.value(), 0);
    }

    ~VulkanDevice() = default;

    vk::Queue getComputeQueue() const { return computeQueue; }
    vk::Device getDevice() const { return *device; }

private:
    vk::UniqueDevice device;
    vk::Queue computeQueue;
};

// RAII wrapper for the Vulkan Instance
class VulkanInstance {
public:
    VulkanInstance() {
        // 1. Application Info
        vk::ApplicationInfo appInfo("VulkanRenderer", 1, "No Engine", 1, VK_API_VERSION_1_3);

        // 2. Validation Layers
        const std::vector<const char*> validationLayers = {
            "VK_LAYER_KHRONOS_validation"
        };

        // Check for validation layer support
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
                throw std::runtime_error("Validation layer not found: " + std::string(layerName));
            }
        }

        // 3. Instance Creation
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        vk::InstanceCreateInfo createInfo({}, &appInfo, validationLayers, extensions);

        instance = vk::createInstanceUnique(createInfo);
    }

    ~VulkanInstance() = default; // unique_ptr handles cleanup

    vk::Instance get() const { return *instance; }

private:
    vk::UniqueInstance instance;
};

// RAII wrapper for GLFW Window
class Window {
public:
    Window(int width, int height, const char* title) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
    }

    ~Window() {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    bool shouldClose() const { return glfwWindowShouldClose(window); }
    void pollEvents() const { glfwPollEvents(); }

private:
    GLFWwindow* window;
};

int main() {
    try {
        Window window(800, 600, "Vulkan Engine");
        VulkanInstance vkInstance;

        // Initialize instance-level dynamic dispatcher
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkInstance.get());

        std::cout << "Vulkan initialized successfully with validation layers." << std::endl;

        // Select optimal physical device with compute queue family index
        TargetDevice target = DeviceSelector::selectOptimalDevice(vkInstance.get());

        // Create logical device and acquire compute queue
        VulkanDevice vulkanDevice(target);

        std::cout << "Logical Device and Compute Queue initialized." << std::endl;

        while (!window.shouldClose()) {
            window.pollEvents();
        }
    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
