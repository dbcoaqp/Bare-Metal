#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <memory>

// RAII wrapper for the Vulkan Instance
class VulkanInstance {
public:
    VulkanInstance() {
        // 1. Application Info
        vk::ApplicationInfo appInfo("VulkanRenderer", 1, "No Engine", 1, VK_API_VERSION_1_2);

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

        std::cout << "Vulkan initialized successfully with validation layers." << std::endl;

        while (!window.shouldClose()) {
            window.pollEvents();
        }
    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
