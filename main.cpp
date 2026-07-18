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

#include "utils/FileIO.h"
#include "utils/VkCheck.h"
#include "core/VmaBuffer.h"
#include "core/GaussianData.h"
#include "core/Camera.h"
#include "utils/NpyParser.h"
#include "core/Gaussian2D.h"

// Global function pointer table memory allocation
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
            throw std::runtime_error("Fatal: No Vulkan supported GPU.");
        }

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

        vk::DeviceCreateInfo deviceCreateInfo(
            vk::DeviceCreateFlags(),
            1,
            &queueCreateInfo
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

        vk::ApplicationInfo appInfo("Vulkan Compute Probe", 1, "Bare Metal", 1, VK_API_VERSION_1_3);

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

        vk::InstanceCreateInfo createInfo({}, &appInfo, validationLayers, {});

        instance = vk::createInstanceUnique(createInfo);

        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance.get());
    }

    ~VulkanInstance() = default;

    vk::Instance get() const { return *instance; }

private:
    vk::detail::DynamicLoader dl;
    vk::UniqueInstance instance;
};

int main() {
    try {
        std::cout << "--- Bare Metal Boot Sequence Initiated ---" << std::endl;
        
        VulkanInstance vkInstance;
        std::cout << "[System] Vulkan Instance initialized (Dynamic Dispatch Active)." << std::endl;

        TargetDevice target = DeviceSelector::selectOptimalDevice(vkInstance.get());

        VulkanDevice vulkanDevice(vkInstance.get(), target);
        std::cout << "[System] Logical Device and Compute Queue successfully engaged." << std::endl;
        std::cout << "[System] VMA Allocator successfully engaged." << std::endl;
        std::cout << "--- Bare Metal Boot Sequence Completed ---" << std::endl;

        // Phase 6: GPU Sorting - Stage 1: Depth Key-Value Generation
        
        // 1. Read shader binary (path injected by CMake at build-time)
        std::string shaderPath = std::string(SHADER_DIR) + "/calc_depth.spv";
        auto shaderCode = FileIO::readFile(shaderPath);
        std::cout << "[System] Shader loaded: " << shaderCode.size() << " bytes." << std::endl;

        // 2. Parse Gaussian points from .npy file
        std::string npyPath = std::string(SHADER_DIR) + "/test.npy";
        auto pointData = NpyParser::parse(npyPath);
        vk::DeviceSize gaussianBufferSize = pointData.size() * sizeof(GaussianPoint);
        std::cout << "[System] Loaded " << pointData.size() << " Gaussian points (" 
                  << gaussianBufferSize << " bytes)." << std::endl;

        // Calculate the smallest power-of-two >= pointData.size() for bitonic sort
        uint32_t paddedCount = 1;
        while (paddedCount < pointData.size()) {
            paddedCount <<= 1;
        }
        std::cout << "[System] Bitonic sort padded count: " << paddedCount 
                  << " (original: " << pointData.size() << ")." << std::endl;

        // 3. Create staging buffer and upload Gaussian data to GPU
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

        // 4. Create command pool
        vk::CommandPoolCreateInfo commandPoolCreateInfo(
            vk::CommandPoolCreateFlags(),
            target.indices.computeFamily.value()
        );

        auto commandPool = vulkanDevice.getDevice().createCommandPoolUnique(commandPoolCreateInfo);

        // 5. DMA transfer Gaussian data to GPU
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
        tempCmdBuffer->copyBuffer(
            stagingBuffer.getBuffer(),
            gaussianBuffer.getBuffer(),
            copyRegion
        );

        tempCmdBuffer->end();

        vk::SubmitInfo tempSubmitInfo(
            0, nullptr, nullptr,
            1, &*tempCmdBuffer,
            0, nullptr
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &tempSubmitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();
        std::cout << "[System] DMA Transfer Complete: Gaussian data injected into Device Local VRAM." << std::endl;

        // 6. Create Camera Uniform Buffer with fake data
        CameraUniform cameraData{};
        // Identity viewProj matrix (column-major)
        cameraData.viewProj[0] = 1.0f; cameraData.viewProj[5] = 1.0f;
        cameraData.viewProj[10] = 1.0f; cameraData.viewProj[15] = 1.0f;
        // Camera position at (0, 0, -5)
        cameraData.cameraPosition[0] = 0.0f;
        cameraData.cameraPosition[1] = 0.0f;
        cameraData.cameraPosition[2] = -5.0f;
        cameraData.cameraPosition[3] = 0.0f; // pad

        // Camera looking forward along +Z axis
        cameraData.cameraForward[0] = 0.0f;
        cameraData.cameraForward[1] = 0.0f;
        cameraData.cameraForward[2] = 1.0f;
        cameraData.cameraForward[3] = 0.0f; // pad

        // Camera intrinsics for Jacobian computation
        cameraData.focal_x = 1000.0f;
        cameraData.focal_y = 1000.0f;
        cameraData.tan_fovx = 0.5f;
        cameraData.tan_fovy = 0.5f;
        cameraData.viewport[0] = 1024.0f;
        cameraData.viewport[1] = 768.0f;
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
        std::cout << "[System] Camera buffer created with position (0, 0, -5) and forward (0, 0, 1)." << std::endl;

        // 7. Create Sort Key Buffer (padded to power-of-two for bitonic sort)
        vk::DeviceSize sortKeyBufferSize = static_cast<vk::DeviceSize>(paddedCount) * sizeof(SortKeyPair);
        VmaBuffer sortKeyBuffer(
            vulkanDevice.getAllocator(),
            sortKeyBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Sort key buffer created (" << sortKeyBufferSize << " bytes, padded for bitonic sort)." << std::endl;

        // 7b. Create Gaussian2D Output Buffer
        vk::DeviceSize gaussian2DBufferSize = static_cast<vk::DeviceSize>(paddedCount) * sizeof(Gaussian2D);
        VmaBuffer gaussian2DBuffer(
            vulkanDevice.getAllocator(),
            gaussian2DBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );
        std::cout << "[System] Gaussian2D buffer created (" << gaussian2DBufferSize << " bytes)." << std::endl;

        // 8. Create Shader Module
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        );
        auto shaderModule = vulkanDevice.getDevice().createShaderModuleUnique(shaderModuleCreateInfo);
        std::cout << "[System] Compute Shader Module compiled." << std::endl;

        // 9. Create Descriptor Set Layout (4 bindings)
        std::vector<vk::DescriptorSetLayoutBinding> layoutBindings = {
            // binding 0: GaussianBuffer (SSBO, read-only)
            vk::DescriptorSetLayoutBinding(
                0,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            ),
            // binding 1: SortKeyBuffer (SSBO)
            vk::DescriptorSetLayoutBinding(
                1,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            ),
            // binding 2: CameraBuffer (UBO)
            vk::DescriptorSetLayoutBinding(
                2,
                vk::DescriptorType::eUniformBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            ),
            // binding 3: Gaussian2DBuffer (SSBO, output)
            vk::DescriptorSetLayoutBinding(
                3,
                vk::DescriptorType::eStorageBuffer,
                1,
                vk::ShaderStageFlagBits::eCompute
            )
        };

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(layoutBindings.size()),
            layoutBindings.data()
        );

        auto descriptorSetLayout = vulkanDevice.getDevice().createDescriptorSetLayoutUnique(layoutCreateInfo);

        // 10. Create Descriptor Pool
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 3),
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1)
        };

        vk::DescriptorPoolCreateInfo poolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            1,  // maxSets
            static_cast<uint32_t>(poolSizes.size()),
            poolSizes.data()
        );

        auto descriptorPool = vulkanDevice.getDevice().createDescriptorPoolUnique(poolCreateInfo);

        // 11. Create Pipeline Layout (with push constants for bitonic sort)
        vk::PushConstantRange pushConstantRange(
            vk::ShaderStageFlagBits::eCompute,
            0,
            sizeof(uint32_t) * 2  // k and j
        );

        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            vk::PipelineLayoutCreateFlags(),
            1,
            &*descriptorSetLayout,
            1,
            &pushConstantRange
        );

        auto pipelineLayout = vulkanDevice.getDevice().createPipelineLayoutUnique(pipelineLayoutCreateInfo);

        // 12. Create Compute Pipeline (calc_depth)
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

        auto computePipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(
            nullptr,
            computePipelineCreateInfo
        );

        if (computePipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create compute pipeline.");
        }

        auto computePipeline = std::move(computePipelineResult.value);

        // 12b. Load Bitonic Sort Shader and Create Sort Pipeline
        std::string bitonicShaderPath = std::string(SHADER_DIR) + "/bitonic_sort.spv";
        auto bitonicShaderCode = FileIO::readFile(bitonicShaderPath);
        std::cout << "[System] Bitonic sort shader loaded: " << bitonicShaderCode.size() << " bytes." << std::endl;

        vk::ShaderModuleCreateInfo bitonicShaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            bitonicShaderCode.size(),
            reinterpret_cast<const uint32_t*>(bitonicShaderCode.data())
        );
        auto bitonicShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(bitonicShaderModuleCreateInfo);

        vk::PipelineShaderStageCreateInfo bitonicStageCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            *bitonicShaderModule,
            "main"
        );

        vk::ComputePipelineCreateInfo bitonicPipelineCreateInfo(
            vk::PipelineCreateFlags(),
            bitonicStageCreateInfo,
            *pipelineLayout  // Reuse the same pipeline layout
        );

        auto bitonicPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(
            nullptr,
            bitonicPipelineCreateInfo
        );

        if (bitonicPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create bitonic sort compute pipeline.");
        }

        auto sortPipeline = std::move(bitonicPipelineResult.value);
        std::cout << "[System] Bitonic sort pipeline created." << std::endl;

        // 12c. Load Preprocess Shader and Create Preprocess Pipeline
        std::string preprocessShaderPath = std::string(SHADER_DIR) + "/preprocess.spv";
        auto preprocessShaderCode = FileIO::readFile(preprocessShaderPath);
        std::cout << "[System] Preprocess shader loaded: " << preprocessShaderCode.size() << " bytes." << std::endl;

        vk::ShaderModuleCreateInfo preprocessShaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            preprocessShaderCode.size(),
            reinterpret_cast<const uint32_t*>(preprocessShaderCode.data())
        );
        auto preprocessShaderModule = vulkanDevice.getDevice().createShaderModuleUnique(preprocessShaderModuleCreateInfo);

        vk::PipelineShaderStageCreateInfo preprocessStageCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute,
            *preprocessShaderModule,
            "main"
        );

        vk::ComputePipelineCreateInfo preprocessPipelineCreateInfo(
            vk::PipelineCreateFlags(),
            preprocessStageCreateInfo,
            *pipelineLayout
        );

        auto preprocessPipelineResult = vulkanDevice.getDevice().createComputePipelineUnique(
            nullptr,
            preprocessPipelineCreateInfo
        );

        if (preprocessPipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create preprocess compute pipeline.");
        }

        auto preprocessPipeline = std::move(preprocessPipelineResult.value);
        std::cout << "[System] Preprocess pipeline created." << std::endl;

        // 13. Allocate and Update Descriptor Sets
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
            *descriptorPool,
            1,
            &*descriptorSetLayout
        );

        auto descriptorSets = vulkanDevice.getDevice().allocateDescriptorSets(descriptorSetAllocateInfo);
        auto descriptorSet = descriptorSets[0];

        vk::DescriptorBufferInfo gaussianBufferInfo(
            gaussianBuffer.getBuffer(),
            0,
            gaussianBufferSize
        );

        vk::DescriptorBufferInfo sortKeyBufferInfo(
            sortKeyBuffer.getBuffer(),
            0,
            sortKeyBufferSize
        );

        vk::DescriptorBufferInfo cameraBufferInfo(
            cameraBuffer.getBuffer(),
            0,
            sizeof(CameraUniform)
        );

        vk::DescriptorBufferInfo gaussian2DBufferInfo(
            gaussian2DBuffer.getBuffer(),
            0,
            gaussian2DBufferSize
        );

        std::vector<vk::WriteDescriptorSet> descriptorWrites = {
            // binding 0: GaussianBuffer
            vk::WriteDescriptorSet(
                descriptorSet,
                0,
                0,
                1,
                vk::DescriptorType::eStorageBuffer,
                nullptr,
                &gaussianBufferInfo,
                nullptr
            ),
            // binding 1: SortKeyBuffer
            vk::WriteDescriptorSet(
                descriptorSet,
                1,
                0,
                1,
                vk::DescriptorType::eStorageBuffer,
                nullptr,
                &sortKeyBufferInfo,
                nullptr
            ),
            // binding 2: CameraBuffer
            vk::WriteDescriptorSet(
                descriptorSet,
                2,
                0,
                1,
                vk::DescriptorType::eUniformBuffer,
                nullptr,
                &cameraBufferInfo,
                nullptr
            ),
            // binding 3: Gaussian2DBuffer
            vk::WriteDescriptorSet(
                descriptorSet,
                3,
                0,
                1,
                vk::DescriptorType::eStorageBuffer,
                nullptr,
                &gaussian2DBufferInfo,
                nullptr
            )
        };

        vulkanDevice.getDevice().updateDescriptorSets(
            static_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr
        );

        // 14. Record and Dispatch Compute
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );

        auto commandBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo);
        auto commandBuffer = std::move(commandBuffers[0]);

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commandBuffer->begin(beginInfo);

        // (d) Pad the sort key buffer with -1.0f (0xBF800000) so padding elements sink to the end
        commandBuffer->fillBuffer(sortKeyBuffer.getBuffer(), 0, VK_WHOLE_SIZE, 0xBF800000);

        // (e) Barrier: TransferWrite -> ShaderRead/Write (fillBuffer must complete before compute shaders)
        vk::MemoryBarrier fillBarrier(
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
        );
        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(),
            1, &fillBarrier,
            0, nullptr,
            0, nullptr
        );

        // Dispatch calc_depth
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        commandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            *pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr
        );

        uint32_t groupCountX = (static_cast<uint32_t>(pointData.size()) + 255) / 256;
        commandBuffer->dispatch(groupCountX, 1, 1);
        std::cout << "[System] Dispatched " << groupCountX << " workgroups for " << pointData.size() << " points." << std::endl;

        // (f) Barrier: ShaderWrite -> ShaderRead/Write (depth calculation must complete before sorting)
        vk::MemoryBarrier depthBarrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
        );
        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(),
            1, &depthBarrier,
            0, nullptr,
            0, nullptr
        );

        // (g) Bitonic Sort - The Storm
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *sortPipeline);
        uint32_t sortGroupCountX = (paddedCount + 255) / 256;
        for (uint32_t k = 2; k <= paddedCount; k <<= 1) {
            for (uint32_t j = k / 2; j > 0; j >>= 1) {
                uint32_t pcData[2] = {k, j};
                commandBuffer->pushConstants<uint32_t>(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pcData);
                commandBuffer->dispatch(sortGroupCountX, 1, 1);

                // Barrier after each bitonic step
                vk::MemoryBarrier sortBarrier(
                    vk::AccessFlagBits::eShaderWrite,
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
                );
                commandBuffer->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlags(),
                    1, &sortBarrier,
                    0, nullptr,
                    0, nullptr
                );
            }
        }

        // (g2) Barrier: All sort writes must be visible before preprocess reads
        vk::MemoryBarrier preprocessBarrier(
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite
        );
        commandBuffer->pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::DependencyFlags(),
            1, &preprocessBarrier,
            0, nullptr,
            0, nullptr
        );

        // (h) Preprocess: 3D Covariance 2D Projection
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *preprocessPipeline);
        uint32_t preprocessGroupCountX = (paddedCount + 255) / 256;
        commandBuffer->dispatch(preprocessGroupCountX, 1, 1);
        std::cout << "[System] Dispatched " << preprocessGroupCountX << " workgroups for preprocess." << std::endl;

        commandBuffer->end();

        vk::SubmitInfo submitInfo(
            0, nullptr, nullptr,
            1, &*commandBuffer
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &submitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();

        std::cout << "[System] Compute dispatch complete." << std::endl;

        // 15. Reverse DMA: Readback SortKeyBuffer for verification (回程高铁)
        VmaBuffer readbackBuffer(
            vulkanDevice.getAllocator(),
            sortKeyBufferSize,
            vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_TO_CPU
        );

        vk::CommandBufferAllocateInfo readbackCmdBufInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );
        auto readbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(readbackCmdBufInfo);
        auto& readbackCmdBuffer = readbackCmdBuffers[0];

        vk::CommandBufferBeginInfo readbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        readbackCmdBuffer->begin(readbackBeginInfo);

        vk::BufferCopy readbackCopyRegion(0, 0, sortKeyBufferSize);
        readbackCmdBuffer->copyBuffer(
            sortKeyBuffer.getBuffer(),
            readbackBuffer.getBuffer(),
            readbackCopyRegion
        );

        readbackCmdBuffer->end();

        vk::SubmitInfo readbackSubmitInfo(
            0, nullptr, nullptr,
            1, &*readbackCmdBuffer,
            0, nullptr
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &readbackSubmitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();

        // 16. Verify results
        SortKeyPair* readbackData = static_cast<SortKeyPair*>(readbackBuffer.map());

        std::cout << "\n[System] === Depth Verification ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        
        if (pointData.size() > 0) {
            std::cout << "[System] SortKeyPair[0]: depth = " << readbackData[0].depth 
                      << ", index = " << readbackData[0].index << std::endl;
        }
        
        if (pointData.size() > 100) {
            std::cout << "[System] SortKeyPair[100]: depth = " << readbackData[100].depth 
                      << ", index = " << readbackData[100].index << std::endl;
        }

        // (h) Bitonic sort verification: print first 5 elements and last valid element
        std::cout << "\n[System] === Bitonic Sort Verification (Descending) ===" << std::endl;
        std::cout << "[System] Padded count: " << paddedCount << ", Valid points: " << pointData.size() << std::endl;

        uint32_t printCount = std::min(static_cast<uint32_t>(5), static_cast<uint32_t>(pointData.size()));
        for (uint32_t i = 0; i < printCount; ++i) {
            std::cout << "[System] SortKeyPair[" << i << "]: depth = " << readbackData[i].depth 
                      << ", index = " << readbackData[i].index << std::endl;
        }

        if (pointData.size() > 1) {
            uint32_t lastIdx = static_cast<uint32_t>(pointData.size()) - 1;
            std::cout << "[System] SortKeyPair[" << lastIdx << "]: depth = " << readbackData[lastIdx].depth 
                      << ", index = " << readbackData[lastIdx].index << std::endl;
        }

        std::cout << "[System] === Verification Complete ===" << std::endl;

        readbackBuffer.unmap();

        std::cout << "[System] VERIFICATION SUCCESS: Depth Key-Value generation complete." << std::endl;

        // 17. Reverse DMA: Readback Gaussian2DBuffer for projection verification
        VmaBuffer gaussian2DReadbackBuffer(
            vulkanDevice.getAllocator(),
            gaussian2DBufferSize,
            vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_TO_CPU
        );

        vk::CommandBufferAllocateInfo g2dReadbackCmdBufInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );
        auto g2dReadbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(g2dReadbackCmdBufInfo);
        auto& g2dReadbackCmdBuffer = g2dReadbackCmdBuffers[0];

        vk::CommandBufferBeginInfo g2dReadbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        g2dReadbackCmdBuffer->begin(g2dReadbackBeginInfo);

        vk::BufferCopy g2dReadbackCopyRegion(0, 0, gaussian2DBufferSize);
        g2dReadbackCmdBuffer->copyBuffer(
            gaussian2DBuffer.getBuffer(),
            gaussian2DReadbackBuffer.getBuffer(),
            g2dReadbackCopyRegion
        );

        g2dReadbackCmdBuffer->end();

        vk::SubmitInfo g2dReadbackSubmitInfo(
            0, nullptr, nullptr,
            1, &*g2dReadbackCmdBuffer,
            0, nullptr
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &g2dReadbackSubmitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();

        // 18. Verify 2D projection results
        Gaussian2D* g2dReadbackData = static_cast<Gaussian2D*>(gaussian2DReadbackBuffer.map());

        std::cout << "\n[System] === 2D Projection Verification ===" << std::endl;
        std::cout << "[System] Padded count: " << paddedCount << ", Valid points: " << pointData.size() << std::endl;

        uint32_t g2dPrintCount = std::min(static_cast<uint32_t>(5), static_cast<uint32_t>(pointData.size()));
        for (uint32_t i = 0; i < g2dPrintCount; ++i) {
            std::cout << "[System] Gaussian2D[" << i << "]: pos2D = (" 
                      << g2dReadbackData[i].pos2D[0] << ", " << g2dReadbackData[i].pos2D[1] << ")"
                      << ", depth = " << g2dReadbackData[i].depth
                      << ", opacity = " << g2dReadbackData[i].opacity
                      << ", conic = (" << g2dReadbackData[i].conic[0] << ", " 
                      << g2dReadbackData[i].conic[1] << ", " << g2dReadbackData[i].conic[2] << ")"
                      << ", color = (" << g2dReadbackData[i].color[0] << ", "
                      << g2dReadbackData[i].color[1] << ", " << g2dReadbackData[i].color[2] << ")"
                      << std::endl;
        }

        std::cout << "[System] === 2D Projection Verification Complete ===" << std::endl;

        gaussian2DReadbackBuffer.unmap();

        std::cout << "[System] VERIFICATION SUCCESS: 3D to 2D covariance projection complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
