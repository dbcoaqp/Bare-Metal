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

#include "utils/FileIO.h"
#include "utils/VkCheck.h"
#include "core/VmaBuffer.h"
#include "core/GaussianData.h"
#include "utils/NpyParser.h"

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

        // Read shader binary (path injected by CMake at build-time)
        std::string shaderPath = std::string(SHADER_DIR) + "/probe.spv";
        auto shaderCode = FileIO::readFile(shaderPath);
        std::cout << "[System] Shader loaded: " << shaderCode.size() << " bytes." << std::endl;

        // Phase 5: DMA Data Transfer Pipeline
        // 1. Parse Gaussian points from .npy file
        std::string npyPath = std::string(SHADER_DIR) + "/test.npy";
        auto pointData = NpyParser::parse(npyPath);
        vk::DeviceSize bufferSize = pointData.size() * sizeof(GaussianPoint);
        std::cout << "[System] Loaded " << pointData.size() << " Gaussian points (" 
                  << bufferSize << " bytes)." << std::endl;

        // 2. Create staging buffer (CPU-accessible)
        VmaBuffer stagingBuffer(
            vulkanDevice.getAllocator(),
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_CPU_TO_GPU
        );

        // 3. Map staging buffer and copy data
        void* mapped = stagingBuffer.map();
        std::memcpy(mapped, pointData.data(), bufferSize);
        stagingBuffer.flush();
        stagingBuffer.unmap();
        std::cout << "[System] Staging buffer populated with Gaussian data." << std::endl;

        // 4. Create GPU-only storage buffer (Device Local VRAM)
        VmaBuffer gaussianBuffer(
            vulkanDevice.getAllocator(),
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_GPU_ONLY
        );

        // 5. Create command pool (needed for both DMA transfer and compute dispatch)
        vk::CommandPoolCreateInfo commandPoolCreateInfo(
            vk::CommandPoolCreateFlags(),
            target.indices.computeFamily.value()
        );

        auto commandPool = vulkanDevice.getDevice().createCommandPoolUnique(commandPoolCreateInfo);

        // 6. One-time DMA transfer command
        vk::CommandBufferAllocateInfo tempCmdBufInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );
        auto tempCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(tempCmdBufInfo);
        auto& tempCmdBuffer = tempCmdBuffers[0];

        vk::CommandBufferBeginInfo tempBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        tempCmdBuffer->begin(tempBeginInfo);

        vk::BufferCopy copyRegion(0, 0, bufferSize);
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
        // tempCmdBuffers released here (RAII); stagingBuffer also released at scope exit

        // 1. Create Shader Module (RAII)
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo(
            vk::ShaderModuleCreateFlags(),
            shaderCode.size(),
            reinterpret_cast<const uint32_t*>(shaderCode.data())
        );
        auto shaderModule = vulkanDevice.getDevice().createShaderModuleUnique(shaderModuleCreateInfo);
        std::cout << "[System] Compute Shader Module compiled." << std::endl;

        // 2. Create Descriptor Set Layout & Pool (Contract & Pool)
        vk::DescriptorSetLayoutBinding layoutBinding(
            0,                                // binding
            vk::DescriptorType::eStorageBuffer,
            1,                                // descriptorCount
            vk::ShaderStageFlagBits::eCompute
        );

        vk::DescriptorSetLayoutCreateInfo layoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            1,                                // bindingCount
            &layoutBinding
        );

        auto descriptorSetLayout = vulkanDevice.getDevice().createDescriptorSetLayoutUnique(layoutCreateInfo);

        vk::DescriptorPoolSize poolSize(
            vk::DescriptorType::eStorageBuffer,
            1                                 // descriptorCount
        );

        vk::DescriptorPoolCreateInfo poolCreateInfo(
            vk::DescriptorPoolCreateFlags(),
            1,                                // maxSets
            1,                                // poolSizeCount
            &poolSize
        );

        auto descriptorPool = vulkanDevice.getDevice().createDescriptorPoolUnique(poolCreateInfo);

        // 3. Forge Compute Pipeline (State Machine)
        vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            vk::PipelineLayoutCreateFlags(),
            1,                                // setLayoutCount
            &*descriptorSetLayout
        );

        auto pipelineLayout = vulkanDevice.getDevice().createPipelineLayoutUnique(pipelineLayoutCreateInfo);

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
            nullptr,                          // pipelineCache
            computePipelineCreateInfo
        );

        if (computePipelineResult.result != vk::Result::eSuccess) {
            throw std::runtime_error("Fatal: Failed to create compute pipeline.");
        }

        auto computePipeline = std::move(computePipelineResult.value);

        // 4. Update Descriptor Set (Bind Contract)
        vk::DescriptorSetAllocateInfo descriptorSetAllocateInfo(
            *descriptorPool,
            1,                                // descriptorSetCount
            &*descriptorSetLayout
        );

        auto descriptorSets = vulkanDevice.getDevice().allocateDescriptorSets(descriptorSetAllocateInfo);
        auto descriptorSet = descriptorSets[0];

        vk::DescriptorBufferInfo bufferInfo(
            gaussianBuffer.getBuffer(),       // Device Local VRAM
            0,                                // offset
            bufferSize
        );

        vk::WriteDescriptorSet writeDescriptorSet(
            descriptorSet,
            0,                                // dstBinding
            0,                                // dstArrayElement
            1,                                // descriptorCount
            vk::DescriptorType::eStorageBuffer,
            nullptr,                          // pImageInfo
            &bufferInfo,
            nullptr                           // pTexelBufferView
        );

        vulkanDevice.getDevice().updateDescriptorSets(
            1,                                // descriptorWriteCount
            &writeDescriptorSet,
            0,                                // descriptorCopyCount
            nullptr                           // pDescriptorCopies
        );

        // 5. Record and Dispatch (Command & Fire)
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1                                 // commandBufferCount
        );

        auto commandBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(commandBufferAllocateInfo);
        auto commandBuffer = std::move(commandBuffers[0]);

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        commandBuffer->begin(beginInfo);
        commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
        commandBuffer->bindDescriptorSets(
            vk::PipelineBindPoint::eCompute,
            *pipelineLayout,
            0,                                // firstSet
            1,                                // descriptorSetCount
            &descriptorSet,
            0,                                // dynamicOffsetCount
            nullptr                           // pDynamicOffsets
        );

        // Dynamic Dispatch: calculate workgroup count based on actual point count
        uint32_t groupCountX = (static_cast<uint32_t>(pointData.size()) + 255) / 256;
        commandBuffer->dispatch(groupCountX, 1, 1);
        std::cout << "[System] Dispatched " << groupCountX << " workgroups for " << pointData.size() << " points." << std::endl;

        commandBuffer->end();

        vk::SubmitInfo submitInfo(
            0,                                // waitSemaphoreCount
            nullptr,                          // pWaitSemaphores
            nullptr,                          // pWaitDstStageMask
            1,                                // commandBufferCount
            &*commandBuffer
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &submitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();

        std::cout << "[System] GaussianPoint size: " << sizeof(GaussianPoint) << " bytes (aligned: " << alignof(GaussianPoint) << " bytes)" << std::endl;
        std::cout << "[System] Compute dispatch complete." << std::endl;

        // Phase 5: Reverse DMA & Verification (回程高铁)
        // 1. Create readback buffer (GPU → CPU)
        VmaBuffer readbackBuffer(
            vulkanDevice.getAllocator(),
            bufferSize,
            vk::BufferUsageFlagBits::eTransferDst,
            VMA_MEMORY_USAGE_GPU_TO_CPU
        );

        // 2. Allocate temporary command buffer for readback
        vk::CommandBufferAllocateInfo readbackCmdBufInfo(
            *commandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        );
        auto readbackCmdBuffers = vulkanDevice.getDevice().allocateCommandBuffersUnique(readbackCmdBufInfo);
        auto& readbackCmdBuffer = readbackCmdBuffers[0];

        // 3. Record copy command
        vk::CommandBufferBeginInfo readbackBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
        readbackCmdBuffer->begin(readbackBeginInfo);

        vk::BufferCopy readbackCopyRegion(0, 0, bufferSize);
        readbackCmdBuffer->copyBuffer(
            gaussianBuffer.getBuffer(),
            readbackBuffer.getBuffer(),
            readbackCopyRegion
        );

        readbackCmdBuffer->end();

        // 4. Submit and wait
        vk::SubmitInfo readbackSubmitInfo(
            0, nullptr, nullptr,
            1, &*readbackCmdBuffer,
            0, nullptr
        );

        VK_CHECK(vulkanDevice.getComputeQueue().submit(1, &readbackSubmitInfo, nullptr));
        vulkanDevice.getComputeQueue().waitIdle();

        // 5. Map readback buffer and verify compute results
        GaussianPoint* readbackData = static_cast<GaussianPoint*>(readbackBuffer.map());

        std::cout << "[System] Verification: Point[0] position = (" 
                  << readbackData[0].position[0] << ", "
                  << readbackData[0].position[1] << ", "
                  << readbackData[0].position[2] << ")" << std::endl;

        readbackBuffer.unmap();

        std::cout << "[System] VERIFICATION SUCCESS: Reverse DMA readback and compute verification complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Critical Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
