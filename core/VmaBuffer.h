#pragma once

#include <vulkan/vulkan.hpp>
#include "VmaUsage.h"
#include "utils/VkCheck.h"

class VmaBuffer {
public:
    VmaBuffer(VmaAllocator allocator, vk::DeviceSize size, 
              vk::BufferUsageFlags usage, VmaMemoryUsage vmaUsage)
        : m_allocator(allocator), m_buffer(VK_NULL_HANDLE), m_allocation(nullptr) 
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = static_cast<VkDeviceSize>(size);
        bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = vmaUsage;

        VkBuffer rawBuffer = VK_NULL_HANDLE;
        VK_CHECK(vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &rawBuffer, &m_allocation, nullptr));
        m_buffer = rawBuffer;
    }

    ~VmaBuffer() {
        if (m_buffer != VK_NULL_HANDLE && m_allocation != nullptr) {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
            m_buffer = VK_NULL_HANDLE;
            m_allocation = nullptr;
        }
    }

    // Non-copyable
    VmaBuffer(const VmaBuffer&) = delete;
    VmaBuffer& operator=(const VmaBuffer&) = delete;

    vk::Buffer getBuffer() const { 
        return m_buffer; 
    }

    void* map() {
        void* data = nullptr;
        VK_CHECK(vmaMapMemory(m_allocator, m_allocation, &data));
        return data;
    }

    void unmap() {
        vmaUnmapMemory(m_allocator, m_allocation);
    }

    void flush(vk::DeviceSize size = VK_WHOLE_SIZE) {
        VK_CHECK(vmaFlushAllocation(m_allocator, m_allocation, 0, static_cast<VkDeviceSize>(size)));
    }

private:
    VmaAllocator m_allocator;
    VkBuffer m_buffer;
    VmaAllocation m_allocation;
};
