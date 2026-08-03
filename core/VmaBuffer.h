#pragma once

#include <vulkan/vulkan.hpp>
#include "VmaUsage.h"
#include "utils/VkCheck.h"

class VmaBuffer {
public:
    VmaBuffer(VmaAllocator allocator, vk::DeviceSize size,
              vk::BufferUsageFlags usage, VmaMemoryUsage vmaUsage)
        : m_allocator(allocator), m_buffer(nullptr), m_allocation(nullptr), m_isMapped(false)
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
        // Ensure proper unmapping before destroying the buffer
        unmap();
        if (m_buffer && m_allocation != nullptr) {
            vmaDestroyBuffer(m_allocator, static_cast<VkBuffer>(m_buffer), m_allocation);
            m_buffer = nullptr;
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
        m_isMapped = true;
        return data;
    }

    void unmap() {
        if (m_isMapped) {
            vmaUnmapMemory(m_allocator, m_allocation);
            m_isMapped = false;
        }
    }

    void flush(vk::DeviceSize size = VK_WHOLE_SIZE) {
        VK_CHECK(vmaFlushAllocation(m_allocator, m_allocation, 0, static_cast<VkDeviceSize>(size)));
    }

private:
    VmaAllocator m_allocator;
    vk::Buffer m_buffer;
    VmaAllocation m_allocation;
    bool m_isMapped;
};
