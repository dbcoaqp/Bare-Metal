#pragma once

#include <vulkan/vulkan.hpp>
#include <stdexcept>
#include <string>
#include <source_location>

/*
// VK_CHECK macro: Every Vulkan API call must be wrapped in this per CONVENTIONS.md
#define VK_CHECK(result)                                                    \
    do {                                                                    \
        VkResult err = result;                                              \
        if (err != VK_SUCCESS) {                                            \
            throw std::runtime_error("Fatal: Vulkan error " +               \
                std::to_string(err) + " at " + __FILE__ + ":" +            \
                std::to_string(__LINE__));                                  \
        }                                                                   \
    } while (0)
*/

inline void VK_CHECK(vk::Result result, const std::source_location location = std::source_location::current()){
    if (result != vk::Result::eSuccess){
        throw std::runtime_error("Fatal: Vulkan error " +               \
            vk::to_string(result) + " at " + __FILE__ + ":" +            \
            std::to_string(__LINE__)); 
    }
}

inline void VK_CHECK(VkResult result, const std::source_location location = std::source_location::current()){
    if (result != VK_SUCCESS){
        throw std::runtime_error("Fatal: Vulkan error " +               \
            std::to_string(result) + " at " + __FILE__ + ":" +            \
            std::to_string(__LINE__)); 
    }
}