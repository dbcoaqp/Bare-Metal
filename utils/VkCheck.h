#pragma once

#include <stdexcept>
#include <string>

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
