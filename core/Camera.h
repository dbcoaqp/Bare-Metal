#pragma once

#include <cstdint>
#include <cstddef>

// 16-byte alignment required for Vulkan std140 uniform buffer layout compatibility
struct alignas(16) CameraUniform {
    float viewProj[16];        // 64 bytes - mat4 in std140
    float cameraPosition[4];   // 16 bytes - vec4 in std140
    float cameraForward[4];    // 16 bytes - vec4 in std140

    float focal_x;             // x-axis focal length
    float focal_y;             // y-axis focal length
    float tan_fovx;            // FOV x tangent
    float tan_fovy;            // FOV y tangent
    float viewport[2];         // screen width and height
    float pad_cam[2];          // padding to 16-byte alignment
};

static_assert(sizeof(CameraUniform) == 128, "CameraUniform must be exactly 128 bytes");
static_assert(alignof(CameraUniform) == 16, "CameraUniform must have 16-byte alignment");
