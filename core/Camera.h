#pragma once

#include <cstdint>
#include <cstddef>

struct alignas(16) CameraUniform {
    float viewProj[16];        // 64 bytes
    float cameraPosition[4];   // 16 bytes (xyz + pad)
    float cameraForward[4];    // 16 bytes (xyz + pad)
};

static_assert(sizeof(CameraUniform) == 96, "CameraUniform must be exactly 96 bytes");
static_assert(alignof(CameraUniform) == 16, "CameraUniform must have 16-byte alignment");
