#pragma once

#include <cstdint>

struct alignas(16) CameraUniform {
    float viewProj[16];
    float cameraPosition[3];
    float pad;
};

static_assert(sizeof(CameraUniform) == 80, "CameraUniform must be exactly 80 bytes");
static_assert(alignof(CameraUniform) == 16, "CameraUniform must have 16-byte alignment");
