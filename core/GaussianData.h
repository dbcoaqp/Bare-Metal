#pragma once

#include <cstddef>

// std430 aligned data structure for GPU compute shaders
// Total size: 48 bytes (12 floats x 4 bytes)
// Alignment: 16 bytes as required by std430

#ifdef _MSC_VER
    __declspec(align(16)) struct GaussianPoint {
#else
    struct __attribute__((aligned(16))) GaussianPoint {
#endif
    float position[3];  // xyz coordinates
    float scale;        // scale factor
    float color[3];     // base color or SH order 0
    float opacity;      // opacity value
    float rot[4];       // quaternion rotation
};

static_assert(sizeof(GaussianPoint) == 48, "GaussianPoint must be exactly 48 bytes for std430 compliance");
static_assert(alignof(GaussianPoint) == 16, "GaussianPoint must have 16-byte alignment for std430");
