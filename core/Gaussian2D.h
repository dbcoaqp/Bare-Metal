#pragma once
#include <cstdint>

// 16-byte alignment for GPU memory layout compatibility (std430)
struct alignas(16) Gaussian2D {
    float pos2D[2];    // Screen pixel coordinates (x, y)
    float depth;       // Linear depth
    float opacity;     // Final opacity

    float conic[3];    // Inverse 2D covariance parameters (a, b, c)
    float pad1;        // 16-byte alignment padding

    float color[3];    // Final RGB color
    float pad2;        // 16-byte alignment padding
};
static_assert(sizeof(Gaussian2D) == 48, "Gaussian2D must be 48 bytes");
static_assert(alignof(Gaussian2D) == 16, "Gaussian2D must be 16-byte aligned");
