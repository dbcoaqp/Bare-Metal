#pragma once
#include <cstdint>

// 16-byte alignment for GPU memory layout compatibility (std430)
struct alignas(16) Gaussian2D {
    float pos2D[2];         // 8 bytes  - Screen pixel coordinates (x, y)
    float depth;            // 4 bytes  - Linear depth
    float opacity;          // 4 bytes  - Final opacity

    float conic[3];         // 12 bytes - Inverse 2D covariance parameters (a, b, c)
    float pad1;             // 4 bytes  - 16-byte alignment padding

    float color[3];         // 12 bytes - Final RGB color
    uint32_t bounds[4];     // 16 bytes - Tile bounding box (min_tx, min_ty, max_tx, max_ty)

    float pad2;             // 4 bytes  - 16-byte alignment padding (total = 64)
};
static_assert(sizeof(Gaussian2D) == 64, "Gaussian2D must be 64 bytes");
static_assert(alignof(Gaussian2D) == 16, "Gaussian2D must be 16-byte aligned");
