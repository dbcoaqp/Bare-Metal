#pragma once

#include <cstdint>

struct alignas(16) GaussianPoint {
    float position[3];
    float scale;
    float color[3];
    float opacity;
    float rot[4];
};

static_assert(sizeof(GaussianPoint) == 48, "GaussianPoint must be exactly 48 bytes");
static_assert(alignof(GaussianPoint) == 16, "GaussianPoint must have 16-byte alignment");

struct alignas(8) SortKeyPair {
    float depth;
    uint32_t index;
};

static_assert(sizeof(SortKeyPair) == 8, "SortKeyPair must be exactly 8 bytes");
static_assert(alignof(SortKeyPair) == 8, "SortKeyPair must have 8-byte alignment");
