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
    uint32_t key;
    uint32_t index;
};

static_assert(sizeof(SortKeyPair) == 8, "SortKeyPair must be exactly 8 bytes");
static_assert(alignof(SortKeyPair) == 8, "SortKeyPair must have 8-byte alignment");

// 64-bit key variant for tile-based radix sort (Phase 15)
// Packs (32-bit tileID | 32-bit depthKey) into the key, plus original gaussian index
struct alignas(16) TileSortKeyPair {
    uint64_t key;       // 8 bytes: (tileID << 32) | depthKey
    uint32_t index;     // 4 bytes: original gaussian index
    uint32_t pad;       // 4 bytes: alignment padding
};

static_assert(sizeof(TileSortKeyPair) == 16, "TileSortKeyPair must be exactly 16 bytes");
static_assert(alignof(TileSortKeyPair) == 16, "TileSortKeyPair must have 16-byte alignment");
