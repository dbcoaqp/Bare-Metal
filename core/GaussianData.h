#pragma once

#include <cstdint>

/*
struct alignas(16) GaussianPoint {
    float position[3];
    float scale;
    float color[3];
    float opacity;
    float rot[4];
};
*/

struct alignas(16) GaussianPoint {
    float px, py, pz;     // 12 bytes
    float opacity;        // 4 bytes (Packed to 16 bytes)
    float sx, sy, sz;     // 12 bytes
    float pad1;           // 4 bytes (Packed to 16 bytes)
    float r, g, b;        // 12 bytes
    float pad2;           // 4 bytes (Packed to 16 bytes)
    float rw, rx, ry, rz; // 16 bytes
}; // Total: 64 bytes

static_assert(sizeof(GaussianPoint) == 64, "GaussianPoint must be exactly 64 bytes");
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
