#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "core/GaussianData.h"

class NpyParser {
public:
    static std::vector<GaussianPoint> parse(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Fatal: Failed to open .npy file: " + filepath);
        }

        // 1. Verify magic string \x93NUMPY
        uint8_t magic[6];
        file.read(reinterpret_cast<char*>(magic), 6);
        if (!file) {
            throw std::runtime_error("Fatal: Failed to read magic bytes from: " + filepath);
        }

        const uint8_t expectedMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        if (std::memcmp(magic, expectedMagic, 6) != 0) {
            throw std::runtime_error("Fatal: Invalid .npy magic string in: " + filepath);
        }

        // 2. Read major/minor version (1 byte each)
        uint8_t majorVersion, minorVersion;
        file.read(reinterpret_cast<char*>(&majorVersion), 1);
        file.read(reinterpret_cast<char*>(&minorVersion), 1);
        if (!file) {
            throw std::runtime_error("Fatal: Failed to read version from: " + filepath);
        }

        // 3. Read header length
        uint32_t headerLength = 0;
        if (majorVersion == 1) {
            // Version 1.x: 2-byte little-endian header length
            uint16_t headerLen16;
            file.read(reinterpret_cast<char*>(&headerLen16), 2);
            headerLength = headerLen16;
        } else if (majorVersion >= 2) {
            // Version 2.x+: 4-byte little-endian header length
            file.read(reinterpret_cast<char*>(&headerLength), 4);
        } else {
            throw std::runtime_error("Fatal: Unsupported .npy version: " + 
                std::to_string(majorVersion) + "." + std::to_string(minorVersion));
        }

        if (!file) {
            throw std::runtime_error("Fatal: Failed to read header length from: " + filepath);
        }

        // 4. Skip header string
        file.seekg(headerLength, std::ios::cur);
        if (!file) {
            throw std::runtime_error("Fatal: Failed to skip header in: " + filepath);
        }

        // 5. Calculate payload size
        auto payloadStart = file.tellg();
        file.seekg(0, std::ios::end);
        auto fileSize = file.tellg();
        size_t payloadSize = static_cast<size_t>(fileSize - payloadStart);

        // 6. Validate payload alignment
        constexpr size_t elementSize = sizeof(GaussianPoint);
        if (payloadSize % elementSize != 0) {
            throw std::runtime_error("Fatal: Payload size (" + std::to_string(payloadSize) + 
                " bytes) is not a multiple of GaussianPoint size (" + std::to_string(elementSize) + " bytes)");
        }

        size_t numPoints = payloadSize / elementSize;
        std::cout << "[System] NpyParser: Found " << numPoints << " Gaussian points in " << filepath << std::endl;

        // 7. Read payload directly into vector
        file.seekg(payloadStart);
        std::vector<GaussianPoint> points(numPoints);
        file.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(payloadSize));
        
        if (!file) {
            throw std::runtime_error("Fatal: Failed to read payload data from: " + filepath);
        }

        return points;
    }
};
