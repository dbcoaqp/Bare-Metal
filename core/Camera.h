#pragma once

#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_LEFT_HANDED
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cstdint>
#include <cstddef>

class Camera {
public:
    glm::vec3 position{0.0f, 0.0f, 5.0f};
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};

    void processMouse(float deltaX, float deltaY) {
        constexpr float sensitivity = 0.0025f;
        const glm::quat yawQuat = glm::angleAxis(-deltaX * sensitivity, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::quat pitchQuat = glm::angleAxis(deltaY * sensitivity, glm::vec3(1.0f, 0.0f, 0.0f));
        orientation = yawQuat * orientation * pitchQuat;
        orientation = glm::normalize(orientation);
    }

    void processKeyboard(glm::vec3 inputDir, float deltaTime) {
        const glm::vec3 forward = orientation * glm::vec3(0.0f, 0.0f, -1.0f);
        const glm::vec3 right = orientation * glm::vec3(1.0f, 0.0f, 0.0f);
        position += (right * inputDir.x + forward * inputDir.z) * (movementSpeed * deltaTime);
        position.y += inputDir.y * movementSpeed * deltaTime;
    }

    glm::mat4 getViewMatrix() const {
        return  glm::mat4_cast(glm::conjugate(orientation)) *
                glm::translate(glm::mat4(1.0f), -position);
    }

    glm::vec3 getForward() const { return orientation * glm::vec3(0.0f, 0.0f, -1.0f); }

    static constexpr float movementSpeed = 5.0f;
};

// std140 CameraUBO layout shared with the compute shaders.
struct alignas(16) CameraUniform {
    float viewProj[16];
    float cameraPosition[4];
    float cameraForward[4];
    float focal_x;
    float focal_y;
    float tan_fovx;
    float tan_fovy;
    float viewport[2];
    float pad_cam[2];
};

static_assert(sizeof(CameraUniform) == 128, "CameraUniform must be exactly 128 bytes");
static_assert(alignof(CameraUniform) == 16, "CameraUniform must have 16-byte alignment");
