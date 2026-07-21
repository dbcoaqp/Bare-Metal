#pragma once
#include <cmath>

class MathUtils {
public:
    
    static void buildViewProjection(float* viewProj, float cx, float cy, float cz, float viewportWidth, float viewportHeight) {
        float fov = 1.0f; // Approximately 57 degrees field of view
        float aspect = viewportWidth / viewportHeight;
        float zNear = 0.1f;
        float zFar = 100.0f;
        float f = 1.0f / std::tan(fov * 0.5f);

        // 1. Vulkan-specific perspective projection matrix (Y-axis downwards, Z-axis 0~1)
        float P00 = f / aspect;
        float P11 = -f; 
        float P22 = zFar / (zFar - zNear);
        float P32 = -(zFar * zNear) / (zFar - zNear);

        // 2. Merging of View matrix and projection matrix (ViewProj = Proj * View)
        // Assume the camera is located at (cx, cy, cz) and is looking along the +Z axis. Column-major memory alignment is used.
        viewProj[0] = P00;  viewProj[4] = 0.0f; viewProj[8] = 0.0f;  viewProj[12] = P00 * (-cx);
        viewProj[1] = 0.0f; viewProj[5] = P11;  viewProj[9] = 0.0f;  viewProj[13] = P11 * (-cy);
        viewProj[2] = 0.0f; viewProj[6] = 0.0f; viewProj[10]= P22;   viewProj[14] = P22 * (-cz) + P32;
        viewProj[3] = 0.0f; viewProj[7] = 0.0f; viewProj[11]= 1.0f;  viewProj[15] = -cz;
    }
};