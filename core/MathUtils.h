#pragma once
#include <cmath>

class MathUtils {
public:
    
    static void buildViewProjection(float* viewProj,
                                     float cx, float cy, float cz,
                                     float fwdX, float fwdY, float fwdZ,
                                     float viewportWidth, float viewportHeight) {
        // Normalize forward direction
        float len = std::sqrt(fwdX * fwdX + fwdY * fwdY + fwdZ * fwdZ);
        float fx = fwdX / len;
        float fy = fwdY / len;
        float fz = fwdZ / len;

        // World up vector
        float ux = 0.0f, uy = 1.0f, uz = 0.0f;

        // Right = forward × up
        float rx = fy * uz - fz * uy;
        float ry = fz * ux - fx * uz;
        float rz = fx * uy - fy * ux;
        len = std::sqrt(rx * rx + ry * ry + rz * rz);
        rx /= len;
        ry /= len;
        rz /= len;

        // Recompute up = right × forward (ensure orthogonality)
        ux = ry * fz - rz * fy;
        uy = rz * fx - rx * fz;
        uz = rx * fy - ry * fx;

        // View-space translation: -dot(axis, position)
        float tx = -(rx * cx + ry * cy + rz * cz);
        float ty = -(ux * cx + uy * cy + uz * cz);
        float tz = -(fx * cx + fy * cy + fz * cz);

        // Vulkan perspective projection parameters
        float fov = 1.0f; // ~57 degrees
        float aspect = viewportWidth / viewportHeight;
        float zNear = 0.1f;
        float zFar = 100.0f;
        float f = 1.0f / std::tan(fov * 0.5f);

        float P00 = f / aspect;
        float P11 = -f;  // Vulkan Y-axis down
        float P22 = zFar / (zFar - zNear);
        float P32 = -(zFar * zNear) / (zFar - zNear);

        // ViewProj = Proj * View (column-major storage)
        //   V col0=[rx,ux,fx,0]  col1=[ry,uy,fy,0]  col2=[rz,uz,fz,0]  col3=[tx,ty,tz,1]
        //   P diag(P00,P11,P22,0) with P[11]=1, P[14]=P32
        viewProj[0]  = P00 * rx;
        viewProj[1]  = P11 * ux;
        viewProj[2]  = P22 * fx;
        viewProj[3]  = fx;

        viewProj[4]  = P00 * ry;
        viewProj[5]  = P11 * uy;
        viewProj[6]  = P22 * fy;
        viewProj[7]  = fy;

        viewProj[8]  = P00 * rz;
        viewProj[9]  = P11 * uz;
        viewProj[10] = P22 * fz;
        viewProj[11] = fz;

        viewProj[12] = P00 * tx;
        viewProj[13] = P11 * ty;
        viewProj[14] = P22 * tz + P32;
        viewProj[15] = tz;
    }
};
