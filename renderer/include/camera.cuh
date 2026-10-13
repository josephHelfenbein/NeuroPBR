#pragma once

#include <cuda_runtime.h>

// World size of one texture tile, centered on the origin
constexpr float kTileWorldSize = 0.5f;

struct PinholeCamera {
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float tanHalfFovY;
    float aspect;
    int width;
    int height;
};

// Pixel position of p in the photo, matching the rays shadeKernel casts. False if p is behind the camera
__host__ __device__ inline bool projectToPixel(const PinholeCamera& cam, float3 p, float& px, float& py) {
    const float3 d = make_float3(p.x - cam.position.x, p.y - cam.position.y, p.z - cam.position.z);
    const float depth = d.x * cam.forward.x + d.y * cam.forward.y + d.z * cam.forward.z;
    if (depth <= 1e-6f) {
        return false;
    }
    const float sensorX = (d.x * cam.right.x + d.y * cam.right.y + d.z * cam.right.z) / depth;
    const float sensorY = (d.x * cam.up.x + d.y * cam.up.y + d.z * cam.up.z) / depth;
    const float ndcX = sensorX / (cam.aspect * cam.tanHalfFovY);
    const float ndcY = -sensorY / cam.tanHalfFovY;
    px = (ndcX + 1.0f) * 0.5f * static_cast<float>(cam.width) - 0.5f;
    py = (ndcY + 1.0f) * 0.5f * static_cast<float>(cam.height) - 0.5f;
    return true;
}

// Plane point under texel (x, row), with rows in the flipped order shadeKernel samples
__host__ __device__ inline float3 texelToPlane(int x, int row, int texWidth, int texHeight) {
    // Pulled in slightly so the last texel doesn't wrap onto the next tile
    const float inset = 1.0f - 1e-5f;
    const float tiledU = static_cast<float>(x) / static_cast<float>(texWidth > 1 ? texWidth - 1 : 1) * inset;
    const float tiledV = static_cast<float>(row) / static_cast<float>(texHeight > 1 ? texHeight - 1 : 1) * inset;
    const float u = (tiledU - 0.5f) * kTileWorldSize;
    const float v = (tiledV - 0.5f) * kTileWorldSize;
    return make_float3(u, 0.0f, -v);
}
