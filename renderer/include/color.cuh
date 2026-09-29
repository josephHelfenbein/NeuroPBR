#pragma once

#include <cuda_runtime.h>
#include <math.h>

constexpr float kColorPi = 3.14159265358979323846f;

__host__ __device__ inline float luminanceRec709(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

constexpr float kExposureMinRadiance = 1e-4f;

// Scales so an 18% grey surface facing up renders at 0.18
__host__ __device__ inline float exposureScale(float3 irradianceUp, float jitterEV) {
    const float whiteRadiance = luminanceRec709(irradianceUp) / kColorPi;
    return exp2f(jitterEV) / fmaxf(whiteRadiance, kExposureMinRadiance);
}

// Khronos PBR Neutral:
// https://github.com/KhronosGroup/ToneMapping/blob/f5dc101149fc5c85c0f9852fe2ba438853e8a7d1/PBR_Neutral/pbrNeutral.glsl
__host__ __device__ inline float3 tonemapPBRNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = fminf(color.x, fminf(color.y, color.z));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color.x -= offset;
    color.y -= offset;
    color.z -= offset;

    float peak = fmaxf(color.x, fmaxf(color.y, color.z));
    if (peak < startCompression) return color;

    const float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    const float scale = newPeak / peak;
    color.x *= scale;
    color.y *= scale;
    color.z *= scale;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    color.x = color.x * (1.0f - g) + newPeak * g;
    color.y = color.y * (1.0f - g) + newPeak * g;
    color.z = color.z * (1.0f - g) + newPeak * g;
    return color;
}

__host__ __device__ inline float srgbEncode(float c) {
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}
