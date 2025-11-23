#pragma once

#include <cuda_runtime.h>
#include <cstddef>

static constexpr int SHADE_TILE_MAX_DIM = 32;
static constexpr int SHADE_TILE_CAPACITY = SHADE_TILE_MAX_DIM * SHADE_TILE_MAX_DIM;
static constexpr size_t SHADE_KERNEL_SHARED_MEM_BYTES =
    static_cast<size_t>(SHADE_TILE_CAPACITY) *
    (2u * sizeof(float4) + 2u * sizeof(float));

extern "C" __global__
void shadeKernel(cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
             int specularMipLevels, cudaTextureObject_t irradianceTex,
             cudaTextureObject_t brdfLutTex, const float4* __restrict__ albedo,
             const float4* __restrict__ normal, const float* __restrict__ roughness,
             const float* __restrict__ metallic, float* __restrict__ outRGBA,
                 int width, int height, float3 cameraPos,
                 float3 cameraForward, float3 cameraRight,
                 float3 cameraUp, float tanHalfFovY, float aspect,
                 bool enableShadows, bool enableCameraArtifacts,
                 unsigned long long artifactSeed,
                 float horizonBrightness,
                 float zenithBrightness, float hardness);

void launchShadeKernel(dim3 gridDim, dim3 blockDim,
                       cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                       int specularMipLevels, cudaTextureObject_t irradianceTex,
                       cudaTextureObject_t brdfLutTex, const float4* __restrict__ albedo,
                       const float4* __restrict__ normal, const float* __restrict__ roughness,
                       const float* __restrict__ metallic, float* __restrict__ outRGBA,
                       int width, int height, float3 cameraPos,
                       float3 cameraForward, float3 cameraRight,
                       float3 cameraUp, float tanHalfFovY, float aspect,
                       bool enableShadows, bool enableCameraArtifacts,
                       unsigned long long artifactSeed,
                       float horizonBrightness,
                       float zenithBrightness, float hardness);