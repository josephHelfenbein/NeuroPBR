#pragma once

#include <cuda_runtime.h>
#include <cstdint>

extern "C" __global__
void shadeKernel(cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
             int specularMipLevels, cudaTextureObject_t irradianceTex,
             cudaTextureObject_t brdfLutTex, const float4* __restrict__ albedo,
             const float4* __restrict__ normal, const float* __restrict__ roughness,
             const float* __restrict__ metallic, uint8_t* __restrict__ outRGB,
                 int width, int height, float3 cameraPos,
                 float3 cameraForward, float3 cameraRight,
                 float3 cameraUp, float tanHalfFovY, float aspect,
                 bool enableShadows, bool enableCameraArtifacts,
                 unsigned long long artifactSeed,
                 float horizonBrightness,
                 float zenithBrightness, float hardness,
                 float exposureJitterEV);

void launchShadeKernel(dim3 gridDim, dim3 blockDim,
                       cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                       int specularMipLevels, cudaTextureObject_t irradianceTex,
                       cudaTextureObject_t brdfLutTex, const float4* __restrict__ albedo,
                       const float4* __restrict__ normal, const float* __restrict__ roughness,
                       const float* __restrict__ metallic, uint8_t* __restrict__ outRGB,
                       int width, int height, float3 cameraPos,
                       float3 cameraForward, float3 cameraRight,
                       float3 cameraUp, float tanHalfFovY, float aspect,
                       bool enableShadows, bool enableCameraArtifacts,
                       unsigned long long artifactSeed,
                       float horizonBrightness,
                       float zenithBrightness, float hardness,
                       float exposureJitterEV, cudaStream_t stream);