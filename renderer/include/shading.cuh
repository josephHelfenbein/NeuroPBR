#pragma once

#include <cuda_runtime.h>

extern "C" __global__
void shadeKernel(cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                 int specularMipLevels, cudaTextureObject_t irradianceTex,
                 cudaTextureObject_t brdfLutTex, const float* albedo,
                 const float* normal, const float* roughness,
                 const float* metallic, float* outRGBA,
                 int width, int height, float3 cameraPos,
                 float3 cameraForward, float3 cameraRight,
                 float3 cameraUp, float tanHalfFovY, float aspect,
                 bool enableShadows, bool enableCameraSmudge,
                 bool enableLensFlare, const float* cameraSmudgeImage,
                 int cameraSmudgeWidth, int cameraSmudgeHeight,
                 int cameraSmudgeChannels, const float* lensFlareImage,
                 int lensFlareWidth, int lensFlareHeight,
                 int lensFlareChannels);

void launchShadeKernel(dim3 gridDim, dim3 blockDim,
                       cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                       int specularMipLevels, cudaTextureObject_t irradianceTex,
                       cudaTextureObject_t brdfLutTex, const float* albedo,
                       const float* normal, const float* roughness,
                       const float* metallic, float* outRGBA,
                       int width, int height, float3 cameraPos,
                       float3 cameraForward, float3 cameraRight,
                       float3 cameraUp, float tanHalfFovY, float aspect,
                       bool enableShadows, bool enableCameraSmudge,
                       bool enableLensFlare, const float* cameraSmudgeImage,
                       int cameraSmudgeWidth, int cameraSmudgeHeight,
                       int cameraSmudgeChannels, const float* lensFlareImage,
                       int lensFlareWidth, int lensFlareHeight,
                       int lensFlareChannels);