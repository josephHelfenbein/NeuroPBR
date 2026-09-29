#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Expands 8-bit material maps to float; albedo is decoded from sRGB
extern "C" __global__
void unpackMaterialKernel(const uchar4* __restrict__ albedoRGBA8, const uchar4* __restrict__ normalRGBA8,
                          const uint8_t* __restrict__ roughnessR8, const uint8_t* __restrict__ metallicR8,
                          float4* __restrict__ albedo, float4* __restrict__ normal,
                          float* __restrict__ roughness, float* __restrict__ metallic,
                          size_t pixelCount);

void launchUnpackMaterial(const uint8_t* albedoRGBA8, const uint8_t* normalRGBA8,
                          const uint8_t* roughnessR8, const uint8_t* metallicR8,
                          float4* albedo, float4* normal, float* roughness, float* metallic,
                          size_t pixelCount, cudaStream_t stream);
