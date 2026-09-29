#include <unpack.cuh>

#include <device_launch_parameters.h>

#include <srgb_lut.cuh>

// 1/255, written out so --use_fast_math can't change it
static constexpr float kUnpackByteToFloat = 0x1.010102p-8f;

extern "C" __global__
void unpackMaterialKernel(const uchar4* __restrict__ albedoRGBA8, const uchar4* __restrict__ normalRGBA8,
                          const uint8_t* __restrict__ roughnessR8, const uint8_t* __restrict__ metallicR8,
                          float4* __restrict__ albedo, float4* __restrict__ normal,
                          float* __restrict__ roughness, float* __restrict__ metallic,
                          size_t pixelCount) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pixelCount) {
        return;
    }

    const uchar4 a = albedoRGBA8[i];
    const uchar4 n = normalRGBA8[i];
    albedo[i] = make_float4(srgbByteToLinear(a.x), srgbByteToLinear(a.y), srgbByteToLinear(a.z), 0.0f);
    normal[i] = make_float4(n.x * kUnpackByteToFloat, n.y * kUnpackByteToFloat, n.z * kUnpackByteToFloat, 0.0f);
    roughness[i] = roughnessR8[i] * kUnpackByteToFloat;
    metallic[i] = metallicR8[i] * kUnpackByteToFloat;
}

void launchUnpackMaterial(const uint8_t* albedoRGBA8, const uint8_t* normalRGBA8,
                          const uint8_t* roughnessR8, const uint8_t* metallicR8,
                          float4* albedo, float4* normal, float* roughness, float* metallic,
                          size_t pixelCount, cudaStream_t stream) {
    if (pixelCount == 0) {
        return;
    }
    constexpr unsigned int kBlockSize = 256;
    const unsigned int gridSize = static_cast<unsigned int>((pixelCount + kBlockSize - 1) / kBlockSize);
    unpackMaterialKernel<<<gridSize, kBlockSize, 0, stream>>>(reinterpret_cast<const uchar4*>(albedoRGBA8),
                                                             reinterpret_cast<const uchar4*>(normalRGBA8),
                                                             roughnessR8, metallicR8,
                                                             albedo, normal, roughness, metallic,
                                                             pixelCount);
}
