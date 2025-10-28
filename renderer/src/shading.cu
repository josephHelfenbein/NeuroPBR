#include <shading.cuh>

#include <device_launch_parameters.h>
#include <math.h>

#include <utils.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" __global__
void shadeKernel(cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                 int specularMipLevels, cudaTextureObject_t irradianceTex,
                 cudaTextureObject_t brdfLutTex, const float* albedo,
                 const float* normal, const float* roughness,
                 const float* metallic, float* outRGBA,
                 int width, int height, float camX, float camY, float camZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;

    float3 albedoColor = make_float3(albedo[3 * idx + 0], albedo[3 * idx + 1], albedo[3 * idx + 2]);
    float3 N = make_float3(normal[3 * idx + 0], normal[3 * idx + 1], normal[3 * idx + 2]);
    N = makeNormalized(N);

    float rough = roughness[idx];
    float metal = metallic[idx];

    float px = (static_cast<float>(x) + 0.5f) / static_cast<float>(width) - 0.5f;
    float py = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) - 0.5f;
    float pz = 0.0f;

    float vx = camX - px;
    float vy = camY - py;
    float vz = camZ - pz;
    float vlen = sqrtf(vx * vx + vy * vy + vz * vz) + 1e-8f;
    vx /= vlen;
    vy /= vlen;
    vz /= vlen;

    float3 V = make_float3(vx, vy, vz);

    float NdotV = fmaxf(dot3(N, V), 0.0f);

    float3 R = make_float3(2.0f * NdotV * N.x - V.x,
                           2.0f * NdotV * N.y - V.y,
                           2.0f * NdotV * N.z - V.z);
    R = makeNormalized(R);

    float invMetal = 1.0f - metal;
    float3 baseReflectance = make_float3(0.04f, 0.04f, 0.04f);
    float3 F0 = make_float3(baseReflectance.x * invMetal + albedoColor.x * metal,
                            baseReflectance.y * invMetal + albedoColor.y * metal,
                            baseReflectance.z * invMetal + albedoColor.z * metal);

    float fresnelTerm = powf(fmaxf(1.0f - NdotV, 0.0f), 5.0f);
    float3 F = make_float3(F0.x + (1.0f - F0.x) * fresnelTerm,
                           F0.y + (1.0f - F0.y) * fresnelTerm,
                           F0.z + (1.0f - F0.z) * fresnelTerm);
    float3 kS = F;
    float3 kD = make_float3((1.0f - kS.x) * invMetal,
                            (1.0f - kS.y) * invMetal,
                            (1.0f - kS.z) * invMetal);

    float4 irrSample = texCubemap<float4>(irradianceTex, N.x, N.y, N.z);
    float3 irradiance = make_float3(irrSample.x, irrSample.y, irrSample.z);

    float3 diffuse = make_float3(kD.x * irradiance.x * albedoColor.x / M_PI,
                                 kD.y * irradiance.y * albedoColor.y / M_PI,
                                 kD.z * irradiance.z * albedoColor.z / M_PI);

    float maxMip = fmaxf(static_cast<float>(specularMipLevels - 1), 0.0f);
    float lod = fmaxf(rough, 0.0f) * maxMip;

    bool hasSpecular = (specularTex != 0);
    bool hasMips = hasSpecular && (specularMipLevels > 1);

    float4 specSample;
    if (hasMips) {
        specSample = texCubemapLod<float4>(specularTex, R.x, R.y, R.z, lod);
    } else if (hasSpecular) {
        specSample = texCubemap<float4>(specularTex, R.x, R.y, R.z);
    } else {
        specSample = texCubemap<float4>(envTex, R.x, R.y, R.z);
    }
    float3 prefilteredColor = make_float3(specSample.x, specSample.y, specSample.z);

    if (!hasSpecular) {
        float4 envSample = texCubemap<float4>(envTex, R.x, R.y, R.z);
        prefilteredColor = make_float3(envSample.x, envSample.y, envSample.z);
    }

    float lutU = fminf(fmaxf(NdotV, 0.0f), 1.0f);
    float lutV = fminf(fmaxf(rough, 0.0f), 1.0f);
    float2 brdfSample = tex2D<float2>(brdfLutTex, lutU, lutV);

    float3 specular = make_float3(prefilteredColor.x * (F.x * brdfSample.x + brdfSample.y),
                                  prefilteredColor.y * (F.y * brdfSample.x + brdfSample.y),
                                  prefilteredColor.z * (F.z * brdfSample.x + brdfSample.y));

    float3 color = make_float3(diffuse.x + specular.x,
                               diffuse.y + specular.y,
                               diffuse.z + specular.z);
    color.x = fmaxf(color.x, 0.0f);
    color.y = fmaxf(color.y, 0.0f);
    color.z = fmaxf(color.z, 0.0f);

    outRGBA[4*idx + 0] = color.x;
    outRGBA[4*idx + 1] = color.y;
    outRGBA[4*idx + 2] = color.z;
    outRGBA[4*idx + 3] = 1.0f;
}

void launchShadeKernel(dim3 gridDim, dim3 blockDim,
                       cudaTextureObject_t envTex, cudaTextureObject_t specularTex,
                       int specularMipLevels, cudaTextureObject_t irradianceTex,
                       cudaTextureObject_t brdfLutTex, const float* albedo,
                       const float* normal, const float* roughness,
                       const float* metallic, float* outRGBA,
                       int width, int height, float camX, float camY, float camZ) {
    shadeKernel<<<gridDim, blockDim>>>(envTex, specularTex, specularMipLevels,
                                       irradianceTex, brdfLutTex, albedo, normal,
                                       roughness, metallic, outRGBA,
                                       width, height, camX, camY, camZ);
}