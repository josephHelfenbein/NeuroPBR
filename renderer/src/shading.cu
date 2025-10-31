#include <shading.cuh>

#include <device_launch_parameters.h>
#include <math.h>

#include <utils.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ inline float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ inline float4 fetchOverlayPixel(const float* data, int channels, int pixelIndex) {
    if (!data || channels <= 0) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    int base = pixelIndex * channels;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 1.0f;

    switch (channels) {
        case 1: {
            r = g = b = data[base];
            break;
        }
        case 2: {
            r = g = b = data[base];
            a = clamp01(data[base + 1]);
            break;
        }
        case 3: {
            r = data[base + 0];
            g = data[base + 1];
            b = data[base + 2];
            break;
        }
        default: {
            r = data[base + 0];
            g = data[base + 1];
            b = data[base + 2];
            a = clamp01(data[base + 3]);
            break;
        }
    }

    return make_float4(r, g, b, a);
}

__device__ float sampleScalar(const float* data, int width, int height, float u, float v) {
    u = clamp01(u);
    v = clamp01(v);

    float x = u * static_cast<float>(width - 1);
    float y = v * static_cast<float>(height - 1);

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    float c00 = data[y0 * width + x0];
    float c10 = data[y0 * width + x1];
    float c01 = data[y1 * width + x0];
    float c11 = data[y1 * width + x1];

    float c0 = c00 * (1.0f - tx) + c10 * tx;
    float c1 = c01 * (1.0f - tx) + c11 * tx;
    return c0 * (1.0f - ty) + c1 * ty;
}

__device__ float3 sampleRGB(const float* data, int width, int height, float u, float v) {
    u = clamp01(u);
    v = clamp01(v);

    float x = u * static_cast<float>(width - 1);
    float y = v * static_cast<float>(height - 1);

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    float tx = x - static_cast<float>(x0);
    float ty = y - static_cast<float>(y0);

    int idx00 = (y0 * width + x0) * 3;
    int idx10 = (y0 * width + x1) * 3;
    int idx01 = (y1 * width + x0) * 3;
    int idx11 = (y1 * width + x1) * 3;

    float3 c00 = make_float3(data[idx00 + 0], data[idx00 + 1], data[idx00 + 2]);
    float3 c10 = make_float3(data[idx10 + 0], data[idx10 + 1], data[idx10 + 2]);
    float3 c01 = make_float3(data[idx01 + 0], data[idx01 + 1], data[idx01 + 2]);
    float3 c11 = make_float3(data[idx11 + 0], data[idx11 + 1], data[idx11 + 2]);

    float3 c0 = make_float3(c00.x * (1.0f - tx) + c10.x * tx,
                            c00.y * (1.0f - tx) + c10.y * tx,
                            c00.z * (1.0f - tx) + c10.z * tx);
    float3 c1 = make_float3(c01.x * (1.0f - tx) + c11.x * tx,
                            c01.y * (1.0f - tx) + c11.y * tx,
                            c01.z * (1.0f - tx) + c11.z * tx);

    return make_float3(c0.x * (1.0f - ty) + c1.x * ty,
                       c0.y * (1.0f - ty) + c1.y * ty,
                       c0.z * (1.0f - ty) + c1.z * ty);
}

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
                 const float* cameraSmudgeImage,
                 int cameraSmudgeWidth, int cameraSmudgeHeight,
                 int cameraSmudgeChannels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;

    float ndcX = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width)) * 2.0f - 1.0f;
    float ndcY = ((static_cast<float>(y) + 0.5f) / static_cast<float>(height)) * 2.0f - 1.0f;

    float sensorX = ndcX * aspect * tanHalfFovY;
    float sensorY = -ndcY * tanHalfFovY;

    float3 rayDir = make_float3(cameraForward.x + sensorX * cameraRight.x + sensorY * cameraUp.x,
                                cameraForward.y + sensorX * cameraRight.y + sensorY * cameraUp.y,
                                cameraForward.z + sensorX * cameraRight.z + sensorY * cameraUp.z);
    rayDir = makeNormalized(rayDir);

    const float3 planeNormal = make_float3(0.0f, 1.0f, 0.0f);
    float denom = dot3(planeNormal, rayDir);
    if (fabsf(denom) < 1e-6f) {
        outRGBA[4 * idx + 0] = 0.0f;
        outRGBA[4 * idx + 1] = 0.0f;
        outRGBA[4 * idx + 2] = 0.0f;
        outRGBA[4 * idx + 3] = 1.0f;
        return;
    }

    float t = -dot3(planeNormal, cameraPos) / denom;
    if (t <= 0.0f) {
        outRGBA[4 * idx + 0] = 0.0f;
        outRGBA[4 * idx + 1] = 0.0f;
        outRGBA[4 * idx + 2] = 0.0f;
        outRGBA[4 * idx + 3] = 1.0f;
        return;
    }

    float3 hit = make_float3(cameraPos.x + rayDir.x * t,
                             cameraPos.y + rayDir.y * t,
                             cameraPos.z + rayDir.z * t);

    if (fabsf(hit.x) > 0.5f || fabsf(hit.z) > 0.5f) {
        outRGBA[4 * idx + 0] = 0.0f;
        outRGBA[4 * idx + 1] = 0.0f;
        outRGBA[4 * idx + 2] = 0.0f;
        outRGBA[4 * idx + 3] = 1.0f;
        return;
    }

    float u = hit.x + 0.5f;
    float v = -hit.z + 0.5f;
    float tiledU = u * 2.0f;
    float tiledV = v * 2.0f;
    tiledU = tiledU - floorf(tiledU);
    tiledV = tiledV - floorf(tiledV);
    tiledU = clamp01(tiledU);
    tiledV = clamp01(tiledV);

    float3 albedoColor = sampleRGB(albedo, width, height, tiledU, tiledV);
    float3 normalSample = sampleRGB(normal, width, height, tiledU, tiledV);
    float rough = sampleScalar(roughness, width, height, tiledU, tiledV);
    float metal = sampleScalar(metallic, width, height, tiledU, tiledV);

    float3 normalTS = make_float3(normalSample.x * 2.0f - 1.0f,
                                  normalSample.y * 2.0f - 1.0f,
                                  normalSample.z * 2.0f - 1.0f);

    const float3 tangent = make_float3(1.0f, 0.0f, 0.0f);
    const float3 bitangent = make_float3(0.0f, 0.0f, 1.0f);

    float3 N = make_float3(normalTS.x * tangent.x + normalTS.y * bitangent.x + normalTS.z * planeNormal.x,
                           normalTS.x * tangent.y + normalTS.y * bitangent.y + normalTS.z * planeNormal.y,
                           normalTS.x * tangent.z + normalTS.y * bitangent.z + normalTS.z * planeNormal.z);
    N = makeNormalized(N);

    float3 V = makeNormalized(make_float3(cameraPos.x - hit.x,
                                          cameraPos.y - hit.y,
                                          cameraPos.z - hit.z));

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

    if (enableShadows) {
        // Placeholder for procedural shadowing; to be implemented.
    }
    if (enableCameraSmudge && cameraSmudgeImage &&
        cameraSmudgeWidth == width && cameraSmudgeHeight == height) {
        float4 smudgeSample = fetchOverlayPixel(cameraSmudgeImage, cameraSmudgeChannels, idx);
        float alpha = clamp01(smudgeSample.w);
        color.x = color.x * (1.0f - alpha) + smudgeSample.x * alpha;
        color.y = color.y * (1.0f - alpha) + smudgeSample.y * alpha;
        color.z = color.z * (1.0f - alpha) + smudgeSample.z * alpha;
    }
    color.x = fmaxf(color.x, 0.0f);
    color.y = fmaxf(color.y, 0.0f);
    color.z = fmaxf(color.z, 0.0f);

    outRGBA[4 * idx + 0] = color.x;
    outRGBA[4 * idx + 1] = color.y;
    outRGBA[4 * idx + 2] = color.z;
    outRGBA[4 * idx + 3] = 1.0f;
}

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
                       const float* cameraSmudgeImage,
                       int cameraSmudgeWidth, int cameraSmudgeHeight,
                       int cameraSmudgeChannels) {
    shadeKernel<<<gridDim, blockDim>>>(envTex, specularTex, specularMipLevels,
                                       irradianceTex, brdfLutTex, albedo, normal,
                                       roughness, metallic, outRGBA,
                                       width, height, cameraPos,
                                       cameraForward, cameraRight,
                                       cameraUp, tanHalfFovY, aspect,
                                       enableShadows, enableCameraSmudge, cameraSmudgeImage,
                                       cameraSmudgeWidth, cameraSmudgeHeight,
                                       cameraSmudgeChannels);
}