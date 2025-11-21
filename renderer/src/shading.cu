#include <shading.cuh>

#include <device_launch_parameters.h>
#include <math.h>

#include <utils.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ inline float hash21(const float2& p) {
    float n = dot2(p, make_float2(127.1f, 311.7f));
    return fractf(sinf(n) * 43758.5453f);
}

__device__ inline float valueNoise(const float2& p) {
    float2 i = make_float2(floorf(p.x), floorf(p.y));
    float2 f = make_float2(fractf(p.x), fractf(p.y));

    float a = hash21(i);
    float b = hash21(make_float2(i.x + 1.0f, i.y));
    float c = hash21(make_float2(i.x, i.y + 1.0f));
    float d = hash21(make_float2(i.x + 1.0f, i.y + 1.0f));

    float2 u = make_float2(f.x * f.x * (3.0f - 2.0f * f.x),
                           f.y * f.y * (3.0f - 2.0f * f.y));

    float lerpX1 = lerp(a, b, u.x);
    float lerpX2 = lerp(c, d, u.x);
    return lerp(lerpX1, lerpX2, u.y);
}

__device__ inline float fbm(const float2& p) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    for (int i = 0; i < 4; ++i) {
        float2 sample = make_float2(p.x * frequency, p.y * frequency);
        sum += amplitude * valueNoise(sample);
        frequency *= 2.0f;
        amplitude *= 0.5f;
    }
    return clamp01(sum);
}

__device__ inline float2 gradientNoise2D(const float2& p) {
    float angle = valueNoise(p) * 6.28318530718f;
    return make_float2(cosf(angle), sinf(angle));
}

__device__ inline float worley(const float2& p, float jitter) {
    float2 cell = make_float2(floorf(p.x), floorf(p.y));
    float minDist = 1e6f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 neighbor = make_float2(cell.x + static_cast<float>(i),
                                          cell.y + static_cast<float>(j));
            float2 offset = make_float2(hash21(neighbor), hash21(make_float2(neighbor.x + 19.19f, neighbor.y + 73.17f)));
            float2 jittered = make_float2((offset.x - 0.5f) * jitter,
                                          (offset.y - 0.5f) * jitter);
            float2 feature = make_float2(neighbor.x + 0.5f + jittered.x,
                                         neighbor.y + 0.5f + jittered.y);
            float2 diff = make_float2(p.x - feature.x, p.y - feature.y);
            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
            minDist = fminf(minDist, dist);
        }
    }
    return minDist;
}

__device__ inline float computeVoronoiShadowMask(float2 uv, float2 offset, float2 rawSeed) {
    float2 frameOffset = offset * 5.0f; 

    float3 seed = make_float3(12.34f + rawSeed.x * 7.0f, 56.78f + rawSeed.y * 11.0f, 90.12f - rawSeed.x * 5.0f);

    float blobFrequency = 1.8f + rawSeed.y * 0.4f;
    float blobJitter = clamp01(0.8f + rawSeed.x * 0.15f);
    
    float2 blobUV = make_float2((uv.x + frameOffset.x) * blobFrequency + seed.z * 17.3f,
                                (uv.y + frameOffset.y) * blobFrequency + seed.x * 23.7f);
    float blobDist = worley(blobUV, blobJitter);
    
    float blobFalloff = smoothstepf(0.25f, 0.45f, blobDist);
    float blobMask = 1.0f - blobFalloff;
    
    float blob2Frequency = 2.4f - rawSeed.x * 0.5f;
    float2 blob2UV = make_float2((uv.x - frameOffset.y * 0.7f) * blob2Frequency + seed.y * 31.1f,
                                 (uv.y + frameOffset.x * 0.6f) * blob2Frequency - seed.z * 19.4f);
    float blob2Dist = worley(blob2UV, blobJitter * 0.85f);
    float blob2Falloff = smoothstepf(0.25f, 0.42f, blob2Dist);
    float blob2Mask = 1.0f - blob2Falloff;
    
    float combinedBlobs = fmaxf(blobMask, blob2Mask * 0.6f);
    
    float warpFrequency = 6.0f + rawSeed.y * 2.0f;
    float warpStrength = 0.12f + rawSeed.x * 0.05f;
    float2 warpSample = make_float2((uv.x + seed.z * 5.3f) * warpFrequency,
                                    (uv.y + seed.y * 7.1f) * warpFrequency);
    float2 warpVec = gradientNoise2D(warpSample);
    float2 warp = make_float2(warpVec.x * warpStrength, warpVec.y * warpStrength);
    
    float2 warpedUV = make_float2(uv.x + warp.x, uv.y + warp.y);
    float2 warpedBlobUV = make_float2((warpedUV.x + frameOffset.x) * blobFrequency + seed.z * 17.3f,
                                      (warpedUV.y + frameOffset.y) * blobFrequency + seed.x * 23.7f);
    float warpedBlobDist = worley(warpedBlobUV, blobJitter);
    float warpedBlobFalloff = smoothstepf(0.25f, 0.45f, warpedBlobDist);
    float warpedBlobMask = 1.0f - warpedBlobFalloff;
    float edgeBlob = lerp(combinedBlobs, warpedBlobMask, 0.35f);
    
    float detailFrequency = 15.0f + rawSeed.x * 3.0f;
    float detailNoise = fbm(make_float2((uv.x + frameOffset.x * 0.3f) * detailFrequency + seed.y * 41.0f,
                                        (uv.y + frameOffset.y * 0.3f) * detailFrequency + seed.z * 37.0f));
    
    float detailModulation = 0.85f + detailNoise * 0.25f;
    float finalMask = clamp01(edgeBlob * detailModulation);
    
    return finalMask;
}

__device__ inline float computeProceduralShadow(float2 uv, float hardness,
                                                float horizonBrightness,
                                                const float3& irradiance,
                                                float2 offset, float2 rawSeed) {
    float shadowMask = computeVoronoiShadowMask(uv, offset, rawSeed);

    float luminance = dot3(irradiance, make_float3(0.2126f, 0.7152f, 0.0722f));
    float horizonFloor = fmaxf(horizonBrightness, 1e-4f);
    float horizonRatio = luminance / horizonFloor;
    float horizonValid = horizonBrightness > 1e-3f ? 1.0f : 0.0f;
    float horizonBoost = horizonValid * clamp01((horizonRatio - 1.0f) * 0.4f);
    shadowMask = lerp(shadowMask, 0.0f, horizonBoost * 0.3f);

    float shadowDarkness = 0.20f + hardness * 0.20f;
    float lightMultiplier = lerp(1.0f, shadowDarkness, shadowMask);

    const float minLightScale = 0.15f;
    float minLight = fmaxf(0.10f, clamp01(luminance * minLightScale));

    return clamp01(fmaxf(minLight, lightMultiplier));
}

__device__ inline float hash12(float2 p) {
    float3 p3  = fractf(make_float3(p.x, p.y, p.x) * .1031f);
    p3 += dot3(p3, make_float3(p3.y + 33.33f, p3.z + 33.33f, p3.x + 33.33f));
    return fractf((p3.x + p3.y) * p3.z);
}

__device__ float4 computeProceduralArtifacts(float2 uv, unsigned long long seed) {
    float s1 = (float)(seed % 10000) / 10000.0f;
    float s2 = (float)((seed / 10000) % 10000) / 10000.0f;
    
    float2 smudgeUV = uv * 2.0f;
    smudgeUV.x += s1 * 10.0f;
    smudgeUV.y += s2 * 10.0f;
    
    float w = fbm(smudgeUV * 0.5f);
    smudgeUV.x += w * 0.5f;
    smudgeUV.y += w * 0.5f;
    
    float smudgeNoise = fbm(smudgeUV * 4.0f);
    
    float smudge = smoothstepf(0.45f, 0.85f, smudgeNoise);
    smudge *= 0.15f;
    
    float scratches = 0.0f;
    for(int i=0; i<3; ++i) {
        float angle = hash12(make_float2(s1 + i * 0.1f, s2)) * 3.14159f;
        float c = cosf(angle);
        float s = sinf(angle);
        float2 rotUV = make_float2(uv.x * c - uv.y * s, uv.x * s + uv.y * c);
        
        float freq = 80.0f + hash12(make_float2(s2, s1 + i * 0.2f)) * 120.0f;
        float n = valueNoise(make_float2(rotUV.x * freq + s1 * 20.0f, rotUV.y * 1.5f + s2 * 20.0f));
        
        float cut = 1.0f - abs(n - 0.5f) * 2.0f;
        cut = powf(cut, 40.0f);
        
        float maskFreq = 4.0f;
        float mask = valueNoise(uv * maskFreq + make_float2(s1 * 30.0f + i, s2 * 30.0f));
        cut *= smoothstepf(0.6f, 0.8f, mask);
        
        scratches += cut;
    }
    scratches = clamp01(scratches * 0.3f);
    
    float3 smudgeColor = make_float3(0.8f, 0.75f, 0.7f); 
    float3 scratchColor = make_float3(0.95f, 0.95f, 0.95f);
    
    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);
    float finalAlpha = 0.0f;
    
    finalColor = smudgeColor;
    finalAlpha = smudge;
    
    finalColor = lerp(finalColor, scratchColor, scratches);
    finalAlpha = fmaxf(finalAlpha, scratches);
    
    return make_float4(finalColor.x, finalColor.y, finalColor.z, finalAlpha);
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
                 bool enableShadows, bool enableCameraArtifacts,
                 unsigned long long artifactSeed,
                 float horizonBrightness,
                 float zenithBrightness, float hardness) {
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

    float4 irradianceSample = texCubemap<float4>(irradianceTex, N.x, N.y, N.z);
    float3 irradiance = make_float3(irradianceSample.x, irradianceSample.y, irradianceSample.z);
    float shadowMultiplier = 1.0f;
    if (enableShadows) {
        float s1 = (float)(artifactSeed % 1000) / 1000.0f;
        float s2 = (float)((artifactSeed / 1000) % 1000) / 1000.0f;
        
        s1 = s1 * 2.0f - 1.0f;
        s2 = s2 * 2.0f - 1.0f;
        
        float2 shadowOffset = make_float2(s1, s2) * 0.2f;
        
        // Large constant so we are in a "good" part of the noise
        shadowOffset.x += 12.34f;
        shadowOffset.y += 56.78f;

        shadowMultiplier = computeProceduralShadow(make_float2(u, v), hardness,
                                                   horizonBrightness, irradiance, shadowOffset, make_float2(s1, s2));
        albedoColor.x *= shadowMultiplier;
        albedoColor.y *= shadowMultiplier;
        albedoColor.z *= shadowMultiplier;
    }

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

    if (enableShadows) {
        const float specularShadowStrength = 0.7f;
        float specFactor = lerp(1.0f, shadowMultiplier, specularShadowStrength);
        prefilteredColor.x *= specFactor;
        prefilteredColor.y *= specFactor;
        prefilteredColor.z *= specFactor;
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

    if (enableCameraArtifacts) {
        float uScreen = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
        float vScreen = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        float4 artifact = computeProceduralArtifacts(make_float2(uScreen, vScreen), artifactSeed);
        
        float alpha = clamp01(artifact.w);
        color.x = color.x * (1.0f - alpha) + artifact.x * alpha;
        color.y = color.y * (1.0f - alpha) + artifact.y * alpha;
        color.z = color.z * (1.0f - alpha) + artifact.z * alpha;
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
                       bool enableShadows, bool enableCameraArtifacts,
                       unsigned long long artifactSeed,
                       float horizonBrightness,
                       float zenithBrightness, float hardness) {
    shadeKernel<<<gridDim, blockDim>>>(envTex, specularTex, specularMipLevels,
                                       irradianceTex, brdfLutTex, albedo, normal,
                                       roughness, metallic, outRGBA,
                                       width, height, cameraPos,
                                       cameraForward, cameraRight,
                                       cameraUp, tanHalfFovY, aspect,
                                       enableShadows, enableCameraArtifacts, artifactSeed,
                                       horizonBrightness, zenithBrightness, hardness);
}