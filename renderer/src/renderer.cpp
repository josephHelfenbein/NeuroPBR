#include <renderer.h>
#include <hdr_image.h>
#include <env_cache.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include <prefilter.cuh>
#include <brdf.cuh>
#include <shading.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ScopedArray {
    cudaArray_t value = nullptr;
    ~ScopedArray() { reset(); }
    void reset(cudaArray_t newValue = nullptr) {
        if (value) {
            cudaFreeArray(value);
        }
        value = newValue;
    }
    cudaArray_t release() {
        cudaArray_t tmp = value;
        value = nullptr;
        return tmp;
    }
};

struct ScopedTexture {
    cudaTextureObject_t value = 0;
    ~ScopedTexture() { reset(); }
    void reset(cudaTextureObject_t newValue = 0) {
        if (value) {
            cudaDestroyTextureObject(value);
        }
        value = newValue;
    }
    cudaTextureObject_t release() {
        cudaTextureObject_t tmp = value;
        value = 0;
        return tmp;
    }
};

struct ScopedSurface {
    cudaSurfaceObject_t value = 0;
    ~ScopedSurface() { reset(); }
    void reset(cudaSurfaceObject_t newValue = 0) {
        if (value) {
            cudaDestroySurfaceObject(value);
        }
        value = newValue;
    }
    cudaSurfaceObject_t release() {
        cudaSurfaceObject_t tmp = value;
        value = 0;
        return tmp;
    }
};

void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string message = std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                              " for `" + expr + "`: " + cudaGetErrorString(err);
        throw CudaError(message);
    }
}

namespace {

dim3 choose2DBlock(int totalThreads) {
    if (totalThreads <= 0) {
        return dim3(16, 16, 1);
    }
    const int warp = 32;
    int blockX = warp;
    while (blockX > 1 && totalThreads % blockX != 0) {
        blockX >>= 1;
    }
    if (totalThreads % blockX != 0) {
        blockX = totalThreads;
    }
    int blockY = std::max(totalThreads / blockX, 1);

    while (blockY > warp && blockX < 64 && blockX * 2 <= 1024) {
        blockX *= 2;
        blockY = std::max(totalThreads / blockX, 1);
    }

    return dim3(static_cast<unsigned>(blockX),
                static_cast<unsigned>(blockY),
                1u);
}

dim3 chooseShadeBlock() {
    int minShadeGrid = 0;
    int optimalShadeBlockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minShadeGrid, &optimalShadeBlockSize, (void*)shadeKernel, 0, 0));
    return choose2DBlock(optimalShadeBlockSize);
}

} // namespace

void renderPlane(const EnvironmentCubemap& env, const BRDFLookupTable& brdf,
                 const float4* dAlbedo, const float4* dNormal,
                 const float* dRoughness, const float* dMetallic,
                 uint8_t* dFrame,
                 int width, int height, uint8_t* hostFrameRGB,
                 cudaStream_t stream,
                 bool enableShadows,
                 bool enableCameraArtifacts,
                 unsigned long long artifactSeed) {
    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t frameBytes = pixelCount * 3u;

    static const dim3 block = chooseShadeBlock();
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> polarDist(0.0f, 360.0f);
    std::uniform_real_distribution<float> azimuthDist(0.0f, 40.0f);

    const float degToRad = static_cast<float>(M_PI) / 180.0f;
    float theta = polarDist(rng) * degToRad;
    float phi = azimuthDist(rng) * degToRad;
    constexpr float radius = 0.6f;

    float camX = radius * sinf(phi) * cosf(theta);
    float camY = radius * cosf(phi);
    float camZ = radius * sinf(phi) * sinf(theta);

    auto normalizeVec = [](const float3& v) {
        float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len < 1e-6f) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        float inv = 1.0f / len;
        return make_float3(v.x * inv, v.y * inv, v.z * inv);
    };

    auto crossVec = [](const float3& a, const float3& b) {
        return make_float3(a.y * b.z - a.z * b.y,
                           a.z * b.x - a.x * b.z,
                           a.x * b.y - a.y * b.x);
    };

    auto dotVec = [](const float3& a, const float3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    };

    float3 cameraPos = make_float3(camX, camY, camZ);
    float3 forward = normalizeVec(make_float3(-camX, -camY, -camZ));
    if (forward.x == 0.0f && forward.y == 0.0f && forward.z == 0.0f) {
        forward = make_float3(0.0f, -1.0f, 0.0f);
    }

    float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
    if (std::fabs(dotVec(forward, worldUp)) > 0.99f) {
        worldUp = make_float3(0.0f, 0.0f, 1.0f);
    }

    float3 right = normalizeVec(crossVec(worldUp, forward));
    if (right.x == 0.0f && right.y == 0.0f && right.z == 0.0f) {
        worldUp = make_float3(0.0f, 0.0f, 1.0f);
        right = normalizeVec(crossVec(worldUp, forward));
    }
    float3 up = normalizeVec(crossVec(forward, right));

    constexpr float kExposureJitterEV = 1.0f;
    std::uniform_real_distribution<float> exposureJitterDist(-kExposureJitterEV, kExposureJitterEV);
    const float exposureJitterEV = exposureJitterDist(rng);

    constexpr float verticalFovDeg = 55.0f;
    float tanHalfFovY = static_cast<float>(std::tan(verticalFovDeg * 0.5f * degToRad));
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    launchShadeKernel(grid, block,
                      env.envTexture, env.specularTexture,
                      static_cast<int>(env.mipLevels),
                      env.irradianceTexture, brdf.texture,
                      dAlbedo, dNormal,
                      dRoughness, dMetallic,
                      dFrame,
                      width, height, cameraPos, forward,
                      right, up, tanHalfFovY, aspectRatio,
                      enableShadows, enableCameraArtifacts, artifactSeed,
                      env.horizonBrightness,
                      env.zenithBrightness, env.hardness,
                      exposureJitterEV, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(hostFrameRGB, dFrame, frameBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

cudaTextureObject_t createCubemapTexture(cudaArray_t array) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;

    cudaTextureDesc desc{};
    desc.normalizedCoords = 1;
    desc.sRGB = 0;
    desc.readMode = cudaReadModeElementType;
    desc.filterMode = cudaFilterModeLinear;
    for (int i = 0; i < 3; ++i) {
        desc.addressMode[i] = cudaAddressModeClamp;
    }

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &desc, nullptr));
    return tex;
}

cudaTextureObject_t createCubemapMipTexture(cudaMipmappedArray_t array, unsigned mipLevels) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeMipmappedArray;
    res.res.mipmap.mipmap = array;

    cudaTextureDesc desc{};
    desc.normalizedCoords = 1;
    desc.sRGB = 0;
    desc.readMode = cudaReadModeElementType;
    desc.filterMode = cudaFilterModeLinear;
    desc.mipmapFilterMode = cudaFilterModeLinear;
    for (int i = 0; i < 3; ++i) {
        desc.addressMode[i] = cudaAddressModeClamp;
    }
    desc.minMipmapLevelClamp = 0.0f;
    desc.maxMipmapLevelClamp = static_cast<float>(mipLevels - 1);

    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &desc, nullptr));
    return tex;
}

void copyHDRToCudaArray(const HDRImage& image, cudaArray_t array) {
    size_t rowBytes = static_cast<size_t>(image.width) * sizeof(float4);
    CUDA_CHECK(cudaMemcpy2DToArray(array, 0, 0, image.pixels.data(), rowBytes, rowBytes, image.height, cudaMemcpyHostToDevice));
}

namespace {

static_assert(sizeof(float4) == 4 * sizeof(float), "cache buffers hold float4 texels as 4 floats");

constexpr unsigned kEnvArrayFlags = cudaArrayCubemap | cudaArraySurfaceLoadStore;

void allocateEnvironmentArrays(EnvironmentCubemap& env) {
    cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();
    const cudaExtent cubeExtent = make_cudaExtent(env.faceSize, env.faceSize, 6);
    CUDA_CHECK(cudaMalloc3DArray(&env.envArray, &float4Desc, cubeExtent, kEnvArrayFlags));
    CUDA_CHECK(cudaMallocMipmappedArray(&env.specularArray, &float4Desc, cubeExtent, env.mipLevels, kEnvArrayFlags));
    const cudaExtent irradianceExtent = make_cudaExtent(env.irradianceSize, env.irradianceSize, 6);
    CUDA_CHECK(cudaMalloc3DArray(&env.irradianceArray, &float4Desc, irradianceExtent, kEnvArrayFlags));
}

void createEnvironmentTextures(EnvironmentCubemap& env) {
    env.envTexture = createCubemapTexture(env.envArray);
    env.specularTexture = createCubemapMipTexture(env.specularArray, env.mipLevels);
    env.irradianceTexture = createCubemapTexture(env.irradianceArray);
}

cudaArray_t specularLevel(const EnvironmentCubemap& env, unsigned level) {
    cudaArray_t levelArray = nullptr;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, env.specularArray, level));
    return levelArray;
}

// Roughness 0 prefilters to the env map itself, so just copy it
void copyEnvToSpecularLevel0(const EnvironmentCubemap& env) {
    cudaMemcpy3DParms copyParams{};
    copyParams.srcArray = env.envArray;
    copyParams.dstArray = specularLevel(env, 0);
    copyParams.extent = make_cudaExtent(env.faceSize, env.faceSize, 6);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
}

void copyCubemapArrayToHost(cudaArray_t array, unsigned faceDim, std::vector<float>& host) {
    host.resize(envCubemapFloatCount(faceDim));
    cudaMemcpy3DParms params{};
    params.srcArray = array;
    params.dstPtr = make_cudaPitchedPtr(host.data(), faceDim * sizeof(float4), faceDim, faceDim);
    params.extent = make_cudaExtent(faceDim, faceDim, 6); // in elements, since one side is an array
    params.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&params));
}

void copyHostToCubemapArray(const std::vector<float>& host, unsigned faceDim, cudaArray_t array) {
    if (host.size() != envCubemapFloatCount(faceDim)) {
        throw std::runtime_error("Cubemap host buffer has " + std::to_string(host.size()) +
                                 " floats, expected " + std::to_string(envCubemapFloatCount(faceDim)));
    }
    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(const_cast<float*>(host.data()), faceDim * sizeof(float4), faceDim, faceDim);
    params.dstArray = array;
    params.extent = make_cudaExtent(faceDim, faceDim, 6);
    params.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&params));
}

EnvCacheData downloadEnvironment(const EnvironmentCubemap& env) {
    EnvCacheData data;
    data.meta.mipLevels = env.mipLevels;
    data.meta.sourceWidth = env.sourceWidth;
    data.meta.sourceHeight = env.sourceHeight;
    data.meta.horizonBrightness = env.horizonBrightness;
    data.meta.zenithBrightness = env.zenithBrightness;
    data.meta.hardness = env.hardness;

    copyCubemapArrayToHost(env.envArray, env.faceSize, data.env);
    data.specular.resize(env.mipLevels > 0 ? env.mipLevels - 1 : 0);
    for (unsigned level = 1; level < env.mipLevels; ++level) {
        copyCubemapArrayToHost(specularLevel(env, level), envMipFaceSize(env.faceSize, level), data.specular[level - 1]);
    }
    copyCubemapArrayToHost(env.irradianceArray, env.irradianceSize, data.irradiance);
    return data;
}

EnvironmentCubemap uploadEnvironment(const std::filesystem::path& filePath, const EnvCacheKey& key,
                                     const EnvCacheData& data) {
    EnvironmentCubemap result;
    result.name = filePath.filename().string();
    result.faceSize = key.faceSize;
    result.irradianceSize = key.irradianceSize;
    result.mipLevels = data.meta.mipLevels;
    result.sourceWidth = data.meta.sourceWidth;
    result.sourceHeight = data.meta.sourceHeight;
    if (result.mipLevels != envMipLevelCount(result.faceSize) || data.specular.size() + 1 != result.mipLevels) {
        throw std::runtime_error("Environment cache data for " + result.name + " has an unexpected mip count");
    }

    allocateEnvironmentArrays(result);
    copyHostToCubemapArray(data.env, result.faceSize, result.envArray);
    copyEnvToSpecularLevel0(result);
    for (unsigned level = 1; level < result.mipLevels; ++level) {
        copyHostToCubemapArray(data.specular[level - 1], envMipFaceSize(result.faceSize, level), specularLevel(result, level));
    }
    copyHostToCubemapArray(data.irradiance, result.irradianceSize, result.irradianceArray);
    createEnvironmentTextures(result);

    result.horizonBrightness = data.meta.horizonBrightness;
    result.zenithBrightness = data.meta.zenithBrightness;
    result.hardness = data.meta.hardness;
    return result;
}

} // namespace

EnvironmentCubemap precomputeEnvironmentCubemap(const std::filesystem::path& filePath,
                                                 unsigned faceSize, unsigned irradianceSize,
                                                 unsigned specularSamples, unsigned diffuseSamples,
                                                 unsigned hdrMaxWidth) {
    auto loadStart = std::chrono::steady_clock::now();
    HDRImage hdr = loadHDRImage(filePath, static_cast<int>(hdrMaxWidth));
    auto loadMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loadStart).count();
    std::cout << "  HDRI " << hdr.sourceWidth << "x" << hdr.sourceHeight << " -> "
              << hdr.width << "x" << hdr.height << " (loaded in " << loadMs << " ms)" << std::endl;

    cudaChannelFormatDesc float4Desc = cudaCreateChannelDesc<float4>();

    ScopedArray hdrArray;
    CUDA_CHECK(cudaMallocArray(&hdrArray.value, &float4Desc, hdr.width, hdr.height));
    copyHDRToCudaArray(hdr, hdrArray.value);

    cudaResourceDesc hdrRes{};
    hdrRes.resType = cudaResourceTypeArray;
    hdrRes.res.array.array = hdrArray.value;

    cudaTextureDesc hdrTexDesc{};
    hdrTexDesc.normalizedCoords = 1;
    hdrTexDesc.readMode = cudaReadModeElementType;
    hdrTexDesc.filterMode = cudaFilterModeLinear;
    hdrTexDesc.addressMode[0] = cudaAddressModeWrap;
    hdrTexDesc.addressMode[1] = cudaAddressModeClamp;

    ScopedTexture hdrTexture;
    CUDA_CHECK(cudaCreateTextureObject(&hdrTexture.value, &hdrRes, &hdrTexDesc, nullptr));

    EnvironmentCubemap result;
    result.name = filePath.filename().string();
    result.faceSize = faceSize;
    result.irradianceSize = irradianceSize;
    result.mipLevels = envMipLevelCount(faceSize);
    result.sourceWidth = hdr.sourceWidth;
    result.sourceHeight = hdr.sourceHeight;

    allocateEnvironmentArrays(result);
    createEnvironmentTextures(result);

    cudaResourceDesc envSurfRes{};
    envSurfRes.resType = cudaResourceTypeArray;
    envSurfRes.res.array.array = result.envArray;

    ScopedSurface envSurface;
    CUDA_CHECK(cudaCreateSurfaceObject(&envSurface.value, &envSurfRes));

    const dim3 block(16, 16, 1);
    const dim3 grid((faceSize + block.x - 1) / block.x,
                    (faceSize + block.y - 1) / block.y,
                    6);

    int currentDevice = 0;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    cudaDeviceProp deviceProps{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, currentDevice));
    const unsigned int targetPrefilterBlocks = static_cast<unsigned int>(deviceProps.multiProcessorCount);

    auto makePrefilterGrid = [](unsigned int size, const dim3& blockDim) {
        return dim3((size + blockDim.x - 1u) / blockDim.x,
                    (size + blockDim.y - 1u) / blockDim.y,
                    6u);
    };

    launchEquirectangularToCubemap(grid, block, hdrTexture.value, envSurface.value, static_cast<int>(faceSize));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    envSurface.reset();
    hdrTexture.reset();
    hdrArray.reset();

    const float horizonMinY = 0.0f;
    const float horizonMaxY = 0.35f;
    const float zenithMinY = 0.85f;

    int blockCount = static_cast<int>(grid.x) * static_cast<int>(grid.y) * static_cast<int>(grid.z);

    float* dBlockAccum = nullptr;
    float* dFinalAccum = nullptr;
    CUDA_CHECK(cudaMalloc(&dBlockAccum, static_cast<size_t>(blockCount) * 4u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFinalAccum, 4 * sizeof(float)));

    launchComputeEnvironmentBrightness(grid, block,
                                        result.envTexture,
                                        static_cast<int>(faceSize),
                                        horizonMinY, horizonMaxY,
                                        zenithMinY, dBlockAccum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    launchReduceEnvironmentBrightness(dBlockAccum, blockCount, dFinalAccum);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float hostBrightness[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMemcpy(hostBrightness, dFinalAccum, sizeof(hostBrightness), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dBlockAccum));
    CUDA_CHECK(cudaFree(dFinalAccum));

    float horizonSum = hostBrightness[0];
    float horizonWeight = hostBrightness[1];
    float zenithSum = hostBrightness[2];
    float zenithWeight = hostBrightness[3];

    result.horizonBrightness = (horizonWeight > 1e-6f) ? horizonSum / horizonWeight : 0.0f;
    result.zenithBrightness = (zenithWeight > 1e-6f) ? zenithSum / zenithWeight : 0.0f;
    const float hardnessScale = 0.5f;
    const float minBlur = 0.02f;
    const float maxBlur = 0.12f;
    auto saturate = [](float value) { return std::max(0.0f, std::min(1.0f, value)); };

    float normalizedHardness = saturate((result.zenithBrightness - result.horizonBrightness) * hardnessScale);
    float blurRange = maxBlur - minBlur;
    result.hardness = minBlur + (1.0f - normalizedHardness) * blurRange;

    ScopedTexture envTextureForSampling;
    envTextureForSampling.reset(createCubemapTexture(result.envArray));

    for (unsigned level = 0; level < result.mipLevels; ++level) {
        if (level == 0) {
            copyEnvToSpecularLevel0(result);
            continue;
        }

        cudaResourceDesc levelRes{};
        levelRes.resType = cudaResourceTypeArray;
        levelRes.res.array.array = specularLevel(result, level);

        ScopedSurface levelSurface;
        CUDA_CHECK(cudaCreateSurfaceObject(&levelSurface.value, &levelRes));

        unsigned mipFaceSize = envMipFaceSize(faceSize, level);
        dim3 mipBlock = block;
        dim3 mipGrid = makePrefilterGrid(mipFaceSize, mipBlock);
        unsigned int totalBlocks = mipGrid.x * mipGrid.y * mipGrid.z;

        while (totalBlocks < targetPrefilterBlocks && (mipBlock.x > 1u || mipBlock.y > 1u)) {
            if (mipBlock.x >= mipBlock.y && mipBlock.x > 1u) {
                mipBlock.x = std::max(1u, mipBlock.x / 2u);
            } else if (mipBlock.y > 1u) {
                mipBlock.y = std::max(1u, mipBlock.y / 2u);
            }

            mipGrid = makePrefilterGrid(mipFaceSize, mipBlock);
            totalBlocks = mipGrid.x * mipGrid.y * mipGrid.z;
        }
        float roughness = result.mipLevels > 1 ?
                          static_cast<float>(level) / static_cast<float>(result.mipLevels - 1) :
                          0.0f;

        launchPrefilterSpecularCubemap(mipGrid, mipBlock,
                       envTextureForSampling.value,
                       levelSurface.value,
                       static_cast<int>(mipFaceSize),
                       roughness,
                       specularSamples);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        levelSurface.reset();
    }

    envTextureForSampling.reset();

    cudaResourceDesc irradianceRes{};
    irradianceRes.resType = cudaResourceTypeArray;
    irradianceRes.res.array.array = result.irradianceArray;

    ScopedSurface irradianceSurface;
    CUDA_CHECK(cudaCreateSurfaceObject(&irradianceSurface.value, &irradianceRes));

    dim3 irradianceGrid((irradianceSize + block.x - 1) / block.x,
                        (irradianceSize + block.y - 1) / block.y,
                        6);

    launchConvolveDiffuseIrradiance(irradianceGrid, block,
                                    result.envTexture,
                                    irradianceSurface.value,
                                    static_cast<int>(irradianceSize),
                                    diffuseSamples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    irradianceSurface.reset();

    return result;
}

std::vector<std::filesystem::path> collectHDRIFiles(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(root)) {
        std::cerr << "Warning: HDRI directory does not exist: " << root << "\n";
        return files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        });
        if (ext == ".hdr" || ext == ".hdri") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

namespace {

std::string describeBrightness(float horizon, float zenith, float hardness) {
    std::ostringstream out;
    out.precision(9);
    out << "horizon " << horizon << ", zenith " << zenith << ", hardness " << hardness;
    return out.str();
}

// NEUROPBR_ENV_CACHE_VERIFY=1 recomputes each cache hit and checks it matches
void verifyCachedEnvironment(const EnvironmentCubemap& cached, const std::filesystem::path& filePath,
                             unsigned specularSamples, unsigned diffuseSamples, unsigned hdrMaxWidth) {
    EnvironmentCubemap fresh = precomputeEnvironmentCubemap(filePath, cached.faceSize, cached.irradianceSize,
                                                            specularSamples, diffuseSamples, hdrMaxWidth);
    const EnvCacheData a = downloadEnvironment(cached);
    const EnvCacheData b = downloadEnvironment(fresh);
    std::vector<float> a0;
    std::vector<float> b0;
    copyCubemapArrayToHost(specularLevel(cached, 0), cached.faceSize, a0);
    copyCubemapArrayToHost(specularLevel(fresh, 0), fresh.faceSize, b0);

    size_t mismatchedBlocks = 0;
    auto compare = [&](const std::string& what, const std::vector<float>& x, const std::vector<float>& y) {
        size_t differing = 0;
        float maxAbsDiff = 0.0f;
        for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
            if (std::memcmp(&x[i], &y[i], sizeof(float)) != 0) {
                ++differing;
                maxAbsDiff = std::max(maxAbsDiff, std::fabs(x[i] - y[i]));
            }
        }
        if (differing > 0 || x.size() != y.size()) {
            ++mismatchedBlocks;
            std::cout << "  VERIFY MISMATCH in " << what << ": " << differing << " of " << y.size()
                      << " floats differ (max abs diff " << maxAbsDiff << ")" << std::endl;
        }
    };
    compare("env cubemap", a.env, b.env);
    compare("specular mip 0", a0, b0);
    for (size_t i = 0; i < a.specular.size() && i < b.specular.size(); ++i) {
        compare("specular mip " + std::to_string(i + 1), a.specular[i], b.specular[i]);
    }
    compare("irradiance", a.irradiance, b.irradiance);
    compare("brightness", {a.meta.horizonBrightness, a.meta.zenithBrightness, a.meta.hardness},
            {b.meta.horizonBrightness, b.meta.zenithBrightness, b.meta.hardness});
    if (mismatchedBlocks == 0) {
        std::cout << "  verify: cached data is bit-identical to a fresh precompute ("
                  << describeBrightness(b.meta.horizonBrightness, b.meta.zenithBrightness, b.meta.hardness)
                  << ")" << std::endl;
    }
}

} // namespace

std::vector<EnvironmentCubemap> loadEnvironmentCubemaps(const std::filesystem::path& directory,
                                                         unsigned faceSize, unsigned irradianceSize,
                                                         unsigned specularSamples, unsigned diffuseSamples,
                                                         unsigned hdrMaxWidth,
                                                         const std::filesystem::path& cacheDir) {
    using Clock = std::chrono::steady_clock;
    auto msSince = [](Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    };
    const auto totalStart = Clock::now();

    std::vector<std::filesystem::path> paths = collectHDRIFiles(directory);
    std::vector<EnvironmentCubemap> environments;
    environments.reserve(paths.size());

    const bool cacheEnabled = !cacheDir.empty();
    bool cacheWritable = cacheEnabled;
    const char* verifyEnv = std::getenv("NEUROPBR_ENV_CACHE_VERIFY");
    const bool verifyHits = verifyEnv != nullptr && std::string(verifyEnv) == "1";
    size_t cacheHits = 0;

    for (size_t i = 0; i < paths.size(); ++i) {
        const auto& path = paths[i];
        std::cout << "Environment " << (i + 1) << "/" << paths.size() << ": " << path << std::endl;
        const auto start = Clock::now();

        EnvCacheKey key;
        std::filesystem::path cacheFile;
        bool useCache = false;
        if (cacheEnabled) {
            std::string error;
            useCache = makeEnvCacheKey(path, faceSize, irradianceSize, specularSamples, diffuseSamples, hdrMaxWidth, key, error);
            if (useCache) {
                cacheFile = envCacheFilePath(cacheDir, key);
            } else {
                std::cerr << "  Warning: not caching this HDRI (" << error << ")" << std::endl;
            }
        }

        if (useCache) {
            EnvCacheData data;
            std::string reason;
            const EnvCacheReadStatus status = readEnvCache(cacheFile, key, data, reason);
            if (status == EnvCacheReadStatus::Hit) {
                environments.push_back(uploadEnvironment(path, key, data));
                ++cacheHits;
                const EnvironmentCubemap& env = environments.back();
                std::cout << "  HDRI " << env.sourceWidth << "x" << env.sourceHeight << ": cache hit ("
                          << msSince(start) << " ms; "
                          << describeBrightness(env.horizonBrightness, env.zenithBrightness, env.hardness) << ")"
                          << std::endl;
                if (verifyHits) {
                    verifyCachedEnvironment(env, path, specularSamples, diffuseSamples, hdrMaxWidth);
                }
                continue;
            }
            if (status == EnvCacheReadStatus::Rejected) {
                std::cout << "  Ignoring cache file " << cacheFile << ": " << reason << std::endl;
            }
        }

        EnvironmentCubemap env = precomputeEnvironmentCubemap(path, faceSize, irradianceSize, specularSamples, diffuseSamples, hdrMaxWidth);
        const auto computeMs = msSince(start);
        const std::string brightness = describeBrightness(env.horizonBrightness, env.zenithBrightness, env.hardness);
        if (useCache && cacheWritable) {
            const auto writeStart = Clock::now();
            std::string error;
            if (writeEnvCache(cacheFile, key, downloadEnvironment(env), error)) {
                std::cout << "  cache miss -> precomputed in " << computeMs << " ms (" << brightness
                          << "), wrote " << cacheFile << " in " << msSince(writeStart) << " ms" << std::endl;
            } else {
                cacheWritable = false;
                std::cerr << "  Warning: cannot write the environment cache (" << error
                          << "); continuing without writing cache files." << std::endl;
                std::cout << "  cache miss -> precomputed in " << computeMs << " ms (" << brightness
                          << "), not cached" << std::endl;
            }
        } else {
            std::cout << "  precomputed in " << computeMs << " ms (" << brightness << ")" << std::endl;
        }
        environments.push_back(std::move(env));
    }

    std::cout << "Loaded " << environments.size() << " environments in " << msSince(totalStart) << " ms";
    if (cacheEnabled) {
        std::cout << " (" << cacheHits << " from cache, " << environments.size() - cacheHits << " precomputed)";
    }
    std::cout << std::endl;
    return environments;
}

BRDFLookupTable createBRDFLUT(unsigned size) {
    BRDFLookupTable lut;
    lut.size = size;

    cudaChannelFormatDesc float2Desc = cudaCreateChannelDesc<float2>();
    CUDA_CHECK(cudaMallocArray(&lut.array, &float2Desc, size, size, cudaArraySurfaceLoadStore));

    cudaResourceDesc surfDesc{};
    surfDesc.resType = cudaResourceTypeArray;
    surfDesc.res.array.array = lut.array;
    CUDA_CHECK(cudaCreateSurfaceObject(&lut.surface, &surfDesc));

    cudaResourceDesc texRes{};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = lut.array;

    cudaTextureDesc texDesc{};
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = 0;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

    CUDA_CHECK(cudaCreateTextureObject(&lut.texture, &texRes, &texDesc, nullptr));

    return lut;
}

void loadBRDFLUT(BRDFLookupTable& lut) {
    if (lut.surface == 0 || lut.array == nullptr || lut.size == 0) {
        throw std::runtime_error("BRDF LUT resources not initialized");
    }

    const dim3 block(16, 16, 1);
    const dim3 grid((lut.size + block.x - 1) / block.x,
                    (lut.size + block.y - 1) / block.y,
                    1);

    launchPrecomputeBRDF(grid, block, lut.surface, static_cast<int>(lut.size), static_cast<int>(lut.size));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}