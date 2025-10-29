#include <renderer.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <prefilter.cuh>
#include <brdf.cuh>
#include <shading.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct HDRImage {
    int width = 0;
    int height = 0;
    std::vector<float4> pixels;
};

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

struct ScopedMipmappedArray {
    cudaMipmappedArray_t value = nullptr;
    ~ScopedMipmappedArray() { reset(); }
    void reset(cudaMipmappedArray_t newValue = nullptr) {
        if (value) {
            cudaFreeMipmappedArray(value);
        }
        value = newValue;
    }
    cudaMipmappedArray_t release() {
        cudaMipmappedArray_t tmp = value;
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

const std::vector<FloatImage>& emptyFloatImageVector() {
    static const std::vector<FloatImage> kEmpty;
    return kEmpty;
}

void renderPlane(const EnvironmentCubemap& env, const BRDFLookupTable& brdf,
                 const float* albedo, const float* normal,
                 const float* roughness, const float* metallic,
                 int width, int height, std::vector<float4>& frameRGBA,
                 bool enableShadows,
                 bool enableCameraSmudge,
                 bool enableLensFlare,
                 const std::vector<FloatImage>& cameraSmudges,
                 const std::vector<FloatImage>& lensFlares) {
    size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    size_t vec3Bytes = pixelCount * 3 * sizeof(float);
    size_t scalarBytes = pixelCount * sizeof(float);
    size_t frameBytes = pixelCount * sizeof(float4);

    float *dAlbedo = nullptr;
    float *dNormal = nullptr;
    float *dRoughness = nullptr;
    float *dMetallic = nullptr;
    float4* dFrame = nullptr;

    CUDA_CHECK(cudaMalloc(&dAlbedo, vec3Bytes));
    CUDA_CHECK(cudaMalloc(&dNormal, vec3Bytes));
    CUDA_CHECK(cudaMalloc(&dRoughness, scalarBytes));
    CUDA_CHECK(cudaMalloc(&dMetallic, scalarBytes));
    CUDA_CHECK(cudaMalloc(&dFrame, frameBytes));

    CUDA_CHECK(cudaMemcpy(dAlbedo, albedo, vec3Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dNormal, normal, vec3Bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dRoughness, roughness, scalarBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMetallic, metallic, scalarBytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    static thread_local std::mt19937 rng([] {
        std::random_device rd;
        return std::mt19937(rd());
    }());
    std::uniform_real_distribution<float> polarDist(0.0f, 360.0f);
    std::uniform_real_distribution<float> azimuthDist(0.0f, 40.0f);

    const float degToRad = static_cast<float>(M_PI) / 180.0f;
    float theta = polarDist(rng) * degToRad;
    float phi = azimuthDist(rng) * degToRad;
    constexpr float radius = 0.3f;

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

    constexpr float verticalFovDeg = 55.0f;
    float tanHalfFovY = static_cast<float>(std::tan(verticalFovDeg * 0.5f * degToRad));
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    float* dCameraSmudge = nullptr;
    int cameraSmudgeWidth = 0;
    int cameraSmudgeHeight = 0;
    int cameraSmudgeChannels = 0;

    float* dLensFlare = nullptr;
    int lensFlareWidth = 0;
    int lensFlareHeight = 0;
    int lensFlareChannels = 0;

    auto selectOverlay = [&rng](const std::vector<FloatImage>& images) -> const FloatImage* {
        if (images.empty()) {
            return nullptr;
        }
        std::uniform_int_distribution<size_t> indexDist(0, images.size() - 1);
        return &images[indexDist(rng)];
    };

    const FloatImage* selectedSmudge = nullptr;
    if (enableCameraSmudge) {
        selectedSmudge = selectOverlay(cameraSmudges);
        if (selectedSmudge) {
            size_t expectedSize = static_cast<size_t>(selectedSmudge->width) *
                                  static_cast<size_t>(selectedSmudge->height) *
                                  static_cast<size_t>(selectedSmudge->channels);
            if (selectedSmudge->width == width && selectedSmudge->height == height &&
                selectedSmudge->channels > 0 &&
                selectedSmudge->data.size() == expectedSize) {
                size_t smudgeBytes = selectedSmudge->data.size() * sizeof(float);
                CUDA_CHECK(cudaMalloc(&dCameraSmudge, smudgeBytes));
                CUDA_CHECK(cudaMemcpy(dCameraSmudge, selectedSmudge->data.data(), smudgeBytes, cudaMemcpyHostToDevice));
                cameraSmudgeWidth = selectedSmudge->width;
                cameraSmudgeHeight = selectedSmudge->height;
                cameraSmudgeChannels = selectedSmudge->channels;
            } else {
                selectedSmudge = nullptr;
            }
        }
    }

    const FloatImage* selectedLens = nullptr;
    if (enableLensFlare) {
        selectedLens = selectOverlay(lensFlares);
        if (selectedLens) {
            size_t expectedSize = static_cast<size_t>(selectedLens->width) *
                                  static_cast<size_t>(selectedLens->height) *
                                  static_cast<size_t>(selectedLens->channels);
            if (selectedLens->width == width && selectedLens->height == height &&
                selectedLens->channels > 0 &&
                selectedLens->data.size() == expectedSize) {
                size_t lensBytes = selectedLens->data.size() * sizeof(float);
                CUDA_CHECK(cudaMalloc(&dLensFlare, lensBytes));
                CUDA_CHECK(cudaMemcpy(dLensFlare, selectedLens->data.data(), lensBytes, cudaMemcpyHostToDevice));
                lensFlareWidth = selectedLens->width;
                lensFlareHeight = selectedLens->height;
                lensFlareChannels = selectedLens->channels;
            } else {
                selectedLens = nullptr;
            }
        }
    }

    bool smudgeActive = enableCameraSmudge && (selectedSmudge != nullptr);
    bool lensActive = enableLensFlare && (selectedLens != nullptr);

    launchShadeKernel(grid, block,
                      env.envTexture, env.specularTexture,
                      static_cast<int>(env.mipLevels),
                      env.irradianceTexture, brdf.texture,
                      dAlbedo, dNormal, dRoughness, dMetallic,
                      reinterpret_cast<float*>(dFrame),
                      width, height, cameraPos, forward,
                      right, up, tanHalfFovY, aspectRatio,
                      enableShadows, smudgeActive,
                      lensActive, dCameraSmudge,
                      cameraSmudgeWidth, cameraSmudgeHeight,
                      cameraSmudgeChannels, dLensFlare,
                      lensFlareWidth, lensFlareHeight,
                      lensFlareChannels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    frameRGBA.resize(pixelCount);
    CUDA_CHECK(cudaMemcpy(frameRGBA.data(), dFrame, frameBytes, cudaMemcpyDeviceToHost));

    cudaFree(dLensFlare);
    cudaFree(dCameraSmudge);
    cudaFree(dFrame);
    cudaFree(dMetallic);
    cudaFree(dRoughness);
    cudaFree(dNormal);
    cudaFree(dAlbedo);
}
HDRImage loadHDRImage(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open HDRI file: " + path.string());
    }

    auto readLine = [&file]() {
        std::string line;
        std::getline(file, line);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return line;
    };

    std::string header = readLine();
    if (header.rfind("#?", 0) != 0) {
        throw std::runtime_error("Invalid HDRI header (missing #?): " + header);
    }

    for (;;) {
        if (!file) {
            throw std::runtime_error("Unexpected EOF while reading HDRI header");
        }
        std::streampos pos = file.tellg();
        std::string line = readLine();
        if (line.empty()) {
            break;
        }
    }

    std::string resolution = readLine();
    if (resolution.empty()) {
        throw std::runtime_error("Missing resolution line in HDRI");
    }

    int width = 0;
    int height = 0;
    char axis1 = 0, axis2 = 0;
    char sign1 = 0, sign2 = 0;
    if (sscanf(resolution.c_str(), "%c%c %d %c%c %d", &sign1, &axis1, &height, &sign2, &axis2, &width) != 6) {
        throw std::runtime_error("Failed to parse HDRI resolution string: " + resolution);
    }
    if ((axis1 != 'Y' && axis1 != 'y') || (axis2 != 'X' && axis2 != 'x')) {
        throw std::runtime_error("Only -Y +X orientation is supported, got: " + resolution);
    }
    if (sign1 != '-' || sign2 != '+') {
        throw std::runtime_error("Unsupported HDRI orientation: " + resolution);
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("HDRI has invalid dimensions: " + resolution);
    }

    HDRImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    std::vector<unsigned char> scanline(static_cast<size_t>(width) * 4u);

    for (int y = 0; y < height; ++y) {
        unsigned char scanlineHeader[4];
        if (!file.read(reinterpret_cast<char*>(scanlineHeader), 4)) {
            throw std::runtime_error("Unexpected EOF reading HDRI scanline header");
        }

        bool rle = false;
        if (scanlineHeader[0] == 2 && scanlineHeader[1] == 2) {
            int scanlineWidth = (int(scanlineHeader[2]) << 8) | int(scanlineHeader[3]);
            if (scanlineWidth == width) {
                rle = true;
            }
        }

        if (!rle) {
            scanline[0] = scanlineHeader[0];
            scanline[width] = scanlineHeader[1];
            scanline[2 * width] = scanlineHeader[2];
            scanline[3 * width] = scanlineHeader[3];
            size_t remaining = static_cast<size_t>(width - 1) * 4u;
            if (!file.read(reinterpret_cast<char*>(scanline.data() + 4), static_cast<std::streamsize>(remaining))) {
                throw std::runtime_error("Unexpected EOF reading legacy HDRI scanline");
            }
            for (size_t i = 0; i < remaining / 4u; ++i) {
                scanline[(i + 1) + 0 * width] = scanline[4 + i * 4 + 0];
                scanline[(i + 1) + 1 * width] = scanline[4 + i * 4 + 1];
                scanline[(i + 1) + 2 * width] = scanline[4 + i * 4 + 2];
                scanline[(i + 1) + 3 * width] = scanline[4 + i * 4 + 3];
            }
        } else {
            for (int channel = 0; channel < 4; ++channel) {
                int index = 0;
                while (index < width) {
                    unsigned char code;
                    file.read(reinterpret_cast<char*>(&code), 1);
                    if (!file) {
                        throw std::runtime_error("Unexpected EOF while decoding HDRI RLE");
                    }
                    if (code > 128) {
                        int count = code - 128;
                        unsigned char value;
                        file.read(reinterpret_cast<char*>(&value), 1);
                        if (!file) {
                            throw std::runtime_error("Unexpected EOF in HDRI RLE run");
                        }
                        for (int i = 0; i < count; ++i) {
                            scanline[channel * width + index++] = value;
                        }
                    } else {
                        int count = code;
                        if (!file.read(reinterpret_cast<char*>(scanline.data() + channel * width + index), count)) {
                            throw std::runtime_error("Unexpected EOF in HDRI RLE literal");
                        }
                        index += count;
                    }
                }
            }
        }

#ifdef USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int x = 0; x < width; ++x) {
            unsigned char r = scanline[x + 0 * width];
            unsigned char g = scanline[x + 1 * width];
            unsigned char b = scanline[x + 2 * width];
            unsigned char e = scanline[x + 3 * width];

            float4& dst = image.pixels[static_cast<size_t>(y) * width + x];
            if (e) {
                float f = std::ldexp(1.0f, int(e) - (128 + 8));
                dst.x = r * f;
                dst.y = g * f;
                dst.z = b * f;
                dst.w = 1.0f;
            } else {
                dst.x = dst.y = dst.z = 0.0f;
                dst.w = 1.0f;
            }
        }
    }

    return image;
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

EnvironmentCubemap precomputeEnvironmentCubemap(const std::filesystem::path& filePath,
                                                 unsigned faceSize, unsigned irradianceSize,
                                                 unsigned specularSamples, unsigned diffuseSamples) {
    HDRImage hdr = loadHDRImage(filePath);

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
    result.mipLevels = static_cast<unsigned>(std::floor(std::log2(faceSize))) + 1u;

    cudaExtent cubeExtent = make_cudaExtent(faceSize, faceSize, 6);

    ScopedArray envArray;
    CUDA_CHECK(cudaMalloc3DArray(&envArray.value, &float4Desc, cubeExtent, cudaArrayCubemap | cudaArraySurfaceLoadStore));

    cudaResourceDesc envSurfRes{};
    envSurfRes.resType = cudaResourceTypeArray;
    envSurfRes.res.array.array = envArray.value;

    ScopedSurface envSurface;
    CUDA_CHECK(cudaCreateSurfaceObject(&envSurface.value, &envSurfRes));

    const dim3 block(16, 16, 1);
    const dim3 grid((faceSize + block.x - 1) / block.x,
                    (faceSize + block.y - 1) / block.y,
                    6);

    launchEquirectangularToCubemap(grid, block, hdrTexture.value, envSurface.value, static_cast<int>(faceSize));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    envSurface.reset();
    hdrTexture.reset();
    hdrArray.reset();

    result.envArray = envArray.release();
    result.envTexture = createCubemapTexture(result.envArray);

    ScopedMipmappedArray specularArray;
    CUDA_CHECK(cudaMallocMipmappedArray(&specularArray.value, &float4Desc,
                                        cubeExtent, result.mipLevels,
                                        cudaArrayCubemap | cudaArraySurfaceLoadStore));

    ScopedTexture envTextureForSampling;
    envTextureForSampling.reset(createCubemapTexture(result.envArray));

    for (unsigned level = 0; level < result.mipLevels; ++level) {
        cudaArray_t levelArray = nullptr;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, specularArray.value, level));

        cudaResourceDesc levelRes{};
        levelRes.resType = cudaResourceTypeArray;
        levelRes.res.array.array = levelArray;

        ScopedSurface levelSurface;
        CUDA_CHECK(cudaCreateSurfaceObject(&levelSurface.value, &levelRes));

        unsigned mipFaceSize = std::max(1u, faceSize >> level);
        dim3 mipGrid((mipFaceSize + block.x - 1) / block.x,
                     (mipFaceSize + block.y - 1) / block.y,
                     6);
        float roughness = result.mipLevels > 1 ?
                          static_cast<float>(level) / static_cast<float>(result.mipLevels - 1) :
                          0.0f;

    launchPrefilterSpecularCubemap(mipGrid, block,
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

    result.specularArray = specularArray.release();
    result.specularTexture = createCubemapMipTexture(result.specularArray, result.mipLevels);

    cudaExtent irradianceExtent = make_cudaExtent(irradianceSize, irradianceSize, 6);
    ScopedArray irradianceArray;
    CUDA_CHECK(cudaMalloc3DArray(&irradianceArray.value,
                                 &float4Desc,
                                 irradianceExtent,
                                 cudaArrayCubemap | cudaArraySurfaceLoadStore));

    cudaResourceDesc irradianceRes{};
    irradianceRes.resType = cudaResourceTypeArray;
    irradianceRes.res.array.array = irradianceArray.value;

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

    result.irradianceArray = irradianceArray.release();
    result.irradianceTexture = createCubemapTexture(result.irradianceArray);

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

std::vector<EnvironmentCubemap> loadEnvironmentCubemaps(const std::filesystem::path& directory,
                                                         unsigned faceSize, unsigned irradianceSize,
                                                         unsigned specularSamples, unsigned diffuseSamples) {
    std::vector<std::filesystem::path> paths = collectHDRIFiles(directory);
    std::vector<EnvironmentCubemap> environments;
    environments.reserve(paths.size());

    for (const auto& path : paths) {
        std::cout << "Precomputing cubemap for " << path << "..." << std::endl;
        environments.push_back(precomputeEnvironmentCubemap(path, faceSize, irradianceSize, specularSamples, diffuseSamples));
    }
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