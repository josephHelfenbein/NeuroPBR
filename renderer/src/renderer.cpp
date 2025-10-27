#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "prefilter.cuh"
#include "brdf.cuh"

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

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what) : std::runtime_error(what) {}
};

inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::string message = std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                              " for `" + expr + "`: " + cudaGetErrorString(err);
        throw CudaError(message);
    }
}

#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

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

struct EnvironmentCubemap {
    std::string name;
    unsigned faceSize = 0;
    unsigned irradianceSize = 0;
    unsigned mipLevels = 0;

    cudaArray_t envArray = nullptr;
    cudaMipmappedArray_t specularArray = nullptr;
    cudaArray_t irradianceArray = nullptr;

    cudaTextureObject_t envTexture = 0;
    cudaTextureObject_t specularTexture = 0;
    cudaTextureObject_t irradianceTexture = 0;

    EnvironmentCubemap() = default;
    ~EnvironmentCubemap() { release(); }

    EnvironmentCubemap(const EnvironmentCubemap&) = delete;
    EnvironmentCubemap& operator=(const EnvironmentCubemap&) = delete;

    EnvironmentCubemap(EnvironmentCubemap&& other) noexcept { moveFrom(std::move(other)); }
    EnvironmentCubemap& operator=(EnvironmentCubemap&& other) noexcept {
        if (this != &other) {
            release();
            moveFrom(std::move(other));
        }
        return *this;
    }

private:
    void release() noexcept {
        if (envTexture) cudaDestroyTextureObject(envTexture);
        if (specularTexture) cudaDestroyTextureObject(specularTexture);
        if (irradianceTexture) cudaDestroyTextureObject(irradianceTexture);
        if (envArray) cudaFreeArray(envArray);
        if (specularArray) cudaFreeMipmappedArray(specularArray);
        if (irradianceArray) cudaFreeArray(irradianceArray);
        envTexture = specularTexture = irradianceTexture = 0;
        envArray = nullptr;
        specularArray = nullptr;
        irradianceArray = nullptr;
    }

    void moveFrom(EnvironmentCubemap&& other) noexcept {
        name = std::move(other.name);
        faceSize = other.faceSize;
        irradianceSize = other.irradianceSize;
        mipLevels = other.mipLevels;

        envArray = other.envArray;
        specularArray = other.specularArray;
        irradianceArray = other.irradianceArray;

        envTexture = other.envTexture;
        specularTexture = other.specularTexture;
        irradianceTexture = other.irradianceTexture;

        other.envArray = nullptr;
        other.specularArray = nullptr;
        other.irradianceArray = nullptr;
        other.envTexture = 0;
        other.specularTexture = 0;
        other.irradianceTexture = 0;
        other.faceSize = 0;
        other.irradianceSize = 0;
        other.mipLevels = 0;
    }
};

struct BRDFLookupTable {
    unsigned size = 0;
    cudaArray_t array = nullptr;
    cudaSurfaceObject_t surface = 0;
    cudaTextureObject_t texture = 0;

    BRDFLookupTable() = default;
    ~BRDFLookupTable() { release(); }

    BRDFLookupTable(const BRDFLookupTable&) = delete;
    BRDFLookupTable& operator=(const BRDFLookupTable&) = delete;

    BRDFLookupTable(BRDFLookupTable&& other) noexcept { moveFrom(std::move(other)); }
    BRDFLookupTable& operator=(BRDFLookupTable&& other) noexcept {
        if (this != &other) {
            release();
            moveFrom(std::move(other));
        }
        return *this;
    }

private:
    void release() noexcept {
        if (texture) cudaDestroyTextureObject(texture);
        if (surface) cudaDestroySurfaceObject(surface);
        if (array) cudaFreeArray(array);
        texture = 0;
        surface = 0;
        array = nullptr;
        size = 0;
    }

    void moveFrom(BRDFLookupTable&& other) noexcept {
        size = other.size;
        array = other.array;
        surface = other.surface;
        texture = other.texture;

        other.size = 0;
        other.array = nullptr;
        other.surface = 0;
        other.texture = 0;
    }
};

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
                                                 unsigned faceSize,
                                                 unsigned irradianceSize,
                                                 unsigned specularSamples,
                                                 unsigned diffuseSamples) {
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

    equirectangularToCubemap<<<grid, block>>>(hdrTexture.value, envSurface.value, static_cast<int>(faceSize));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    envSurface.reset();
    hdrTexture.reset();
    hdrArray.reset();

    result.envArray = envArray.release();
    result.envTexture = createCubemapTexture(result.envArray);

    ScopedMipmappedArray specularArray;
    CUDA_CHECK(cudaMallocMipmappedArray(&specularArray.value,
                                        &float4Desc,
                                        cubeExtent,
                                        result.mipLevels,
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

        prefilterSpecularCubemap<<<mipGrid, block>>>(envTextureForSampling.value,
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

    convolveDiffuseIrradiance<<<irradianceGrid, block>>>(result.envTexture,
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
                                                         unsigned faceSize,
                                                         unsigned irradianceSize,
                                                         unsigned specularSamples,
                                                         unsigned diffuseSamples) {
    std::vector<std::filesystem::path> paths = collectHDRIFiles(directory);
    std::vector<EnvironmentCubemap> environments;
    environments.reserve(paths.size());

    for (const auto& path : paths) {
        std::cout << "Precomputing cubemap for " << path << "..." << std::endl;
        environments.push_back(precomputeEnvironmentCubemap(path,
                                                             faceSize,
                                                             irradianceSize,
                                                             specularSamples,
                                                             diffuseSamples));
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

    precomputeBRDF<<<grid, block>>>(lut.surface, static_cast<int>(lut.size), static_cast<int>(lut.size));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}