#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& what)
        : std::runtime_error(what) {}
};

void cudaCheck(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

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

    EnvironmentCubemap(EnvironmentCubemap&& other) noexcept {
        moveFrom(std::move(other));
    }

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

    BRDFLookupTable(BRDFLookupTable&& other) noexcept {
        moveFrom(std::move(other));
    }

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

std::vector<std::filesystem::path> collectHDRIFiles(const std::filesystem::path& root);

std::vector<EnvironmentCubemap> loadEnvironmentCubemaps(const std::filesystem::path& directory,
                                                        unsigned faceSize, unsigned irradianceSize,
                                                        unsigned specularSamples, unsigned diffuseSamples);

BRDFLookupTable createBRDFLUT(unsigned size);

void loadBRDFLUT(BRDFLookupTable& lut);

void renderPlane(const EnvironmentCubemap& env, const BRDFLookupTable& brdf,
                 const float* albedo, const float* normal,
                 const float* roughness, const float* metallic,
                 int width, int height, std::vector<float4>& frameRGBA);

EnvironmentCubemap precomputeEnvironmentCubemap(const std::filesystem::path& filePath, 
                                                 unsigned faceSize, unsigned irradianceSize,
                                                 unsigned specularSamples, unsigned diffuseSamples);