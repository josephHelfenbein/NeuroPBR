#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Bump if the HDR decode, downscale or precompute changes
constexpr uint32_t kEnvCacheVersion = 1;

struct EnvCacheKey {
    std::string sourceName;
    uint64_t sourceSize = 0;
    int64_t sourceMtime = 0;
    uint32_t faceSize = 0;
    uint32_t irradianceSize = 0;
    uint32_t specularSamples = 0;
    uint32_t diffuseSamples = 0;
    uint32_t hdrMaxWidth = 0;
    uint32_t cacheVersion = kEnvCacheVersion;
};

struct EnvCacheMeta {
    uint32_t mipLevels = 0;
    int32_t sourceWidth = 0;
    int32_t sourceHeight = 0;
    float horizonBrightness = 0.0f;
    float zenithBrightness = 0.0f;
    float hardness = 0.0f;
};

struct EnvCacheData {
    EnvCacheMeta meta;
    std::vector<float> env;
    std::vector<std::vector<float>> specular; // mips 1..n (mip 0 equals env)
    std::vector<float> irradiance;
};

enum class EnvCacheReadStatus {
    Hit,
    Missing,
    Rejected,
};

unsigned envMipLevelCount(unsigned faceSize);
unsigned envMipFaceSize(unsigned faceSize, unsigned level);
size_t envCubemapFloatCount(unsigned faceDim);

bool makeEnvCacheKey(const std::filesystem::path& hdrPath,
                     unsigned faceSize, unsigned irradianceSize,
                     unsigned specularSamples, unsigned diffuseSamples,
                     unsigned hdrMaxWidth,
                     EnvCacheKey& key, std::string& error) noexcept;

std::filesystem::path envCacheFilePath(const std::filesystem::path& cacheDir, const EnvCacheKey& key);

// Only a hit if every key field and the file size match
EnvCacheReadStatus readEnvCache(const std::filesystem::path& file, const EnvCacheKey& expected,
                                EnvCacheData& data, std::string& reason) noexcept;

bool writeEnvCache(const std::filesystem::path& file, const EnvCacheKey& key,
                   const EnvCacheData& data, std::string& error) noexcept;
