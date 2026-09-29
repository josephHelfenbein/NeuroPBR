#include <env_cache.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <thread>
#include <type_traits>

// File layout (native byte order):
//   char[8]  magic "NPBRENVC"
//   u32      format version (kEnvCacheFormatVersion)
//   u32      endian tag 0x01020304
//   u32      bytes per texel (16)
//   key:     u32 cacheVersion, u64 sourceSize, i64 sourceMtime, u32 faceSize, irradianceSize,
//            specularSamples, diffuseSamples, hdrMaxWidth, u32 sourceName length
//   meta:    u32 mipLevels, i32 sourceWidth, sourceHeight, f32 horizonBrightness,
//            zenithBrightness, hardness
//   char[]   sourceName
//   float[]  env cubemap, specular mips 1..mipLevels-1, irradiance cubemap

namespace {

constexpr char kMagic[8] = {'N', 'P', 'B', 'R', 'E', 'N', 'V', 'C'};
// Bump when the file layout changes
constexpr uint32_t kEnvCacheFormatVersion = 1;
constexpr uint32_t kEndianTag = 0x01020304u;
constexpr uint32_t kTexelBytes = 4 * sizeof(float);
constexpr size_t kFixedHeaderBytes = 88;
// CUDA's cubemap size limit
constexpr uint32_t kMaxFaceSize = 32768;

static_assert(sizeof(float) == 4, "cache files store 32-bit floats");

class ByteWriter {
public:
    template <typename T>
    void put(T value) {
        static_assert(std::is_trivially_copyable<T>::value, "raw copy");
        const size_t offset = bytes.size();
        bytes.resize(offset + sizeof(T));
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }
    void putBytes(const void* data, size_t size) {
        const size_t offset = bytes.size();
        bytes.resize(offset + size);
        if (size > 0) {
            std::memcpy(bytes.data() + offset, data, size);
        }
    }
    std::vector<uint8_t> bytes;
};

class ByteReader {
public:
    explicit ByteReader(const uint8_t* data) : data(data) {}
    template <typename T>
    T get() {
        T value;
        std::memcpy(&value, data + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }
    const uint8_t* data;
    size_t offset = 0;
};

void putKeyFields(ByteWriter& w, const EnvCacheKey& key) {
    w.put<uint32_t>(key.cacheVersion);
    w.put<uint64_t>(key.sourceSize);
    w.put<int64_t>(key.sourceMtime);
    w.put<uint32_t>(key.faceSize);
    w.put<uint32_t>(key.irradianceSize);
    w.put<uint32_t>(key.specularSamples);
    w.put<uint32_t>(key.diffuseSamples);
    w.put<uint32_t>(key.hdrMaxWidth);
    w.put<uint32_t>(static_cast<uint32_t>(key.sourceName.size()));
}

std::vector<uint8_t> serializeHeader(const EnvCacheKey& key, const EnvCacheMeta& meta) {
    ByteWriter w;
    w.putBytes(kMagic, sizeof(kMagic));
    w.put<uint32_t>(kEnvCacheFormatVersion);
    w.put<uint32_t>(kEndianTag);
    w.put<uint32_t>(kTexelBytes);
    putKeyFields(w, key);
    w.put<uint32_t>(meta.mipLevels);
    w.put<int32_t>(meta.sourceWidth);
    w.put<int32_t>(meta.sourceHeight);
    w.put<float>(meta.horizonBrightness);
    w.put<float>(meta.zenithBrightness);
    w.put<float>(meta.hardness);
    w.putBytes(key.sourceName.data(), key.sourceName.size());
    return std::move(w.bytes);
}

uint64_t hashKey(const EnvCacheKey& key) {
    ByteWriter w;
    putKeyFields(w, key);
    w.putBytes(key.sourceName.data(), key.sourceName.size());
    uint64_t hash = 14695981039346656037ull; // FNV-1a 64
    for (uint8_t b : w.bytes) {
        hash ^= b;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool expectedBlockFloats(const EnvCacheKey& key, std::vector<size_t>& blocks) {
    if (key.faceSize == 0 || key.faceSize > kMaxFaceSize ||
        key.irradianceSize == 0 || key.irradianceSize > kMaxFaceSize) {
        return false;
    }
    blocks.clear();
    blocks.push_back(envCubemapFloatCount(key.faceSize));
    const unsigned mipLevels = envMipLevelCount(key.faceSize);
    for (unsigned level = 1; level < mipLevels; ++level) {
        blocks.push_back(envCubemapFloatCount(envMipFaceSize(key.faceSize, level)));
    }
    blocks.push_back(envCubemapFloatCount(key.irradianceSize));
    return true;
}

std::string hex64(uint64_t value) {
    static const char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<size_t>(i)] = digits[value & 0xF];
        value >>= 4;
    }
    return out;
}

template <typename T>
bool keyFieldMatches(const char* name, T stored, T expected, std::string& reason) {
    if (stored == expected) {
        return true;
    }
    reason = std::string("key mismatch: ") + name + " is " + std::to_string(stored) +
             ", expected " + std::to_string(expected);
    return false;
}

std::string uniqueSuffix() {
    std::random_device device;
    uint64_t value = (static_cast<uint64_t>(device()) << 32) ^ device();
    value ^= static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    value ^= static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id())) * 0x9E3779B97F4A7C15ull;
    return hex64(value);
}

} // namespace

unsigned envMipLevelCount(unsigned faceSize) {
    unsigned levels = 1;
    while (faceSize > 1) {
        faceSize >>= 1;
        ++levels;
    }
    return levels;
}

unsigned envMipFaceSize(unsigned faceSize, unsigned level) {
    if (level >= 32) {
        return 1;
    }
    unsigned size = faceSize >> level;
    return size > 0 ? size : 1u;
}

size_t envCubemapFloatCount(unsigned faceDim) {
    const size_t dim = static_cast<size_t>(faceDim);
    return dim * dim * 6u * 4u;
}

bool makeEnvCacheKey(const std::filesystem::path& hdrPath,
                     unsigned faceSize, unsigned irradianceSize,
                     unsigned specularSamples, unsigned diffuseSamples,
                     unsigned hdrMaxWidth,
                     EnvCacheKey& key, std::string& error) noexcept {
    try {
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(hdrPath, ec);
        if (ec) {
            error = "cannot read size of " + hdrPath.string() + ": " + ec.message();
            return false;
        }
        const auto mtime = std::filesystem::last_write_time(hdrPath, ec);
        if (ec) {
            error = "cannot read modification time of " + hdrPath.string() + ": " + ec.message();
            return false;
        }
        key = EnvCacheKey{};
        key.sourceName = hdrPath.filename().string();
        key.sourceSize = static_cast<uint64_t>(size);
        key.sourceMtime = static_cast<int64_t>(mtime.time_since_epoch().count());
        key.faceSize = faceSize;
        key.irradianceSize = irradianceSize;
        key.specularSamples = specularSamples;
        key.diffuseSamples = diffuseSamples;
        key.hdrMaxWidth = hdrMaxWidth;
        key.cacheVersion = kEnvCacheVersion;
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    } catch (...) {
        error = "unknown error";
        return false;
    }
}

std::filesystem::path envCacheFilePath(const std::filesystem::path& cacheDir, const EnvCacheKey& key) {
    const std::string stem = std::filesystem::path(key.sourceName).stem().string();
    return cacheDir / (stem + "_" + hex64(hashKey(key)) + ".envcache");
}

EnvCacheReadStatus readEnvCache(const std::filesystem::path& file, const EnvCacheKey& expected,
                                EnvCacheData& data, std::string& reason) noexcept {
    try {
        std::error_code ec;
        const auto status = std::filesystem::status(file, ec);
        if (status.type() == std::filesystem::file_type::not_found) {
            reason = "no cache file";
            return EnvCacheReadStatus::Missing;
        }
        if (ec) {
            reason = "cannot stat: " + ec.message();
            return EnvCacheReadStatus::Rejected;
        }
        if (status.type() != std::filesystem::file_type::regular) {
            reason = "not a regular file";
            return EnvCacheReadStatus::Rejected;
        }
        const uintmax_t fileSize = std::filesystem::file_size(file, ec);
        if (ec) {
            reason = "cannot read size: " + ec.message();
            return EnvCacheReadStatus::Rejected;
        }
        if (fileSize < kFixedHeaderBytes) {
            reason = "truncated header (" + std::to_string(fileSize) + " bytes)";
            return EnvCacheReadStatus::Rejected;
        }

        std::ifstream in(file, std::ios::binary);
        if (!in) {
            reason = "cannot open for reading";
            return EnvCacheReadStatus::Rejected;
        }

        uint8_t header[kFixedHeaderBytes];
        if (!in.read(reinterpret_cast<char*>(header), sizeof(header))) {
            reason = "short read in header";
            return EnvCacheReadStatus::Rejected;
        }
        if (std::memcmp(header, kMagic, sizeof(kMagic)) != 0) {
            reason = "bad magic (not an env cache file)";
            return EnvCacheReadStatus::Rejected;
        }
        ByteReader r(header);
        r.offset = sizeof(kMagic);
        const uint32_t formatVersion = r.get<uint32_t>();
        if (formatVersion != kEnvCacheFormatVersion) {
            reason = "format version " + std::to_string(formatVersion) + ", expected " +
                     std::to_string(kEnvCacheFormatVersion);
            return EnvCacheReadStatus::Rejected;
        }
        const uint32_t endianTag = r.get<uint32_t>();
        const uint32_t texelBytes = r.get<uint32_t>();
        if (endianTag != kEndianTag || texelBytes != kTexelBytes) {
            reason = "written on a platform with a different byte order or float size";
            return EnvCacheReadStatus::Rejected;
        }

        EnvCacheKey stored;
        stored.cacheVersion = r.get<uint32_t>();
        stored.sourceSize = r.get<uint64_t>();
        stored.sourceMtime = r.get<int64_t>();
        stored.faceSize = r.get<uint32_t>();
        stored.irradianceSize = r.get<uint32_t>();
        stored.specularSamples = r.get<uint32_t>();
        stored.diffuseSamples = r.get<uint32_t>();
        stored.hdrMaxWidth = r.get<uint32_t>();
        const uint32_t nameLength = r.get<uint32_t>();

        EnvCacheMeta meta;
        meta.mipLevels = r.get<uint32_t>();
        meta.sourceWidth = r.get<int32_t>();
        meta.sourceHeight = r.get<int32_t>();
        meta.horizonBrightness = r.get<float>();
        meta.zenithBrightness = r.get<float>();
        meta.hardness = r.get<float>();

        if (!keyFieldMatches("cache version", stored.cacheVersion, expected.cacheVersion, reason) ||
            !keyFieldMatches("source file size", stored.sourceSize, expected.sourceSize, reason) ||
            !keyFieldMatches("source mtime", stored.sourceMtime, expected.sourceMtime, reason) ||
            !keyFieldMatches("faceSize", stored.faceSize, expected.faceSize, reason) ||
            !keyFieldMatches("irradianceSize", stored.irradianceSize, expected.irradianceSize, reason) ||
            !keyFieldMatches("specularSamples", stored.specularSamples, expected.specularSamples, reason) ||
            !keyFieldMatches("diffuseSamples", stored.diffuseSamples, expected.diffuseSamples, reason) ||
            !keyFieldMatches("hdrMaxWidth", stored.hdrMaxWidth, expected.hdrMaxWidth, reason) ||
            !keyFieldMatches("source name length", static_cast<uint64_t>(nameLength),
                             static_cast<uint64_t>(expected.sourceName.size()), reason)) {
            return EnvCacheReadStatus::Rejected;
        }
        const uint32_t expectedMipLevels = envMipLevelCount(expected.faceSize);
        if (!keyFieldMatches("mipLevels", meta.mipLevels, expectedMipLevels, reason)) {
            return EnvCacheReadStatus::Rejected;
        }

        std::vector<size_t> blocks;
        if (!expectedBlockFloats(expected, blocks)) {
            reason = "unsupported cubemap sizes in key";
            return EnvCacheReadStatus::Rejected;
        }
        uintmax_t expectedSize = kFixedHeaderBytes + nameLength;
        for (size_t floats : blocks) {
            expectedSize += static_cast<uintmax_t>(floats) * sizeof(float);
        }
        if (fileSize != expectedSize) {
            reason = "file is " + std::to_string(fileSize) + " bytes, expected " + std::to_string(expectedSize);
            return EnvCacheReadStatus::Rejected;
        }

        std::string name(nameLength, '\0');
        if (nameLength > 0 && !in.read(&name[0], nameLength)) {
            reason = "short read in source name";
            return EnvCacheReadStatus::Rejected;
        }
        if (name != expected.sourceName) {
            reason = "key mismatch: source name is \"" + name + "\", expected \"" + expected.sourceName + "\"";
            return EnvCacheReadStatus::Rejected;
        }

        auto readBlock = [&](std::vector<float>& dst, size_t floats) {
            dst.resize(floats);
            return static_cast<bool>(in.read(reinterpret_cast<char*>(dst.data()),
                                             static_cast<std::streamsize>(floats * sizeof(float))));
        };

        data.meta = meta;
        data.specular.assign(meta.mipLevels - 1, std::vector<float>());
        bool ok = readBlock(data.env, blocks.front());
        for (size_t i = 0; ok && i < data.specular.size(); ++i) {
            ok = readBlock(data.specular[i], blocks[i + 1]);
        }
        ok = ok && readBlock(data.irradiance, blocks.back());
        if (!ok) {
            reason = "short read in texel data";
            return EnvCacheReadStatus::Rejected;
        }
        return EnvCacheReadStatus::Hit;
    } catch (const std::exception& e) {
        reason = std::string("read failed: ") + e.what();
        return EnvCacheReadStatus::Rejected;
    } catch (...) {
        reason = "read failed: unknown error";
        return EnvCacheReadStatus::Rejected;
    }
}

bool writeEnvCache(const std::filesystem::path& file, const EnvCacheKey& key,
                   const EnvCacheData& data, std::string& error) noexcept {
    std::filesystem::path tmpPath;
    try {
        std::vector<size_t> blocks;
        if (!expectedBlockFloats(key, blocks) ||
            data.meta.mipLevels != envMipLevelCount(key.faceSize) ||
            data.specular.size() + 2 != blocks.size() ||
            data.env.size() != blocks.front() ||
            data.irradiance.size() != blocks.back()) {
            error = "cubemap data does not match the cache key's sizes";
            return false;
        }
        for (size_t i = 0; i < data.specular.size(); ++i) {
            if (data.specular[i].size() != blocks[i + 1]) {
                error = "specular mip " + std::to_string(i + 1) + " has the wrong size";
                return false;
            }
        }

        std::error_code ec;
        const std::filesystem::path dir = file.parent_path();
        if (!dir.empty()) {
            std::filesystem::create_directories(dir, ec);
            if (ec) {
                error = "cannot create " + dir.string() + ": " + ec.message();
                return false;
            }
        }

        const std::vector<uint8_t> header = serializeHeader(key, data.meta);
        if (header.size() != kFixedHeaderBytes + key.sourceName.size()) {
            error = "internal error: unexpected header size";
            return false;
        }

        const std::string suffix = ".tmp-" + uniqueSuffix();
        tmpPath = file;
        tmpPath += suffix;
        {
            std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
            if (!out) {
                error = "cannot create " + tmpPath.string();
                return false;
            }
            auto writeBlock = [&](const std::vector<float>& block) {
                out.write(reinterpret_cast<const char*>(block.data()),
                          static_cast<std::streamsize>(block.size() * sizeof(float)));
            };
            out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
            writeBlock(data.env);
            for (const auto& level : data.specular) {
                writeBlock(level);
            }
            writeBlock(data.irradiance);
            out.close();
            if (out.fail()) {
                error = "failed writing " + tmpPath.string();
                std::filesystem::remove(tmpPath, ec);
                return false;
            }
        }

        std::filesystem::rename(tmpPath, file, ec);
        if (ec) {
            error = "cannot rename " + tmpPath.string() + " to " + file.string() + ": " + ec.message();
            std::filesystem::remove(tmpPath, ec);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        error = std::string("write failed: ") + e.what();
    } catch (...) {
        error = "write failed: unknown error";
    }
    if (!tmpPath.empty()) {
        std::error_code ec;
        std::filesystem::remove(tmpPath, ec);
    }
    return false;
}
