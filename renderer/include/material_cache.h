#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

struct MaterialMaps {
    static constexpr size_t kBytesPerPixel = 4 + 4 + 1 + 1;

    int width = 0;
    int height = 0;
    std::vector<uint8_t> albedo;    // RGBA8
    std::vector<uint8_t> normal;    // RGBA8
    std::vector<uint8_t> roughness; // R8
    std::vector<uint8_t> metallic;  // R8

    size_t pixelCount() const { return static_cast<size_t>(width) * static_cast<size_t>(height); }
    size_t byteSize() const { return albedo.size() + normal.size() + roughness.size() + metallic.size(); }
};

MaterialMaps loadMaterialMaps(const std::filesystem::path& materialDir);

// No eviction: picks are uniform, so any full cache is as good as another. Not thread-safe.
class MaterialCache {
public:
    explicit MaterialCache(size_t budgetBytes) : budgetBytes_(budgetBytes) {}

    const MaterialMaps* find(const std::string& name);

    // nullptr if over budget; maps is left untouched
    const MaterialMaps* tryInsert(const std::string& name, MaterialMaps& maps);

    size_t budgetBytes() const { return budgetBytes_; }
    size_t residentBytes() const { return residentBytes_; }
    size_t size() const { return entries_.size(); }
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }

private:
    std::unordered_map<std::string, MaterialMaps> entries_;
    size_t budgetBytes_ = 0;
    size_t residentBytes_ = 0;
    size_t hits_ = 0;
    size_t misses_ = 0;
};
