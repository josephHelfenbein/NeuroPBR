#include <material_cache.h>

#include <io.h>

#include <stdexcept>
#include <utility>

MaterialMaps loadMaterialMaps(const std::filesystem::path& materialDir) {
    ByteImage albedo = loadPNGImage8(materialDir / "albedo.png", 4, true);
    ByteImage normal = loadPNGImage8(materialDir / "normal.png", 4, true);
    ByteImage roughness = loadPNGImage8(materialDir / "roughness.png", 1, true);
    ByteImage metallic = loadPNGImage8(materialDir / "metallic.png", 1, true);

    auto sameSize = [&](const ByteImage& img) {
        return img.width == albedo.width && img.height == albedo.height;
    };
    if (!sameSize(normal) || !sameSize(roughness) || !sameSize(metallic)) {
        throw std::runtime_error("Material maps in " + materialDir.string() + " have mismatched sizes (albedo " +
                                 std::to_string(albedo.width) + "x" + std::to_string(albedo.height) +
                                 ", normal " + std::to_string(normal.width) + "x" + std::to_string(normal.height) +
                                 ", roughness " + std::to_string(roughness.width) + "x" + std::to_string(roughness.height) +
                                 ", metallic " + std::to_string(metallic.width) + "x" + std::to_string(metallic.height) + ")");
    }

    MaterialMaps maps;
    maps.width = albedo.width;
    maps.height = albedo.height;
    maps.albedo = std::move(albedo.data);
    maps.normal = std::move(normal.data);
    maps.roughness = std::move(roughness.data);
    maps.metallic = std::move(metallic.data);
    return maps;
}

const MaterialMaps* MaterialCache::find(const std::string& name) {
    auto it = entries_.find(name);
    if (it == entries_.end()) {
        ++misses_;
        return nullptr;
    }
    ++hits_;
    return &it->second;
}

const MaterialMaps* MaterialCache::tryInsert(const std::string& name, MaterialMaps& maps) {
    const size_t bytes = maps.byteSize();
    if (bytes == 0 || bytes > budgetBytes_ - residentBytes_ || entries_.count(name) != 0) {
        return nullptr;
    }
    auto inserted = entries_.emplace(name, std::move(maps));
    residentBytes_ += bytes;
    return &inserted.first->second;
}
