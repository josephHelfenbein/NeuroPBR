#pragma once

#include <material_cache.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

// Per-pixel sizes of the GPUMemorySlot buffers
constexpr size_t kSlotAlbedoBytesPerPixel = 4 * sizeof(float);
constexpr size_t kSlotNormalBytesPerPixel = 4 * sizeof(float);
constexpr size_t kSlotRoughnessBytesPerPixel = sizeof(float);
constexpr size_t kSlotMetallicBytesPerPixel = sizeof(float);
constexpr size_t kSlotFrameBytesPerPixel = 3;
constexpr size_t kSlotStagingBytesPerPixel = MaterialMaps::kBytesPerPixel;

constexpr size_t kViewsPerSample = 3;
constexpr size_t kOutputBytesPerPixel = 3;

// Pinned host buffers in each slot
constexpr size_t kPinnedStagingBytesPerPixel = MaterialMaps::kBytesPerPixel;
constexpr size_t kPinnedFramesBytesPerPixel = kViewsPerSample * kOutputBytesPerPixel;

constexpr int kMaxPipelineDepth = 8; // default --pipeline-depth
constexpr int kPipelineDepthLimit = 64;
constexpr int kMaxResLimit = 16384;

constexpr size_t slotBytesPerPixel() {
    return kSlotAlbedoBytesPerPixel + kSlotNormalBytesPerPixel +
           kSlotRoughnessBytesPerPixel + kSlotMetallicBytesPerPixel +
           kSlotFrameBytesPerPixel + kSlotStagingBytesPerPixel;
}

// Slot buffers are flat, so capacity is a pixel count rather than a width and height
constexpr size_t kDefaultMaxSlotPixels = 2048ULL * 2048ULL;

inline size_t pixelCount(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    return static_cast<size_t>(width) * static_cast<size_t>(height);
}

inline size_t slotBytes(size_t slotPixels) {
    return slotPixels * slotBytesPerPixel();
}

inline size_t slotPinnedBytes(size_t slotPixels) {
    return slotPixels * (kPinnedStagingBytesPerPixel + kPinnedFramesBytesPerPixel);
}

// Decode buffer for cache misses; one per loader, not per slot
inline size_t loaderDecodeBytes(size_t slotPixels) {
    return slotPixels * MaterialMaps::kBytesPerPixel;
}

inline bool materialFitsSlot(int width, int height, size_t slotPixels) {
    const size_t pixels = pixelCount(width, height);
    return pixels > 0 && pixels <= slotPixels;
}

enum class DepthLimit { Cpu, Gpu, PipelineDepth };

inline const char* depthLimitName(DepthLimit limit) {
    switch (limit) {
        case DepthLimit::Cpu: return "host RAM";
        case DepthLimit::Gpu: return "VRAM";
        case DepthLimit::PipelineDepth: return "pipeline depth cap";
    }
    return "unknown";
}

struct PipelineDepthChoice {
    size_t cpuItems = 0;
    size_t gpuItems = 0;
    int depth = 1;
    DepthLimit limit = DepthLimit::PipelineDepth;
    bool belowMinimum = false; // < 2 slots, so stages can't overlap
};

inline PipelineDepthChoice choosePipelineDepth(size_t availableRam, size_t cpuItemBytes,
                                               size_t availableVram, size_t slotBytesEach,
                                               int maxDepth) {
    PipelineDepthChoice choice;
    maxDepth = (std::max)(maxDepth, 1);
    choice.cpuItems = cpuItemBytes > 0 ? availableRam / cpuItemBytes : static_cast<size_t>(maxDepth);
    choice.gpuItems = slotBytesEach > 0 ? availableVram / slotBytesEach : static_cast<size_t>(maxDepth);

    const size_t memoryItems = (std::min)(choice.cpuItems, choice.gpuItems);
    choice.belowMinimum = memoryItems < 2;

    if (static_cast<size_t>(maxDepth) <= memoryItems) {
        choice.depth = maxDepth;
        choice.limit = DepthLimit::PipelineDepth;
    } else {
        choice.depth = (std::max)(static_cast<int>(memoryItems), 1);
        choice.limit = choice.gpuItems <= choice.cpuItems ? DepthLimit::Gpu : DepthLimit::Cpu;
    }
    return choice;
}

inline bool parseIntInRange(const std::string& value, int maxValue, int& out) {
    size_t parsedChars = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &parsedChars);
    } catch (const std::exception&) {
        return false;
    }
    if (parsedChars != value.size() || parsed < 1 || parsed > maxValue) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}
