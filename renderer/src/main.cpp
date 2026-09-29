#include <renderer.h>
#include <hdr_image.h>
#include <io.h>
#include <material_cache.h>
#include <pipeline_sizing.h>
#include <unpack.cuh>
#include <fpng/src/fpng.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <array>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t maxSize) : maxSize(maxSize), done(false), aborted(false) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex);
        notFull.wait(lock, [this] { return queue.size() < maxSize || aborted; });
        if (aborted) return false;
        queue.push(std::move(item));
        notEmpty.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        notEmpty.wait(lock, [this] { return !queue.empty() || done || aborted; });
        if (aborted || (queue.empty() && done)) return false;
        item = std::move(queue.front());
        queue.pop();
        notFull.notify_one();
        return true;
    }

    void setDone() {
        std::unique_lock<std::mutex> lock(mutex);
        done = true;
        notEmpty.notify_all();
    }

    // Wakes every waiting push and pop, which then fail
    void abort() {
        std::unique_lock<std::mutex> lock(mutex);
        aborted = true;
        notEmpty.notify_all();
        notFull.notify_all();
    }

private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable notEmpty;
    std::condition_variable notFull;
    size_t maxSize;
    bool done;
    bool aborted;
};

// Owned by one pipeline stage at a time, so no locking is needed
struct GPUMemorySlot {
    float4* dAlbedo = nullptr;
    float4* dNormal = nullptr;
    float* dRoughness = nullptr;
    float* dMetallic = nullptr;
    uint8_t* dFrame = nullptr;
    uint8_t* dMaterialStaging = nullptr;
    uint8_t* hMaterialStaging = nullptr; // pinned
    uint8_t* hFrames = nullptr; // pinned
};

static_assert(sizeof(float4) == kSlotAlbedoBytesPerPixel, "slot sizing assumes float4 albedo");
static_assert(sizeof(float4) == kSlotNormalBytesPerPixel, "slot sizing assumes float4 normal");
static_assert(sizeof(float) == kSlotRoughnessBytesPerPixel, "slot sizing assumes float roughness");
static_assert(sizeof(float) == kSlotMetallicBytesPerPixel, "slot sizing assumes float metallic");

GPUMemorySlot allocateGPUSlot(size_t pixelCount) {
    GPUMemorySlot slot;
    CUDA_CHECK(cudaMalloc(&slot.dAlbedo, pixelCount * kSlotAlbedoBytesPerPixel));
    CUDA_CHECK(cudaMalloc(&slot.dNormal, pixelCount * kSlotNormalBytesPerPixel));
    CUDA_CHECK(cudaMalloc(&slot.dRoughness, pixelCount * kSlotRoughnessBytesPerPixel));
    CUDA_CHECK(cudaMalloc(&slot.dMetallic, pixelCount * kSlotMetallicBytesPerPixel));
    CUDA_CHECK(cudaMalloc(&slot.dFrame, pixelCount * kSlotFrameBytesPerPixel));
    CUDA_CHECK(cudaMalloc(&slot.dMaterialStaging, pixelCount * kSlotStagingBytesPerPixel));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&slot.hMaterialStaging),
                             pixelCount * kPinnedStagingBytesPerPixel, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&slot.hFrames),
                             pixelCount * kPinnedFramesBytesPerPixel, cudaHostAllocDefault));
    return slot;
}

void freeGPUSlot(const GPUMemorySlot& slot) {
    cudaFree(slot.dAlbedo);
    cudaFree(slot.dNormal);
    cudaFree(slot.dRoughness);
    cudaFree(slot.dMetallic);
    cudaFree(slot.dFrame);
    cudaFree(slot.dMaterialStaging);
    cudaFreeHost(slot.hMaterialStaging);
    cudaFreeHost(slot.hFrames);
}

uint8_t* slotHostFrame(const GPUMemorySlot& slot, size_t view, int width, int height) {
    const size_t frameBytes = pixelCount(width, height) * kOutputBytesPerPixel;
    return slot.hFrames + view * frameBytes;
}

struct RenderRequest {
    size_t environmentIndex;
    GPUMemorySlot gpuSlot;
    std::filesystem::path targetDir;
    std::string textureName;
    bool dirtySet;
    bool enableShadows;
    bool enableCameraArtifacts;
    unsigned long long artifactSeed;
    int width;
    int height;
};

struct RenderResult {
    // Point into gpuSlot.hFrames
    std::array<const uint8_t*, kViewsPerSample> frames{};
    std::filesystem::path targetDir;
    std::string textureName;
    int width;
    int height;
    GPUMemorySlot gpuSlot;
};

struct PipelineQueues {
    ThreadSafeQueue<RenderRequest> load;
    ThreadSafeQueue<RenderResult> write;
    ThreadSafeQueue<GPUMemorySlot> freeSlots;

    explicit PipelineQueues(size_t depth) : load(depth), write(depth), freeSlots(depth) {}

    // Called when a stage dies, so the other stages don't wait on it forever
    void abort() {
        load.abort();
        write.abort();
        freeSlots.abort();
    }
};

constexpr size_t kCubemapFaces = 6ULL;
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;

std::string formatMB(size_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f MB", static_cast<double>(bytes) / static_cast<double>(kBytesPerMB));
    return buffer;
}

size_t cubemapFaceBytes(unsigned size) {
    if (size == 0) {
        return 0;
    }
    size_t dim = static_cast<size_t>(size);
    return dim * dim * kCubemapFaces * sizeof(float4);
}

size_t specularMipBytes(const EnvironmentCubemap& env) {
    size_t total = 0;
    for (unsigned level = 0; level < env.mipLevels; ++level) {
        unsigned faceDim = std::max(1u, env.faceSize >> level);
        total += cubemapFaceBytes(faceDim);
    }
    return total;
}

size_t estimateEnvironmentResidentBytes(const EnvironmentCubemap& env) {
    return cubemapFaceBytes(env.faceSize) + specularMipBytes(env) + cubemapFaceBytes(env.irradianceSize);
}

size_t estimateTotalEnvironmentResidentBytes(const std::vector<EnvironmentCubemap>& envs) {
    size_t total = 0;
    for (const auto& env : envs) {
        total += estimateEnvironmentResidentBytes(env);
    }
    return total;
}

size_t estimateBRDFResidentBytes(const BRDFLookupTable& lut) {
    if (lut.size == 0) {
        return 0;
    }
    size_t dim = static_cast<size_t>(lut.size);
    return dim * dim * sizeof(float2);
}

size_t getSystemMemorySize() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return status.ullTotalPhys;
    }
    return 4ULL * 1024 * 1024 * 1024; // Fallback 4GB
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return (size_t)pages * (size_t)page_size;
    }
    return 4ULL * 1024 * 1024 * 1024; // Fallback 4GB
#endif
}

size_t getFreeVideoMemory() {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

struct RetryInfo {
    std::string sampleName;
    std::string textureName;
    bool isDirty;
};

bool stageMaterial(MaterialCache& cache, const std::filesystem::path& texturesDir,
                   const std::string& textureName, const GPUMemorySlot& slot,
                   size_t slotPixels, cudaStream_t stream, int& W, int& H) {
    const MaterialMaps* maps = cache.find(textureName);
    MaterialMaps uncached;
    if (!maps) {
        // Check the header first so oversized materials aren't decoded on every pick
        int headerW = 0;
        int headerH = 0;
        if (readPNGSize(texturesDir / textureName / "albedo.png", headerW, headerH) &&
            !materialFitsSlot(headerW, headerH, slotPixels)) {
            std::cerr << "Texture " << textureName << " is too large (" << headerW << "x" << headerH << "). GPU slots hold up to " << slotPixels << " pixels; raise --max-res to render it. Skipping..." << std::endl;
            return false;
        }

        try {
            uncached = loadMaterialMaps(texturesDir / textureName);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load texture maps for " << textureName << ": " << e.what() << ". Skipping..." << std::endl;
            return false;
        }

        if (!materialFitsSlot(uncached.width, uncached.height, slotPixels)) {
            std::cerr << "Texture " << textureName << " is too large (" << uncached.width << "x" << uncached.height << "). Skipping..." << std::endl;
            return false;
        }

        maps = cache.tryInsert(textureName, uncached);
        if (!maps) {
            maps = &uncached; // Cache is full
        }
    }

    // Staging layout: albedo RGBA8 | normal RGBA8 | roughness R8 | metallic R8
    const size_t pixelCount = maps->pixelCount();
    const size_t albedoOffset = 0;
    const size_t normalOffset = albedoOffset + pixelCount * 4;
    const size_t roughnessOffset = normalOffset + pixelCount * 4;
    const size_t metallicOffset = roughnessOffset + pixelCount;
    const size_t stagingBytes = metallicOffset + pixelCount;

    std::memcpy(slot.hMaterialStaging + albedoOffset, maps->albedo.data(), pixelCount * 4);
    std::memcpy(slot.hMaterialStaging + normalOffset, maps->normal.data(), pixelCount * 4);
    std::memcpy(slot.hMaterialStaging + roughnessOffset, maps->roughness.data(), pixelCount);
    std::memcpy(slot.hMaterialStaging + metallicOffset, maps->metallic.data(), pixelCount);
    CUDA_CHECK(cudaMemcpyAsync(slot.dMaterialStaging, slot.hMaterialStaging, stagingBytes,
                               cudaMemcpyHostToDevice, stream));

    launchUnpackMaterial(slot.dMaterialStaging + albedoOffset, slot.dMaterialStaging + normalOffset,
                         slot.dMaterialStaging + roughnessOffset, slot.dMaterialStaging + metallicOffset,
                         slot.dAlbedo, slot.dNormal, slot.dRoughness, slot.dMetallic,
                         pixelCount, stream);
    CUDA_CHECK(cudaGetLastError());
    // The render thread uses another stream, so wait for the upload here
    CUDA_CHECK(cudaStreamSynchronize(stream));

    W = maps->width;
    H = maps->height;
    return true;
}

std::string materialCacheStats(const MaterialCache& cache) {
    const size_t lookups = cache.hits() + cache.misses();
    const size_t hitPercent = lookups > 0 ? (cache.hits() * 100) / lookups : 0;
    return "material cache: " + std::to_string(cache.hits()) + " hits, " +
           std::to_string(cache.misses()) + " misses (" + std::to_string(hitPercent) + "% hit), " +
           std::to_string(cache.size()) + " materials, " +
           std::to_string(cache.residentBytes() / kBytesPerMB) + "/" +
           std::to_string(cache.budgetBytes() / kBytesPerMB) + " MB resident";
}


void loaderThread(PipelineQueues& queues,
                  const std::filesystem::path& texturesDir,
                  const std::filesystem::path& outputDir,
                  const std::vector<std::string>& textureNames,
                  int maxRenders,
                  size_t numEnvironments,
                  size_t startIndex,
                  const std::vector<RetryInfo>& retryRequests,
                  size_t materialCacheBytes,
                  size_t slotPixels) {
    ThreadSafeQueue<RenderRequest>& queue = queues.load;
    ThreadSafeQueue<GPUMemorySlot>& freeSlots = queues.freeSlots;
    try {
    CUDA_CHECK(cudaSetDevice(0));
    const cudaStream_t stream = cudaStreamPerThread;
    MaterialCache materialCache(materialCacheBytes);
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    constexpr float P_CLEAN = 0.75f;
    constexpr float P_SHADOW = 0.75f;
    constexpr float P_SMUDGE = 0.60f;

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> textureIndexDist(0, textureNames.size() - 1);
    std::uniform_int_distribution<size_t> environmentIndexDist(0, numEnvironments - 1);
    std::uniform_int_distribution<unsigned long long> seedDist;

    bool stopped = false;
    for (const auto& retry : retryRequests) {
        GPUMemorySlot slot;
        bool holdingSlot = false;
        try {
            if (!freeSlots.pop(slot)) {
                stopped = true;
                break;
            }
            holdingSlot = true;

            std::string currentTextureName = retry.textureName;
            if (currentTextureName.empty()) {
                size_t randomTexIndex = textureIndexDist(rng);
                currentTextureName = textureNames[randomTexIndex];
            } else {
                bool texFound = false;
                for (const auto& t : textureNames) {
                    if (t == currentTextureName) {
                        texFound = true;
                        break;
                    }
                }
                if (!texFound) {
                    std::cerr << "Warning: Texture " << currentTextureName << " for retry " << retry.sampleName << " not found in texture list. Attempting load anyway..." << std::endl;
                }
            }

            int W = 0;
            int H = 0;
            if (!stageMaterial(materialCache, texturesDir, currentTextureName, slot, slotPixels, stream, W, H)) {
                std::cerr << "Skipping retry " << retry.sampleName << "." << std::endl;
                freeSlots.push(slot);
                continue;
            }

            std::filesystem::path targetDir;
            bool enableShadows = false;
            bool enableCameraArtifacts = false;
            unsigned long long artifactSeed = 0;

            if (retry.isDirty) {
                targetDir = outputDir / "dirty" / retry.sampleName;
                enableShadows = uni(rng) < P_SHADOW;
                enableCameraArtifacts = uni(rng) < P_SMUDGE;
                artifactSeed = seedDist(rng);
            } else {
                targetDir = outputDir / "clean" / retry.sampleName;
            }
            std::filesystem::create_directories(targetDir);

            RenderRequest req{
                environmentIndexDist(rng),
                slot,
                targetDir,
                currentTextureName,
                retry.isDirty,
                enableShadows,
                enableCameraArtifacts,
                artifactSeed,
                W, H
            };

            holdingSlot = false;
            if (!queue.push(std::move(req))) {
                stopped = true;
                break;
            }
            std::cout << "Queued retry for " << retry.sampleName << " (" << currentTextureName << ")" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Retry Warning: " << e.what() << ". Skipping..." << std::endl;
            if (holdingSlot) {
                freeSlots.push(slot);
            }
        }
    }

    int frameIndex = 0;
    // Give up if almost nothing loads, e.g. --max-res too small
    const size_t maxConsecutiveFailures = (std::max)(static_cast<size_t>(100), 4 * textureNames.size());
    size_t consecutiveFailures = 0;
    while (!stopped && frameIndex < maxRenders) {
        if (consecutiveFailures >= maxConsecutiveFailures) {
            std::cerr << "Loader Thread: " << consecutiveFailures
                      << " consecutive materials failed to load; stopping early." << std::endl;
            break;
        }
        GPUMemorySlot slot;
        bool holdingSlot = false;
        try {
            if (!freeSlots.pop(slot)) {
                break;
            }
            holdingSlot = true;

            const std::string& textureName = textureNames[textureIndexDist(rng)];
            int W = 0;
            int H = 0;
            if (!stageMaterial(materialCache, texturesDir, textureName, slot, slotPixels, stream, W, H)) {
                freeSlots.push(slot); // Return slot
                ++consecutiveFailures;
                continue; // Skip this one and try again
            }
            consecutiveFailures = 0;

            std::string sampleName = "sample_" + std::to_string(startIndex + static_cast<size_t>(frameIndex));
            std::filesystem::path targetDir;
            bool dirtySet = uni(rng) > P_CLEAN;
            bool enableShadows = false;
            bool enableCameraArtifacts = false;
            unsigned long long artifactSeed = 0;

            if (dirtySet) {
                enableShadows = uni(rng) < P_SHADOW;
                enableCameraArtifacts = uni(rng) < P_SMUDGE;
                artifactSeed = seedDist(rng);
                targetDir = outputDir / "dirty" / sampleName;
            } else {
                targetDir = outputDir / "clean" / sampleName;
            }
            std::filesystem::create_directories(targetDir);

            RenderRequest req{
                environmentIndexDist(rng),
                slot,
                targetDir,
                textureName,
                dirtySet,
                enableShadows,
                enableCameraArtifacts,
                artifactSeed,
                W, H
            };

            holdingSlot = false;
            if (!queue.push(std::move(req))) {
                break;
            }
            frameIndex++;

            if (frameIndex % 10 == 0) {
                std::cout << "Loaded " << frameIndex << "/" << maxRenders << " requests... ("
                          << materialCacheStats(materialCache) << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Loader Thread Warning: " << e.what() << ". Skipping..." << std::endl;
            if (holdingSlot) {
                freeSlots.push(slot);
            }
            ++consecutiveFailures;
        }
    }
    std::cout << "Loader finished (" << materialCacheStats(materialCache) << ")" << std::endl;
    queue.setDone();
    } catch (const std::exception& e) {
        std::cerr << "Loader Thread Fatal Error: " << e.what() << std::endl;
        queue.setDone();
    } catch (...) {
        std::cerr << "Loader Thread Fatal Error: Unknown exception" << std::endl;
        queue.setDone();
    }
}

void renderThread(PipelineQueues& queues,
                  const std::vector<EnvironmentCubemap>& environments,
                  const BRDFLookupTable& brdfLut) {
    ThreadSafeQueue<RenderRequest>& inQueue = queues.load;
    ThreadSafeQueue<RenderResult>& outQueue = queues.write;
    try {
    CUDA_CHECK(cudaSetDevice(0));
    const cudaStream_t stream = cudaStreamPerThread;
    RenderRequest req;
    while (inQueue.pop(req)) {
        RenderResult res;
        res.targetDir = req.targetDir;
        res.width = req.width;
        res.height = req.height;
        res.textureName = req.textureName;
        res.gpuSlot = req.gpuSlot;

        for (size_t j = 0; j < kViewsPerSample; ++j) {
            uint8_t* hostFrame = slotHostFrame(req.gpuSlot, j, req.width, req.height);
            renderPlane(environments[req.environmentIndex], brdfLut,
                        req.gpuSlot.dAlbedo, req.gpuSlot.dNormal,
                        req.gpuSlot.dRoughness, req.gpuSlot.dMetallic,
                        req.gpuSlot.dFrame,
                        req.width, req.height, hostFrame, stream,
                        req.enableShadows, req.enableCameraArtifacts, req.artifactSeed);
            res.frames[j] = hostFrame;
        }
        
        if (!outQueue.push(std::move(res))) {
            break;
        }
    }
    outQueue.setDone();
    } catch (const std::exception& e) {
        std::cerr << "Render Thread Error: " << e.what() << std::endl;
        queues.abort();
    } catch (...) {
        std::cerr << "Render Thread Error: Unknown exception" << std::endl;
        queues.abort();
    }
}

void writerThread(PipelineQueues& queues, const std::filesystem::path& outputDir) {
    ThreadSafeQueue<RenderResult>& queue = queues.write;
    ThreadSafeQueue<GPUMemorySlot>& freeSlots = queues.freeSlots;
    try {
    std::map<std::string, std::string> metadataEntries;
    std::filesystem::path metadataPath = outputDir / "render_metadata.json";
    
    // Load existing metadata once at start
    loadMetadata(metadataPath, metadataEntries);

    RenderResult res;
    int saveCounter = 0;
    while (queue.pop(res)) {
        for (size_t j = 0; j < kViewsPerSample; ++j) {
            std::filesystem::path outputPath = res.targetDir / (std::to_string(j) + ".png");
            writePNGImage(outputPath, res.frames[j], res.width, res.height);
            
            // Update in-memory map
            std::string sampleKey = res.targetDir.filename().string();
            metadataEntries[sampleKey] = res.textureName;
        }
        
        // Return the GPU slot to the free list (res.frames point into it)
        freeSlots.push(res.gpuSlot);

        // Save metadata every 10 renders to avoid excessive I/O
        saveCounter++;
        if (saveCounter % 10 == 0) {
            saveMetadata(metadataPath, metadataEntries);
        }
    }
    // Save remaining metadata at the end
    saveMetadata(metadataPath, metadataEntries);
    } catch (const std::exception& e) {
        std::cerr << "Writer Thread Error: " << e.what() << std::endl;
        queues.abort();
    } catch (...) {
        std::cerr << "Writer Thread Error: Unknown exception" << std::endl;
        queues.abort();
    }
}

struct MaterialSizeScan {
    int largestWidth = 0; // of the largest material that fits the cap
    int largestHeight = 0;
    size_t readable = 0;
    size_t unreadable = 0;
    size_t overCap = 0;
};

MaterialSizeScan scanMaterialSizes(const std::filesystem::path& texturesDir,
                                   const std::vector<std::string>& textureNames,
                                   size_t maxPixels) {
    MaterialSizeScan scan;
    for (const auto& name : textureNames) {
        int w = 0;
        int h = 0;
        if (!readPNGSize(texturesDir / name / "albedo.png", w, h)) {
            ++scan.unreadable;
            continue;
        }
        ++scan.readable;
        if (pixelCount(w, h) > maxPixels) {
            ++scan.overCap;
        } else if (pixelCount(w, h) > pixelCount(scan.largestWidth, scan.largestHeight)) {
            scan.largestWidth = w;
            scan.largestHeight = h;
        }
    }
    return scan;
}

int main(int argc, char** argv) {
    const auto printUsage = [&]() {
        std::cerr << "Usage: " << argv[0] << " <textures directory> <output directory> <num renders> [--continuing] [--material-cache-mb N] [--env-cache-dir DIR] [--no-env-cache] [--max-res N] [--pipeline-depth N]" << std::endl;
    };
    if (argc < 4) {
        printUsage();
        return 1;
    }
    try {
        const int maxRenders = std::stoi(argv[3]);
        bool continuing = false;
        long long materialCacheMB = -1; // -1 = default
        std::filesystem::path envCacheDir = std::filesystem::path("cache") / "envmaps";
        bool envCacheEnabled = true;
        int maxResOverride = 0; // 0 = largest material
        int pipelineDepthCap = kMaxPipelineDepth;

        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--continuing" || arg == "-c") {
                continuing = true;
            } else if (arg == "--material-cache-mb") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --material-cache-mb" << std::endl;
                    printUsage();
                    return 1;
                }
                std::string value = argv[++i];
                size_t parsedChars = 0;
                try {
                    materialCacheMB = std::stoll(value, &parsedChars);
                } catch (const std::exception&) {
                    parsedChars = 0;
                }
                if (parsedChars != value.size() || materialCacheMB < 0) {
                    std::cerr << "Invalid value for --material-cache-mb: " << value << std::endl;
                    printUsage();
                    return 1;
                }
            } else if (arg == "--env-cache-dir") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --env-cache-dir" << std::endl;
                    printUsage();
                    return 1;
                }
                std::string value = argv[++i];
                if (value.empty()) {
                    std::cerr << "Invalid value for --env-cache-dir: (empty)" << std::endl;
                    printUsage();
                    return 1;
                }
                envCacheDir = value;
            } else if (arg == "--no-env-cache") {
                envCacheEnabled = false;
            } else if (arg == "--max-res" || arg == "--pipeline-depth") {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for " << arg << std::endl;
                    printUsage();
                    return 1;
                }
                std::string value = argv[++i];
                const bool isMaxRes = (arg == "--max-res");
                int& target = isMaxRes ? maxResOverride : pipelineDepthCap;
                if (!parseIntInRange(value, isMaxRes ? kMaxResLimit : kPipelineDepthLimit, target)) {
                    std::cerr << "Invalid value for " << arg << ": " << value << " (expected 1.."
                              << (isMaxRes ? kMaxResLimit : kPipelineDepthLimit) << ")" << std::endl;
                    printUsage();
                    return 1;
                }
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                printUsage();
                return 1;
            }
        }
        if (!envCacheEnabled) {
            envCacheDir.clear();
        }

        CUDA_CHECK(cudaSetDevice(0));
        fpng::fpng_init();

        const std::filesystem::path hdrDir = std::filesystem::path("assets") / "hdris";

        constexpr unsigned kEnvFaceSize = 512;
        constexpr unsigned kIrradianceFaceSize = 32;
        constexpr unsigned kSpecularSamples = 1024;
        constexpr unsigned kDiffuseSamples = 512;
        constexpr unsigned kHDRMaxWidth = kEnvFaceSize * kHDRWidthPerCubeFace;

        std::cout << "Loading and precomputing environment cubemaps from " << hdrDir << "..." << std::endl;
        if (envCacheDir.empty()) {
            std::cout << "Environment cache: disabled" << std::endl;
        } else {
            std::cout << "Environment cache: " << envCacheDir << std::endl;
        }

        auto environments = loadEnvironmentCubemaps(hdrDir, kEnvFaceSize, kIrradianceFaceSize, kSpecularSamples, kDiffuseSamples, kHDRMaxWidth, envCacheDir);

        if (environments.empty()) {
            throw std::runtime_error("No HDRI environments found. Ensure the assets/hdris directory contains .hdr files.");
        }

        std::cout << "Generated cubemaps: " << environments.size() << std::endl;
        std::cout << "Precomputing BRDF lookup table..." << std::endl;

        const size_t environmentResidentBytes = estimateTotalEnvironmentResidentBytes(environments);

        BRDFLookupTable brdfLut = createBRDFLUT(512);
        loadBRDFLUT(brdfLut);
        const size_t brdfResidentBytes = estimateBRDFResidentBytes(brdfLut);

        std::cout << "Precomputation complete. Starting rendering. Press CTRL+C to stop." << std::endl;

        std::vector<std::string> textureNames;
        const std::filesystem::path texturesDir = argv[1];
        const std::filesystem::path outputDir = argv[2];
        for (const auto& entry : std::filesystem::directory_iterator(texturesDir)) {
            if (entry.is_directory()) {
                textureNames.push_back(entry.path().filename().string());
            }
        }
        if (textureNames.empty()) {
            throw std::runtime_error("No material subdirectories found in textures directory.");
        }

        size_t slotPixels = pixelCount(maxResOverride, maxResOverride);
        if (maxResOverride > 0) {
            std::cout << "Material slot size: " << maxResOverride << "x" << maxResOverride << " pixels (--max-res)" << std::endl;
        } else {
            const auto scanStart = std::chrono::steady_clock::now();
            const MaterialSizeScan scan = scanMaterialSizes(texturesDir, textureNames, kDefaultMaxSlotPixels);
            const auto scanMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - scanStart).count();
            if (scan.readable == 0) {
                throw std::runtime_error("Could not read the size of any material's albedo.png; pass --max-res N to size GPU slots explicitly.");
            }
            if (scan.overCap == scan.readable) {
                throw std::runtime_error("Every material is larger than 2048x2048; pass --max-res N to render them.");
            }
            slotPixels = pixelCount(scan.largestWidth, scan.largestHeight);
            std::cout << "Material slot size: " << scan.largestWidth << "x" << scan.largestHeight << " pixels (largest of "
                      << scan.readable << " materials, scanned in " << scanMs << " ms";
            if (scan.unreadable > 0) {
                std::cout << "; " << scan.unreadable << " without a readable albedo.png";
            }
            std::cout << ")" << std::endl;
            if (scan.overCap > 0) {
                std::cerr << "Warning: " << scan.overCap << " materials are larger than 2048x2048 and will be skipped; pass --max-res N to include them." << std::endl;
            }
        }
        
        size_t startIndex = 0;
        std::vector<RetryInfo> retryRequests;
        if (continuing) {
            std::cout << "Continuing mode enabled. Scanning for incomplete samples..." << std::endl;
            std::map<std::string, std::string> metadata;
            std::filesystem::path metadataPath = outputDir / "render_metadata.json";
            loadMetadata(metadataPath, metadata);

            size_t maxIndex = 0;
            
            auto scanDir = [&](const std::filesystem::path& dir, bool isDirty) {
                if (!std::filesystem::exists(dir)) return;
                for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                    if (entry.is_directory()) {
                        std::string name = entry.path().filename().string();
                        if (name.rfind("sample_", 0) == 0) {
                            // Extract index
                            try {
                                size_t idx = std::stoull(name.substr(7));
                                if (idx > maxIndex) maxIndex = idx;

                                // Check completeness
                                bool complete = true;
                                for (int j = 0; j < 3; ++j) {
                                    auto imgPath = entry.path() / (std::to_string(j) + ".png");
                                    if (!std::filesystem::exists(imgPath) || !isPNGReadable(imgPath)) {
                                        complete = false;
                                        break;
                                    }
                                }

                                if (!complete) {
                                    if (metadata.count(name)) {
                                        retryRequests.push_back({name, metadata[name], isDirty});
                                    } else {
                                        std::cout << "Queueing incomplete sample " << name << " for regeneration (no metadata found)" << std::endl;
                                        retryRequests.push_back({name, "", isDirty});
                                    }
                                }
                            } catch (...) {}
                        }
                    }
                }
            };

            scanDir(outputDir / "clean", false);
            scanDir(outputDir / "dirty", true);

            startIndex = maxIndex + 1;
            std::cout << "Found " << retryRequests.size() << " incomplete samples to retry." << std::endl;
            std::cout << "New start index: " << startIndex << std::endl;
        }
        
        const size_t staticCpuReservation = environmentResidentBytes + brdfResidentBytes;

        size_t totalRam = getSystemMemorySize();

        // Default: min(2 GB, 25% of system RAM)
        size_t materialCacheBytes = (std::min)(static_cast<size_t>(2048) * kBytesPerMB, totalRam / 4);
        if (materialCacheMB >= 0) {
            const size_t totalRamMB = totalRam / kBytesPerMB;
            if (static_cast<unsigned long long>(materialCacheMB) > totalRamMB) {
                std::cerr << "Warning: --material-cache-mb " << materialCacheMB << " exceeds system RAM; using "
                          << totalRamMB << " MB." << std::endl;
            }
            materialCacheBytes = (std::min)(static_cast<size_t>(materialCacheMB), totalRamMB) * kBytesPerMB;
        }
        if (materialCacheBytes == 0) {
            std::cout << "Material cache: disabled" << std::endl;
        } else {
            std::cout << "Material cache budget: " << materialCacheBytes / kBytesPerMB << " MB" << std::endl;
        }

        size_t reservedRam = 8ULL * 1024 * 1024 * 1024; // Reserve 8GB for OS/other
        if (totalRam < reservedRam) reservedRam = totalRam / 2; // If low RAM, reserve half

        size_t availableRam = totalRam - reservedRam;
        const size_t decodeReservation = loaderDecodeBytes(slotPixels);
        const size_t hostReservation = staticCpuReservation + materialCacheBytes + decodeReservation;
        if (hostReservation > 0) {
            std::cout << "Host reservation: " << staticCpuReservation / kBytesPerMB
                      << " MB precomputed assets + " << materialCacheBytes / kBytesPerMB
                      << " MB material cache + " << formatMB(decodeReservation)
                      << " loader decode buffer" << std::endl;
            if (hostReservation >= availableRam) {
                std::cerr << "Warning: Precomputed assets, the material cache and the loader decode buffer consume the dynamic host budget." << std::endl;
                availableRam = 0;
            } else {
                availableRam -= hostReservation;
            }
        }

        // Memory calculation per batch item (GPU slot), for 512x512 materials:
        // 1. Pinned material staging (upload source):
        // - Albedo RGBA8 + Normal RGBA8 + Roughness R8 + Metallic R8 = 10 bytes per pixel
        // - Size: 512 * 512 * 10 bytes = 2,621,440 bytes (~2.5 MB)
        // 2. Pinned output frames (writer reads these directly):
        // - 3 views, packed 8-bit RGB
        // - Size: 3 * 512 * 512 * 3 bytes = 2,359,296 bytes (~2.25 MB)
        // Total per item: ~4.75 MB (~76 MB at 2048x2048)
        // The loader's decode buffer (10 bytes per pixel) and the material cache are reserved once above.
        const size_t memoryPerBatchItemCPU = slotPinnedBytes(slotPixels);

        // GPU Memory Calculation
        size_t freeVRAM = getFreeVideoMemory();
        size_t reservedVRAM = 1ULL * 1024 * 1024 * 1024; // Reserve 1GB
        if (freeVRAM < reservedVRAM) reservedVRAM = freeVRAM / 4;
        size_t availableVRAM = freeVRAM - reservedVRAM;

        // GPU Memory per slot (512x512)
        // Albedo (float4) + Normal (float4) + Roughness (float) + Metallic (float) + Frame (RGB8) + Staging (10 x uint8)
        // (4 + 4 + 1 + 1) * 4 + 3 + 10 = 53 bytes per pixel
        // 53 * 262,144 = 13,893,632 bytes (~13.25 MB; ~212 MB at 2048x2048)
        const size_t memoryPerBatchItemGPU = slotBytes(slotPixels);

        const PipelineDepthChoice pipeline = choosePipelineDepth(availableRam, memoryPerBatchItemCPU,
                                                                 availableVRAM, memoryPerBatchItemGPU,
                                                                 pipelineDepthCap);
        const int batchSize = pipeline.depth;

        std::cout << "Detected System RAM: " << totalRam / kBytesPerMB << " MB. CPU Batch: " << pipeline.cpuItems
                  << " (" << formatMB(memoryPerBatchItemCPU) << " per request)" << std::endl;
        std::cout << "Detected Free VRAM: " << freeVRAM / kBytesPerMB << " MB. GPU Batch: " << pipeline.gpuItems
                  << " (" << formatMB(memoryPerBatchItemGPU) << " per slot)" << std::endl;
        std::cout << "Using pipeline depth: " << batchSize << " (limited by " << depthLimitName(pipeline.limit)
                  << "; --pipeline-depth " << pipelineDepthCap << "), " << batchSize << " x "
                  << formatMB(memoryPerBatchItemGPU) << " = "
                  << formatMB(static_cast<size_t>(batchSize) * memoryPerBatchItemGPU)
                  << " of GPU slots" << std::endl;
        if (pipeline.belowMinimum) {
            std::cerr << "Warning: Available memory fits fewer than 2 in-flight requests; loading, rendering and writing will not overlap." << std::endl;
        }

        // Pre-allocate GPU memory
        std::vector<GPUMemorySlot> gpuSlots;
        gpuSlots.reserve(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            gpuSlots.push_back(allocateGPUSlot(slotPixels));
        }

        PipelineQueues queues(batchSize);
        for (const auto& slot : gpuSlots) {
            queues.freeSlots.push(slot);
        }

        std::thread t1(loaderThread, std::ref(queues), texturesDir, outputDir, textureNames, maxRenders, environments.size(), startIndex, retryRequests, materialCacheBytes, slotPixels);
        std::thread t2(renderThread, std::ref(queues), std::cref(environments), std::cref(brdfLut));
        std::thread t3(writerThread, std::ref(queues), outputDir);

        t1.join();
        t2.join();
        t3.join();

        // Cleanup GPU and pinned host memory
        for (const auto& slot : gpuSlots) {
            freeGPUSlot(slot);
        }

        std::cout << "Rendering complete!" << std::endl;
        return 0;
    } catch (const CudaError& e) {
        std::cerr << "CUDA Failure: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}