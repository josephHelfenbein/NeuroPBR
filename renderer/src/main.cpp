#include <renderer.h>
#include <io.h>
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

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(size_t maxSize) : maxSize(maxSize), done(false) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex);
        notFull.wait(lock, [this] { return queue.size() < maxSize; });
        queue.push(std::move(item));
        notEmpty.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        notEmpty.wait(lock, [this] { return !queue.empty() || done; });
        if (queue.empty() && done) return false;
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

private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable notEmpty;
    std::condition_variable notFull;
    size_t maxSize;
    bool done;
};

struct RenderRequest {
    size_t environmentIndex;
    FloatImage albedo;
    FloatImage normal;
    FloatImage roughness;
    FloatImage metallic;
    std::filesystem::path targetDir;
    bool dirtySet;
    bool enableShadows;
    bool enableCameraArtifacts;
    unsigned long long artifactSeed;
    int width;
    int height;
};

struct RenderResult {
    std::vector<std::vector<float4>> frameRGBAs;
    std::filesystem::path targetDir;
    int width;
    int height;
};

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

void loaderThread(ThreadSafeQueue<RenderRequest>& queue,
                  const std::filesystem::path& texturesDir,
                  const std::vector<std::string>& textureNames,
                  int maxRenders,
                  size_t numEnvironments) {
    std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    constexpr float P_CLEAN = 0.75f;
    constexpr float P_SHADOW = 0.75f;
    constexpr float P_SMUDGE = 0.60f;

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::uniform_int_distribution<size_t> textureIndexDist(0, textureNames.size() - 1);
    std::uniform_int_distribution<size_t> environmentIndexDist(0, numEnvironments - 1);
    std::uniform_int_distribution<unsigned long long> seedDist;

    int frameIndex = 0;
    while (frameIndex < maxRenders) {
        size_t randomTexIndex = textureIndexDist(rng);
        FloatImage dAlbedo = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "albedo.png", 3, true);
        FloatImage dNormal = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "normal.png", 3, true);
        FloatImage dRoughness = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "roughness.png", 1, true);
        FloatImage dMetallic = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "metallic.png", 1, true);

        if (dAlbedo.data.empty() || dNormal.data.empty() || dRoughness.data.empty() || dMetallic.data.empty()) {
            std::cerr << "Failed to load all required texture maps for " << textureNames[randomTexIndex] << std::endl;
            continue; // Skip this one and try again
        }

        int W = dAlbedo.width;
        int H = dAlbedo.height;
        std::string sampleName = "sample_" + std::to_string(frameIndex);
        std::filesystem::path targetDir;
        bool dirtySet = uni(rng) < P_CLEAN;
        bool enableShadows = false;
        bool enableCameraArtifacts = false;
        unsigned long long artifactSeed = 0;

        if (dirtySet) {
            enableShadows = uni(rng) < P_SHADOW;
            enableCameraArtifacts = uni(rng) < P_SMUDGE;
            artifactSeed = seedDist(rng);
            targetDir = std::filesystem::path("output") / "dirty" / sampleName;
        } else {
            targetDir = std::filesystem::path("output") / "clean" / sampleName;
        }
        std::filesystem::create_directories(targetDir);

        RenderRequest req{
            environmentIndexDist(rng),
            std::move(dAlbedo),
            std::move(dNormal),
            std::move(dRoughness),
            std::move(dMetallic),
            targetDir,
            dirtySet,
            enableShadows,
            enableCameraArtifacts,
            artifactSeed,
            W, H
        };

        queue.push(std::move(req));
        frameIndex++;

        if (frameIndex % 10 == 0) {
             std::cout << "Loaded " << frameIndex << "/" << maxRenders << " requests..." << std::endl;
        }
    }
    queue.setDone();
}

void renderThread(ThreadSafeQueue<RenderRequest>& inQueue,
                  ThreadSafeQueue<RenderResult>& outQueue,
                  const std::vector<EnvironmentCubemap>& environments,
                  const BRDFLookupTable& brdfLut) {
    CUDA_CHECK(cudaSetDevice(0));
    RenderRequest req;
    while (inQueue.pop(req)) {
        RenderResult res;
        res.targetDir = req.targetDir;
        res.width = req.width;
        res.height = req.height;
        res.frameRGBAs.resize(3);

        for (int j = 0; j < 3; ++j) {
            renderPlane(environments[req.environmentIndex], brdfLut,
                        req.albedo.data.data(), req.normal.data.data(),
                        req.roughness.data.data(), req.metallic.data.data(),
                        req.width, req.height, res.frameRGBAs[j],
                        req.enableShadows, req.enableCameraArtifacts, req.artifactSeed);
        }
        
        outQueue.push(std::move(res));
    }
    outQueue.setDone();
}

void writerThread(ThreadSafeQueue<RenderResult>& queue) {
    RenderResult res;
    while (queue.pop(res)) {
        for (int j = 0; j < 3; ++j) {
            std::filesystem::path outputPath = res.targetDir / (std::to_string(j) + ".png");
            writePNGImage(outputPath, res.frameRGBAs[j].data(), res.width, res.height, true);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <textures directory>" << std::endl;
        return 1;
    }
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::filesystem::path hdrDir = std::filesystem::path("assets") / "hdris";

        constexpr unsigned kEnvFaceSize = 512;
        constexpr unsigned kIrradianceFaceSize = 32;
        constexpr unsigned kSpecularSamples = 1024;
        constexpr unsigned kDiffuseSamples = 512;

        std::cout << "Loading and precomputing environment cubemaps from " << hdrDir << "..." << std::endl;

        auto environments = loadEnvironmentCubemaps(hdrDir, kEnvFaceSize, kIrradianceFaceSize, kSpecularSamples, kDiffuseSamples);

        if (environments.empty()) {
            throw std::runtime_error("No HDRI environments found. Ensure the assets/hdris directory contains .hdr files.");
        }

        std::cout << "Generated cubemaps: " << environments.size() << std::endl;
        std::cout << "Precomputing BRDF lookup table..." << std::endl;

        BRDFLookupTable brdfLut = createBRDFLUT(512);
        loadBRDFLUT(brdfLut);

        std::cout << "Precomputation complete. Starting rendering. Press CTRL+C to stop." << std::endl;

        std::vector<std::string> textureNames;
        const std::filesystem::path texturesDir = argv[1];
        for (const auto& entry : std::filesystem::directory_iterator(texturesDir)) {
            if (entry.is_directory()) {
                textureNames.push_back(entry.path().filename().string());
            }
        }
        if (textureNames.empty()) {
            throw std::runtime_error("No material subdirectories found in textures directory.");
        }
        
        const int maxRenders = std::stoi(argv[2]);
        
        size_t totalRam = getSystemMemorySize();
        size_t reservedRam = 4ULL * 1024 * 1024 * 1024; // Reserve 4GB for OS/other
        if (totalRam < reservedRam) reservedRam = totalRam / 2; // If low RAM, reserve half

        size_t availableRam = totalRam - reservedRam;
        
        // Memory calculation per batch item (Queue entry):
        // 1. Load Queue (RenderRequest):
        // - 4 images (Albedo, Normal, Roughness, Metallic)
        // - Total channels: 3 + 3 + 1 + 1 = 8 channels
        // - Size: 2048 * 2048 * 8 floats * 4 bytes = 134,217,728 bytes (~128 MB)
        // 2. Write Queue (RenderResult):
        // - 3 frames
        // - Stored as float4 (RGBA) for alignment/CUDA
        // - Size: 3 * 2048 * 2048 * 16 bytes (float4) = 201,326,592 bytes (~192 MB)
        // Total per item: ~128 MB + ~192 MB = ~320 MB
        // Using 350 MB to be safe.
        size_t memoryPerBatchItem = 350ULL * 1024 * 1024; 

        int batchSize = (int)(availableRam / memoryPerBatchItem);
        if (batchSize < 2) batchSize = 2;
        if (batchSize > 64) batchSize = 64;

        std::cout << "Detected System RAM: " << totalRam / (1024*1024) << " MB. Using batch size: " << batchSize << std::endl;

        ThreadSafeQueue<RenderRequest> loadQueue(batchSize);
        ThreadSafeQueue<RenderResult> writeQueue(batchSize);

        std::thread t1(loaderThread, std::ref(loadQueue), texturesDir, textureNames, maxRenders, environments.size());
        std::thread t2(renderThread, std::ref(loadQueue), std::ref(writeQueue), std::cref(environments), std::cref(brdfLut));
        std::thread t3(writerThread, std::ref(writeQueue));

        t1.join();
        t2.join();
        t3.join();

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