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
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

std::filesystem::path metadataPath;

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

struct GPUMemorySlot {
    float4* dAlbedo = nullptr;
    float4* dNormal = nullptr;
    float* dRoughness = nullptr;
    float* dMetallic = nullptr;
    float4* dFrame = nullptr;
};

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
    std::vector<std::vector<float4>> frameRGBAs;
    std::filesystem::path targetDir;
    std::string textureName;
    int width;
    int height;
    GPUMemorySlot gpuSlot;
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

size_t getFreeVideoMemory() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

void loaderThread(ThreadSafeQueue<RenderRequest>& queue,
                  ThreadSafeQueue<GPUMemorySlot>& freeSlots,
                  const std::filesystem::path& texturesDir,
                  const std::vector<std::string>& textureNames,
                  int maxRenders,
                  size_t numEnvironments,
                  size_t startIndex) {
    try {
    CUDA_CHECK(cudaSetDevice(0));
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
        try {
            GPUMemorySlot slot;
            if (!freeSlots.pop(slot)) {
                break; // Should not happen unless done
            }

            size_t randomTexIndex = textureIndexDist(rng);
            FloatImage dAlbedo = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "albedo.png", 3, true);
            FloatImage dNormal = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "normal.png", 3, true);
            FloatImage dRoughness = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "roughness.png", 1, true);
            FloatImage dMetallic = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "metallic.png", 1, true);

            if (dAlbedo.data.empty() || dNormal.data.empty() || dRoughness.data.empty() || dMetallic.data.empty()) {
                std::cerr << "Failed to load all required texture maps for " << textureNames[randomTexIndex] << std::endl;
                freeSlots.push(slot); // Return slot
                continue; // Skip this one and try again
            }

            std::string textureName = textureNames[randomTexIndex];

            int W = dAlbedo.width;
            int H = dAlbedo.height;
            size_t pixelCount = static_cast<size_t>(W) * static_cast<size_t>(H);

            if (pixelCount > 2048 * 2048) {
                std::cerr << "Texture " << textureNames[randomTexIndex] << " is too large (" << W << "x" << H << "). Max supported is 2048x2048. Skipping..." << std::endl;
                freeSlots.push(slot);
                continue;
            }

            // Pack RGB to RGBA for Albedo and Normal
            std::vector<float4> packedAlbedo(pixelCount);
            std::vector<float4> packedNormal(pixelCount);
            
            for (size_t i = 0; i < pixelCount; ++i) {
                packedAlbedo[i] = make_float4(dAlbedo.data[i * 3 + 0], dAlbedo.data[i * 3 + 1], dAlbedo.data[i * 3 + 2], 0.0f);
                packedNormal[i] = make_float4(dNormal.data[i * 3 + 0], dNormal.data[i * 3 + 1], dNormal.data[i * 3 + 2], 0.0f);
            }

            CUDA_CHECK(cudaMemcpy(slot.dAlbedo, packedAlbedo.data(), pixelCount * sizeof(float4), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(slot.dNormal, packedNormal.data(), pixelCount * sizeof(float4), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(slot.dRoughness, dRoughness.data.data(), pixelCount * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(slot.dMetallic, dMetallic.data.data(), pixelCount * sizeof(float), cudaMemcpyHostToDevice));

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
                targetDir = std::filesystem::path("output") / "dirty" / sampleName;
            } else {
                targetDir = std::filesystem::path("output") / "clean" / sampleName;
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

            queue.push(std::move(req));
            frameIndex++;

            if (frameIndex % 10 == 0) {
                std::cout << "Loaded " << frameIndex << "/" << maxRenders << " requests..." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Loader Thread Warning: " << e.what() << ". Skipping..." << std::endl;
        }
    }
    queue.setDone();
    } catch (const std::exception& e) {
        std::cerr << "Loader Thread Fatal Error: " << e.what() << std::endl;
        queue.setDone();
    } catch (...) {
        std::cerr << "Loader Thread Fatal Error: Unknown exception" << std::endl;
        queue.setDone();
    }
}

void renderThread(ThreadSafeQueue<RenderRequest>& inQueue,
                  ThreadSafeQueue<RenderResult>& outQueue,
                  const std::vector<EnvironmentCubemap>& environments,
                  const BRDFLookupTable& brdfLut) {
    try {
    CUDA_CHECK(cudaSetDevice(0));
    RenderRequest req;
    while (inQueue.pop(req)) {
        RenderResult res;
        res.targetDir = req.targetDir;
        res.width = req.width;
        res.height = req.height;
        res.frameRGBAs.resize(3);
        res.textureName = req.textureName;
        res.gpuSlot = req.gpuSlot;

        for (int j = 0; j < 3; ++j) {
            renderPlane(environments[req.environmentIndex], brdfLut,
                        req.gpuSlot.dAlbedo, req.gpuSlot.dNormal,
                        req.gpuSlot.dRoughness, req.gpuSlot.dMetallic,
                        req.gpuSlot.dFrame,
                        req.width, req.height, res.frameRGBAs[j],
                        req.enableShadows, req.enableCameraArtifacts, req.artifactSeed);
        }
        
        outQueue.push(std::move(res));
    }
    outQueue.setDone();
    } catch (const std::exception& e) {
        std::cerr << "Render Thread Error: " << e.what() << std::endl;
        outQueue.setDone();
    } catch (...) {
        std::cerr << "Render Thread Error: Unknown exception" << std::endl;
        outQueue.setDone();
    }
}

void writerThread(ThreadSafeQueue<RenderResult>& queue, ThreadSafeQueue<GPUMemorySlot>& freeSlots) {
    try {
    std::map<std::string, std::string> metadataEntries;
    std::filesystem::path metadataPath = std::filesystem::path("output") / "render_metadata.json";
    
    // Load existing metadata once at start
    loadMetadata(metadataPath, metadataEntries);

    RenderResult res;
    int saveCounter = 0;
    while (queue.pop(res)) {
        for (int j = 0; j < 3; ++j) {
            std::filesystem::path outputPath = res.targetDir / (std::to_string(j) + ".png");
            writePNGImage(outputPath, res.frameRGBAs[j].data(), res.width, res.height, true);
            
            // Update in-memory map
            std::string sampleKey = res.targetDir.filename().string();
            metadataEntries[sampleKey] = res.textureName;
        }
        
        // Return the GPU slot to the free list
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
    } catch (...) {
        std::cerr << "Writer Thread Error: Unknown exception" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <textures directory> <num renders> [--start-index N]" << std::endl;
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
        size_t startIndex = 0;

        for (int i = 3; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--start-index" || arg == "-s") && i + 1 < argc) {
                long long parsed = std::stoll(argv[++i]);
                if (parsed < 0) {
                    throw std::invalid_argument("start index must be non-negative");
                }
                startIndex = static_cast<size_t>(parsed);
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                std::cerr << "Usage: " << argv[0] << " <textures directory> <num renders> [--start-index N]" << std::endl;
                return 1;
            }
        }
        
        size_t totalRam = getSystemMemorySize();
        size_t reservedRam = 8ULL * 1024 * 1024 * 1024; // Reserve 8GB for OS/other
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
        // Using 512 MB to be safe.
        size_t memoryPerBatchItemCPU = 512ULL * 1024 * 1024; 

        int cpuBatchSize = (int)(availableRam / memoryPerBatchItemCPU);
        if (cpuBatchSize < 2) cpuBatchSize = 2;
        if (cpuBatchSize > 64) cpuBatchSize = 64;

        // GPU Memory Calculation
        size_t freeVRAM = getFreeVideoMemory();
        size_t reservedVRAM = 1ULL * 1024 * 1024 * 1024; // Reserve 1GB
        if (freeVRAM < reservedVRAM) reservedVRAM = freeVRAM / 4;
        size_t availableVRAM = freeVRAM - reservedVRAM;

        // GPU Memory per slot (2048x2048)
        // Albedo (float4) + Normal (float4) + Roughness (float) + Metallic (float) + Frame (float4)
        // (4 + 4 + 1 + 1 + 4) * 4 bytes * 2048 * 2048
        // 14 * 4 * 4194304 = 234,881,024 bytes (~224 MB)
        size_t memoryPerBatchItemGPU = 235ULL * 1024 * 1024;

        int gpuBatchSize = (int)(availableVRAM / memoryPerBatchItemGPU);
        if (gpuBatchSize < 2) gpuBatchSize = 2;

        int batchSize = (std::min)(cpuBatchSize, gpuBatchSize);

        std::cout << "Detected System RAM: " << totalRam / (1024*1024) << " MB. CPU Batch: " << cpuBatchSize << std::endl;
        std::cout << "Detected Free VRAM: " << freeVRAM / (1024*1024) << " MB. GPU Batch: " << gpuBatchSize << std::endl;
        std::cout << "Using batch size: " << batchSize << std::endl;

        // Pre-allocate GPU memory
        std::vector<GPUMemorySlot> gpuSlots(batchSize);
        size_t pixelCount = 2048 * 2048;
        for (int i = 0; i < batchSize; ++i) {
            CUDA_CHECK(cudaMalloc(&gpuSlots[i].dAlbedo, pixelCount * sizeof(float4)));
            CUDA_CHECK(cudaMalloc(&gpuSlots[i].dNormal, pixelCount * sizeof(float4)));
            CUDA_CHECK(cudaMalloc(&gpuSlots[i].dRoughness, pixelCount * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuSlots[i].dMetallic, pixelCount * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpuSlots[i].dFrame, pixelCount * sizeof(float4)));
        }

        ThreadSafeQueue<RenderRequest> loadQueue(batchSize);
        ThreadSafeQueue<RenderResult> writeQueue(batchSize);
        ThreadSafeQueue<GPUMemorySlot> freeSlots(batchSize);

        for (const auto& slot : gpuSlots) {
            freeSlots.push(slot);
        }

        std::thread t1(loaderThread, std::ref(loadQueue), std::ref(freeSlots), texturesDir, textureNames, maxRenders, environments.size(), startIndex);
        std::thread t2(renderThread, std::ref(loadQueue), std::ref(writeQueue), std::cref(environments), std::cref(brdfLut));
        std::thread t3(writerThread, std::ref(writeQueue), std::ref(freeSlots));

        t1.join();
        t2.join();
        t3.join();

        // Cleanup GPU memory
        for (int i = 0; i < batchSize; ++i) {
            cudaFree(gpuSlots[i].dAlbedo);
            cudaFree(gpuSlots[i].dNormal);
            cudaFree(gpuSlots[i].dRoughness);
            cudaFree(gpuSlots[i].dMetallic);
            cudaFree(gpuSlots[i].dFrame);
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