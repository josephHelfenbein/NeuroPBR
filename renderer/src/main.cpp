#include <renderer.h>
#include <io.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <array>
#include <string>
#include <random>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <textures directory>" << std::endl;
        return 1;
    }
    try {
        CUDA_CHECK(cudaSetDevice(0));

        const std::filesystem::path hdrDir = std::filesystem::path("assets") / "hdris";
        const std::filesystem::path lensflaresDir = std::filesystem::path("assets") / "lensflares";
        const std::filesystem::path smudgesDir = std::filesystem::path("assets") / "camerasmudges";

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

        std::cout << "Loading camera smudge textures..." << std::endl;
        std::vector<FloatImage> cameraSmudges;
        for (const auto& entry : std::filesystem::directory_iterator(smudgesDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                cameraSmudges.push_back(loadPNGImage(entry.path(), 4, true));
            }
        }
        std::vector<FloatImage> lensFlares;
        for (const auto& entry : std::filesystem::directory_iterator(lensflaresDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                lensFlares.push_back(loadPNGImage(entry.path(), 4, true));
            }
        }

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
        int W = 0, H = 0, frameIndex = 0;
        const int maxRenders = std::stoi(argv[2]);
        std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        constexpr float P_CLEAN = 0.75f;

        constexpr float P_SHADOW = 0.50f;
        constexpr float P_SMUDGE = 0.75f;

        std::uniform_real_distribution<float> uni(0.0f, 1.0f);
        std::uniform_int_distribution<size_t> textureIndexDist(0, textureNames.size() - 1);
        std::uniform_int_distribution<size_t> environmentIndexDist(0, environments.size() - 1);

        while (frameIndex < maxRenders) {
            size_t randomTexIndex = textureIndexDist(rng);
            size_t randomEnvIndex = environmentIndexDist(rng);
            FloatImage dAlbedo = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "albedo.png", 3, true);
            FloatImage dNormal = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "normal.png", 3, true);
            FloatImage dRoughness = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "roughness.png", 1, true);
            FloatImage dMetallic = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "metallic.png", 1, true);
            W = dAlbedo.width;
            H = dAlbedo.height;

            std::array<std::vector<float4>, 3> frameRGBAs;
            std::string sampleName = "sample_" + std::to_string(frameIndex);

            bool dirtySet = (uni(rng) >= P_CLEAN);
            std::filesystem::path targetDir;
            if (dirtySet) {
                bool enableShadows = (uni(rng) < P_SHADOW);
                bool enableCameraSmudge = (uni(rng) < P_SMUDGE);
                targetDir = std::filesystem::path("output") / "dirty" / sampleName;
                std::filesystem::create_directories(targetDir);
                
                FloatImage* cameraSmudge = nullptr;
                if (!cameraSmudges.empty()) {
                    std::uniform_int_distribution<size_t> smudgeIndexDist(0, cameraSmudges.size() - 1);
                    cameraSmudge = &cameraSmudges[smudgeIndexDist(rng)];
                }

                for (int i = 0; i < 3; ++i) {
                    renderPlane(environments[randomEnvIndex], brdfLut,
                                dAlbedo.data.data(), dNormal.data.data(),
                                dRoughness.data.data(), dMetallic.data.data(),
                                W, H, frameRGBAs[i], 
                                enableShadows, enableCameraSmudge, cameraSmudge);

                    std::filesystem::path outputPath = targetDir /
                        (std::to_string(i) + ".png");
                    writePNGImage(outputPath, frameRGBAs[i].data(), W, H, true);
                }
            } else {
                targetDir = std::filesystem::path("output") / "clean" / sampleName;
                std::filesystem::create_directories(targetDir);
                for (int i = 0; i < 3; ++i) {
                    renderPlane(environments[randomEnvIndex], brdfLut,
                                dAlbedo.data.data(), dNormal.data.data(),
                                dRoughness.data.data(), dMetallic.data.data(),
                                W, H, frameRGBAs[i],
                                false, false, nullptr);

                    std::filesystem::path outputPath = targetDir /
                        (std::to_string(i) + ".png");
                    writePNGImage(outputPath, frameRGBAs[i].data(), W, H, true);
                }
            }
            appendRenderMetadata(std::filesystem::path("output") / "render_metadata.json",
                                 targetDir.string(), textureNames[randomTexIndex]);
            frameIndex++;
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