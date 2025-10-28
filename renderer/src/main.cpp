#include <renderer.h>
#include <io.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <array>
#include <string>

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
        int W = 0, H = 0, frameIndex = 0;
        const int maxRenders = std::stoi(argv[2]);
        while (frameIndex < maxRenders) {
            int randomTexIndex = rand() % static_cast<int>(textureNames.size());
            int randomEnvIndex = rand() % static_cast<int>(environments.size());
            FloatImage dAlbedo = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "albedo.png", 3, true);
            FloatImage dNormal = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "normal.png", 3, true);
            FloatImage dRoughness = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "roughness.png", 1, true);
            FloatImage dMetallic = loadPNGImage(texturesDir / textureNames[randomTexIndex] / "metallic.png", 1, true);
            W = dAlbedo.width;
            H = dAlbedo.height;

            std::array<std::vector<float4>, 3> frameRGBAs;
            std::filesystem::path sampleDir = std::filesystem::path("output") /
                                              ("sample_" + std::to_string(frameIndex));
            std::filesystem::create_directories(sampleDir);
            for (int i = 0; i < 3; ++i) {
                renderPlane(environments[randomEnvIndex], brdfLut,
                            dAlbedo.data.data(), dNormal.data.data(),
                            dRoughness.data.data(), dMetallic.data.data(),
                            W, H, frameRGBAs[i]);

                std::filesystem::path outputPath = sampleDir /
                    (std::to_string(i) + ".png");
                writePNGImage(outputPath, frameRGBAs[i].data(), W, H, true);

            }
            appendRenderMetadata(std::filesystem::path("output") / "render_metadata.json", 
                                sampleDir.stem().string(), textureNames[randomTexIndex]);
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