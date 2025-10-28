#include "renderer.h"
#include "io.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main() {
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

        std::cout << "Precomputation complete. Starting rendering." << std::endl;

        std::vector<float4> frameRGBA;
        FloatImage dAlbedo = loadPNGImage(std::filesystem::path("assets") / "textures" / "albedo.png", 3, true);
        FloatImage dNormal = loadPNGImage(std::filesystem::path("assets") / "textures" / "normal.png", 3, true);
        FloatImage dRoughness = loadPNGImage(std::filesystem::path("assets") / "textures" / "roughness.png", 1, true);
        FloatImage dMetallic = loadPNGImage(std::filesystem::path("assets") / "textures" / "metallic.png", 1, true);
        const int W = dAlbedo.width;
        const int H = dAlbedo.height;

        if (dNormal.width != W || dNormal.height != H ||
            dRoughness.width != W || dRoughness.height != H ||
            dMetallic.width != W || dMetallic.height != H) {
            throw std::runtime_error("All texture maps must share the same dimensions.");
        }

        if (dAlbedo.channels != 3 || dNormal.channels != 3 ||
            dRoughness.channels != 1 || dMetallic.channels != 1) {
            throw std::runtime_error("Unexpected channel count in texture maps. Expected RGB for albedo/normal and single channel for roughness/metallic.");
        }

        renderPlane(environments[0], brdfLut,
                    dAlbedo.data.data(), dNormal.data.data(),
                    dRoughness.data.data(), dMetallic.data.data(),
                    W, H, frameRGBA);

        std::filesystem::path outputPath = std::filesystem::path("output") / "renderedPlane.png";
        std::filesystem::create_directories(outputPath.parent_path());
        writePNGImage(outputPath, frameRGBA.data(), W, H, true);

        std::cout << "Rendering complete. Output saved to " << outputPath << std::endl;
        
        return 0;
    } catch (const CudaError& e) {
        std::cerr << "CUDA Failure: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}