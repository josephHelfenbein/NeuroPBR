#include "renderer.cpp"
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

        auto environments = loadEnvironmentCubemaps(hdrDir, kEnvFaceSize, kIrradianceFaceSize, kSpecularSamples, kDiffuseSamples);

        std::cout << "Generated cubemaps: " << environments.size() << std::endl;

        BRDFLookupTable brdfLut = createBRDFLUT(512);
        loadBRDFLUT(brdfLut);

        std::cout << "Precomputation complete." << std::endl;

        return 0;
    } catch (const CudaError& e) {
        std::cerr << "CUDA Failure: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}