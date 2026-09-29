#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

// About one equirect texel per cube face texel
constexpr unsigned kHDRWidthPerCubeFace = 4;

struct HDRImage {
    int width = 0;
    int height = 0;
    int sourceWidth = 0;
    int sourceHeight = 0;
    std::vector<float4> pixels;
};

// Wider images are downscaled while decoding; maxWidth <= 0 keeps full resolution
HDRImage loadHDRImage(const std::filesystem::path& path, int maxWidth = 0);

HDRImage decodeHDRImage(const uint8_t* data, size_t size, int maxWidth = 0);

int downscaledHeight(int srcWidth, int srcHeight, int dstWidth);

HDRImage downscaleHDRImage(const HDRImage& src, int maxWidth);
