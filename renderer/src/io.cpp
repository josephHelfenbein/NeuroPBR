#include "io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

constexpr float kByteToFloat = 1.0f / 255.0f;
constexpr float kFloatToByte = 255.0f;

inline uint8_t toByte(float value) {
    float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lround(clamped * kFloatToByte));
}

FloatImage loadPNGImage(const std::filesystem::path& filePath, int desiredChannels, bool flipY) {
    if (desiredChannels != 1 && desiredChannels != 3 && desiredChannels != 4) {
        throw std::invalid_argument("desiredChannels must be 1, 3, or 4");
    }

    stbi_set_flip_vertically_on_load(flipY ? 1 : 0);

    int width = 0;
    int height = 0;
    int actualChannels = 0;
    std::string utf8Path = filePath.string();

    unsigned char* rawData = stbi_load(utf8Path.c_str(), &width, &height, &actualChannels, 0);
    if (!rawData) {
        const char* reason = stbi_failure_reason();
        stbi_set_flip_vertically_on_load(0);
        throw std::runtime_error(reason ? reason : "Failed to load PNG image");
    }

    if (actualChannels <= 0) {
        stbi_image_free(rawData);
        stbi_set_flip_vertically_on_load(0);
        throw std::runtime_error("PNG returned zero channels");
    }

    FloatImage image;
    image.width = width;
    image.height = height;
    image.channels = desiredChannels;
    image.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * desiredChannels);

    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t srcStride = static_cast<size_t>(actualChannels);

    for (size_t i = 0; i < texelCount; ++i) {
        const unsigned char* src = rawData + i * srcStride;
        float* dst = image.data.data() + i * desiredChannels;

        if (desiredChannels == 1) {
            // Roughness/metallic read the red channel from RGB(A) textures
            dst[0] = src[0] * kByteToFloat;
            continue;
        }

        dst[0] = src[0] * kByteToFloat;
        dst[1] = (actualChannels > 1 ? src[1] : src[0]) * kByteToFloat;
        dst[2] = (actualChannels > 2 ? src[2] : src[0]) * kByteToFloat;

        if (desiredChannels == 4) {
            dst[3] = (actualChannels > 3 ? src[3] * kByteToFloat : 1.0f);
        }
    }

    stbi_image_free(rawData);
    stbi_set_flip_vertically_on_load(0);
    return image;
}

void writePNGImage(const std::filesystem::path& filePath, const float4* frameData, 
                    int width, int height, bool flipY) {
    if (frameData == nullptr) {
        throw std::invalid_argument("frameData cannot be null");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }

    stbi_flip_vertically_on_write(flipY ? 1 : 0);

    std::vector<uint8_t> rawPixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float4& pixel = frameData[static_cast<size_t>(y) * width + x];
            size_t dstIndex = (static_cast<size_t>(y) * width + x) * 4;
            rawPixels[dstIndex + 0] = toByte(pixel.x);
            rawPixels[dstIndex + 1] = toByte(pixel.y);
            rawPixels[dstIndex + 2] = toByte(pixel.z);
            rawPixels[dstIndex + 3] = toByte(pixel.w);
        }
    }

    std::string utf8Path = filePath.string();
    if (stbi_write_png(utf8Path.c_str(), width, height, 4, rawPixels.data(), width * 4) == 0) {
        throw std::runtime_error("Failed to write PNG image");
    }
    stbi_flip_vertically_on_write(0);
}
