#pragma once

#include <filesystem>
#include <vector>
#include <cuda_runtime.h>

struct FloatImage {
	int width = 0;
	int height = 0;
	int channels = 0;
	std::vector<float> data;
};

FloatImage loadPNGImage(const std::filesystem::path& filePath, int desiredChannels = 3, bool flipY = true);

void writePNGImage(const std::filesystem::path& filePath, const float4* frameData, int width, int height, bool flipY = true);

void appendRenderMetadata(const std::filesystem::path& metadataPath,
                          const std::string& renderFilename,
                          const std::string& materialName);
