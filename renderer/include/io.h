#pragma once

#include <filesystem>
#include <vector>
#include <map>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>

struct ByteImage {
	int width = 0;
	int height = 0;
	int channels = 0;
	std::vector<uint8_t> data;
};

ByteImage loadPNGImage8(const std::filesystem::path& filePath, int desiredChannels = 3, bool flipY = true);

// Lightweight readability/size check for PNG files used during continue-mode scans
bool isPNGReadable(const std::filesystem::path& filePath);

bool readPNGSize(const std::filesystem::path& filePath, int& width, int& height);

void writePNGImage(const std::filesystem::path& filePath, const uint8_t* rgb, int width, int height);

void loadMetadata(const std::filesystem::path& metadataPath, std::map<std::string, std::string>& entries);
void saveMetadata(const std::filesystem::path& metadataPath, const std::map<std::string, std::string>& entries);

