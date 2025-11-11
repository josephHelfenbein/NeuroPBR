#include <io.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <iterator>
#include <map>
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

inline std::string escapeJsonString(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size() + 8);
    for (char c : input) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(c)));
                    escaped += buffer;
                } else {
                    escaped += c;
                }
                break;
        }
    }
    return escaped;
}

inline int hexValue(char c) {
    if (c >= '0' && c <= '9') {
        return static_cast<int>(c - '0');
    }
    if (c >= 'a' && c <= 'f') {
        return static_cast<int>(10 + (c - 'a'));
    }
    if (c >= 'A' && c <= 'F') {
        return static_cast<int>(10 + (c - 'A'));
    }
    return -1;
}

inline std::string unescapeJsonString(const std::string& input) {
    std::string result;
    result.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c != '\\') {
            result += c;
            continue;
        }

        if (i + 1 >= input.size()) {
            break;
        }

        char next = input[++i];
        switch (next) {
            case '"': result += '"'; break;
            case '\\': result += '\\'; break;
            case '/': result += '/'; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u': {
                if (i + 4 >= input.size()) {
                    break;
                }
                unsigned int codepoint = 0;
                bool valid = true;
                for (int k = 0; k < 4; ++k) {
                    int hv = hexValue(input[i + 1 + k]);
                    if (hv < 0) {
                        valid = false;
                        break;
                    }
                    codepoint = (codepoint << 4) | static_cast<unsigned int>(hv);
                }
                if (valid) {
                    if (codepoint <= 0x7F) {
                        result += static_cast<char>(codepoint);
                    } else if (codepoint <= 0x7FF) {
                        result += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    } else {
                        result += static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F));
                        result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                        result += static_cast<char>(0x80 | (codepoint & 0x3F));
                    }
                }
                i += 4;
                break;
            }
            default:
                result += next;
                break;
        }
    }
    return result;
}

inline bool isEscapedQuote(const std::string& text, size_t quoteIndex, size_t start) {
    if (quoteIndex == 0 || quoteIndex <= start) {
        return false;
    }
    size_t backslashCount = 0;
    size_t idx = quoteIndex;
    while (idx > start && text[idx - 1] == '\\') {
        ++backslashCount;
        --idx;
    }
    return (backslashCount % 2) == 1;
}

void parseExistingMetadata(const std::string& content, std::map<std::string, std::string>& entries) {
    std::string currentRender;
    size_t pos = 0;

    while (true) {
        size_t keyStart = content.find('"', pos);
        if (keyStart == std::string::npos) {
            break;
        }
        size_t keyEnd = content.find('"', keyStart + 1);
        if (keyEnd == std::string::npos) {
            break;
        }

        std::string key = unescapeJsonString(content.substr(keyStart + 1, keyEnd - keyStart - 1));
        pos = keyEnd + 1;

        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) {
            break;
        }
        pos = colon + 1;

        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
            ++pos;
        }
        if (pos >= content.size()) {
            break;
        }

        std::string value;
        if (content[pos] == '"') {
            size_t valueStart = pos + 1;
            size_t valueEnd = valueStart;
            while (valueEnd < content.size()) {
                if (content[valueEnd] == '"' && !isEscapedQuote(content, valueEnd, valueStart)) {
                    break;
                }
                ++valueEnd;
            }
            if (valueEnd >= content.size()) {
                break;
            }
            value = unescapeJsonString(content.substr(valueStart, valueEnd - valueStart));
            pos = valueEnd + 1;
        } else {
            size_t valueEnd = pos;
            while (valueEnd < content.size() && content[valueEnd] != ',' && content[valueEnd] != '}' && content[valueEnd] != ']') {
                ++valueEnd;
            }
            value = content.substr(pos, valueEnd - pos);
            pos = valueEnd;
        }

        if (key == "render") {
            currentRender = std::filesystem::path(value).filename().string();
        } else if (key == "material") {
            if (!currentRender.empty()) {
                entries[currentRender] = value;
                currentRender.clear();
            }
        }
    }
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
    const std::int64_t texelCount64 = static_cast<std::int64_t>(texelCount);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < texelCount64; ++i) {
        size_t idx = static_cast<size_t>(i);
        const unsigned char* src = rawData + idx * srcStride;
        float* dst = image.data.data() + idx * desiredChannels;

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

    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> rawPixels(texelCount * 4u);
    const std::int64_t texelCount64 = static_cast<std::int64_t>(texelCount);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < texelCount64; ++i) {
        size_t idx = static_cast<size_t>(i);
        const float4& pixel = frameData[idx];
        size_t dstIndex = idx * 4u;
        rawPixels[dstIndex + 0] = toByte(pixel.x);
        rawPixels[dstIndex + 1] = toByte(pixel.y);
        rawPixels[dstIndex + 2] = toByte(pixel.z);
        rawPixels[dstIndex + 3] = toByte(pixel.w);
    }

    std::string utf8Path = filePath.string();
    if (stbi_write_png(utf8Path.c_str(), width, height, 4, rawPixels.data(), width * 4) == 0) {
        throw std::runtime_error("Failed to write PNG image");
    }
    stbi_flip_vertically_on_write(0);
}

void appendRenderMetadata(const std::filesystem::path& metadataPath,
                          const std::string& renderFilename,
                          const std::string& materialName) {
    if (metadataPath.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(metadataPath.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("Failed to create metadata directory: " + ec.message());
        }
    }

    std::map<std::string, std::string> entries;

    if (std::filesystem::exists(metadataPath)) {
        std::ifstream in(metadataPath, std::ios::in);
        if (!in) {
            throw std::runtime_error("Failed to open metadata file for read: " + metadataPath.string());
        }
        std::string existingContent((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        parseExistingMetadata(existingContent, entries);
    }

    std::string sampleKey = std::filesystem::path(renderFilename).filename().string();
    entries[sampleKey] = materialName;

    std::ofstream out(metadataPath, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Failed to open metadata file for write: " + metadataPath.string());
    }

    out << "{\n";
    size_t idx = 0;
    for (const auto& [render, material] : entries) {
        out << "  \"" << escapeJsonString(render) << "\": \""
            << escapeJsonString(material) << "\"";
        if (idx + 1 < entries.size()) {
            out << ",";
        }
        out << "\n";
        ++idx;
    }
    out << "}\n";
}
