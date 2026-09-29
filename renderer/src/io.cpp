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
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <fpng/src/fpng.h>

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
    static const std::regex kEntry(R"(("(?:[^"\\]|\\.)+")\s*:\s*("(?:[^"\\]|\\.)+"))");

    for (std::sregex_iterator it(content.begin(), content.end(), kEntry), end; it != end; ++it) {
        const std::string rawKey = (*it)[1].str();
        const std::string rawValue = (*it)[2].str();

        if (rawKey.size() < 2 || rawValue.size() < 2) {
            continue;
        }

        std::string key = unescapeJsonString(rawKey.substr(1, rawKey.size() - 2));
        std::string value = unescapeJsonString(rawValue.substr(1, rawValue.size() - 2));
        entries[key] = value;
    }
}

ByteImage loadPNGImage8(const std::filesystem::path& filePath, int desiredChannels, bool flipY) {
    if (desiredChannels != 1 && desiredChannels != 3 && desiredChannels != 4) {
        throw std::invalid_argument("desiredChannels must be 1, 3, or 4");
    }

    int width = 0;
    int height = 0;
    int actualChannels = 0;
    std::string utf8Path = filePath.string();

    // Flip here rather than through stb's global flip flag
    unsigned char* rawData = stbi_load(utf8Path.c_str(), &width, &height, &actualChannels, 0);
    if (!rawData) {
        const char* reason = stbi_failure_reason();
        std::string msg = "Failed to load PNG image: " + utf8Path;
        if (reason) {
            msg += " (Reason: ";
            msg += reason;
            msg += ")";
        }
        throw std::runtime_error(msg);
    }

    if (actualChannels <= 0) {
        stbi_image_free(rawData);
        throw std::runtime_error("PNG returned zero channels");
    }

    ByteImage image;
    image.width = width;
    image.height = height;
    image.channels = desiredChannels;
    image.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * desiredChannels);

    const size_t rowTexels = static_cast<size_t>(width);
    const size_t srcStride = static_cast<size_t>(actualChannels);
    const size_t dstStride = static_cast<size_t>(desiredChannels);

    for (int y = 0; y < height; ++y) {
        const int srcY = flipY ? (height - 1 - y) : y;
        const unsigned char* srcRow = rawData + static_cast<size_t>(srcY) * rowTexels * srcStride;
        std::uint8_t* dstRow = image.data.data() + static_cast<size_t>(y) * rowTexels * dstStride;

        for (size_t x = 0; x < rowTexels; ++x) {
            const unsigned char* src = srcRow + x * srcStride;
            std::uint8_t* dst = dstRow + x * dstStride;

            if (desiredChannels == 1) {
                // Roughness/metallic read the red channel from RGB(A) textures
                dst[0] = src[0];
                continue;
            }

            dst[0] = src[0];
            dst[1] = actualChannels > 1 ? src[1] : src[0];
            dst[2] = actualChannels > 2 ? src[2] : src[0];

            if (desiredChannels == 4) {
                dst[3] = actualChannels > 3 ? src[3] : 255;
            }
        }
    }

    stbi_image_free(rawData);
    return image;
}

bool isPNGReadable(const std::filesystem::path& filePath) {
    std::error_code ec;
    auto sz = std::filesystem::file_size(filePath, ec);
    if (ec || sz < 16) { // trivially reject zero/very small files
        return false;
    }

    int w = 0, h = 0, c = 0;
    if (stbi_info(filePath.string().c_str(), &w, &h, &c) == 0) {
        return false;
    }
    return (w > 0 && h > 0);
}

bool readPNGSize(const std::filesystem::path& filePath, int& width, int& height) {
    int w = 0, h = 0, c = 0;
    if (stbi_info(filePath.string().c_str(), &w, &h, &c) == 0 || w <= 0 || h <= 0) {
        return false;
    }
    width = w;
    height = h;
    return true;
}

void writePNGImage(const std::filesystem::path& filePath, const uint8_t* rgb,
                   int width, int height) {
    if (rgb == nullptr) {
        throw std::invalid_argument("rgb cannot be null");
    }
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid image dimensions");
    }

    // Write atomically
    auto tmpPath = filePath;
    tmpPath += ".tmp";

    std::string utf8Tmp = tmpPath.string();
    int attempts = 3;
    for (int attempt = 1; attempt <= attempts; ++attempt) {
        if (fpng::fpng_encode_image_to_file(utf8Tmp.c_str(), rgb,
                                            static_cast<uint32_t>(width),
                                            static_cast<uint32_t>(height), 3)) {
            std::error_code ec;
            std::filesystem::rename(tmpPath, filePath, ec);
            if (ec) {
                std::filesystem::remove(tmpPath);
                throw std::runtime_error("Failed to rename temp PNG: " + ec.message());
            }
            return;
        }

        if (attempt == attempts) {
            std::filesystem::remove(tmpPath);
            throw std::runtime_error("Failed to write PNG image after retries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50 * attempt));
    }
}

void loadMetadata(const std::filesystem::path& metadataPath, std::map<std::string, std::string>& entries) {
    if (std::filesystem::exists(metadataPath)) {
        std::ifstream in(metadataPath, std::ios::in);
        if (in) {
            std::string existingContent((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            parseExistingMetadata(existingContent, entries);
        }
    }
}

void saveMetadata(const std::filesystem::path& metadataPath, const std::map<std::string, std::string>& entries) {
    if (metadataPath.has_parent_path()) {
        std::filesystem::create_directories(metadataPath.parent_path());
    }

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
