#include <hdr_image.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Area-weighted box filter, fed one row at a time. Integer overlaps keep integer ratios exact.
class BoxDownscaler {
public:
    BoxDownscaler(int srcWidth, int srcHeight, int dstWidth, int dstHeight, bool allowFastPath = true)
        : srcWidth_(srcWidth), srcHeight_(srcHeight), dstWidth_(dstWidth), dstHeight_(dstHeight) {
        if (allowFastPath && srcWidth % dstWidth == 0) {
            columnRatio_ = srcWidth / dstWidth;
        }
        if (allowFastPath && srcHeight % dstHeight == 0) {
            rowRatio_ = srcHeight / dstHeight;
        }
        if (columnRatio_ == 0) {
            buildColumnTaps();
        }
        rowSums_.resize(static_cast<size_t>(dstWidth));
        pixels_.assign(static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight),
                       float4{0.0f, 0.0f, 0.0f, 0.0f});
    }

    void addRow(int y, const float4* row) {
        filterRow(row);

        if (rowRatio_ > 0) {
            accumulate(y / rowRatio_, 1.0f, false);
            return;
        }
        const int64_t srcH = srcHeight_;
        const int64_t dstH = dstHeight_;
        const int64_t rowBegin = y * dstH;
        const int64_t rowEnd = rowBegin + dstH;
        const int firstOut = static_cast<int>(rowBegin / srcH);
        const int lastOut = static_cast<int>((rowEnd - 1) / srcH);
        for (int oy = firstOut; oy <= lastOut; ++oy) {
            int64_t overlap = std::min(rowEnd, (oy + 1) * srcH) - std::max(rowBegin, oy * srcH);
            accumulate(oy, static_cast<float>(overlap) / static_cast<float>(dstH), true);
        }
    }

    std::vector<float4> finish() {
        const float scale = static_cast<float>((static_cast<double>(dstWidth_) * dstHeight_) /
                                               (static_cast<double>(srcWidth_) * srcHeight_));
        for (float4& p : pixels_) {
            p.x *= scale;
            p.y *= scale;
            p.z *= scale;
            p.w = 1.0f;
        }
        return std::move(pixels_);
    }

private:
    struct ColumnTaps {
        size_t begin = 0;
        size_t count = 0;
    };
    struct Tap {
        int column = 0;
        float weight = 0.0f;
    };

    void buildColumnTaps() {
        const int64_t srcW = srcWidth_;
        const int64_t dstW = dstWidth_;
        columns_.resize(static_cast<size_t>(dstWidth_));
        for (int64_t ox = 0; ox < dstW; ++ox) {
            const int64_t outBegin = ox * srcW;
            const int64_t outEnd = outBegin + srcW;
            ColumnTaps& c = columns_[static_cast<size_t>(ox)];
            c.begin = taps_.size();
            for (int64_t x = outBegin / dstW; x <= (outEnd - 1) / dstW; ++x) {
                int64_t overlap = std::min(outEnd, (x + 1) * dstW) - std::max(outBegin, x * dstW);
                taps_.push_back({static_cast<int>(x), static_cast<float>(overlap) / static_cast<float>(dstW)});
            }
            c.count = taps_.size() - c.begin;
        }
    }

    void filterRow(const float4* row) {
        if (columnRatio_ > 0) {
            for (int ox = 0; ox < dstWidth_; ++ox) {
                const float4* src = row + static_cast<size_t>(ox) * columnRatio_;
                float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
                for (int i = 0; i < columnRatio_; ++i) {
                    sum.x += src[i].x;
                    sum.y += src[i].y;
                    sum.z += src[i].z;
                }
                rowSums_[static_cast<size_t>(ox)] = sum;
            }
            return;
        }
        for (int ox = 0; ox < dstWidth_; ++ox) {
            const ColumnTaps& c = columns_[static_cast<size_t>(ox)];
            float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t t = c.begin; t < c.begin + c.count; ++t) {
                const float4& s = row[taps_[t].column];
                const float w = taps_[t].weight;
                sum.x += w * s.x;
                sum.y += w * s.y;
                sum.z += w * s.z;
            }
            rowSums_[static_cast<size_t>(ox)] = sum;
        }
    }

    void accumulate(int oy, float weight, bool weighted) {
        float4* dst = pixels_.data() + static_cast<size_t>(oy) * static_cast<size_t>(dstWidth_);
        if (!weighted) {
            for (int ox = 0; ox < dstWidth_; ++ox) {
                dst[ox].x += rowSums_[ox].x;
                dst[ox].y += rowSums_[ox].y;
                dst[ox].z += rowSums_[ox].z;
            }
            return;
        }
        for (int ox = 0; ox < dstWidth_; ++ox) {
            dst[ox].x += weight * rowSums_[ox].x;
            dst[ox].y += weight * rowSums_[ox].y;
            dst[ox].z += weight * rowSums_[ox].z;
        }
    }

    int srcWidth_;
    int srcHeight_;
    int dstWidth_;
    int dstHeight_;
    int columnRatio_ = 0; // nonzero for integer ratios
    int rowRatio_ = 0;
    std::vector<ColumnTaps> columns_;
    std::vector<Tap> taps_;
    std::vector<float4> rowSums_;
    std::vector<float4> pixels_;
};

bool needsDownscale(int width, int maxWidth) {
    return maxWidth > 0 && width > maxWidth;
}

struct ByteReader {
    const uint8_t* data;
    size_t size;
    size_t pos = 0;

    size_t remaining() const { return size - pos; }
    bool atEnd() const { return pos >= size; }

    std::string readLine() {
        if (atEnd()) {
            return std::string();
        }
        const uint8_t* begin = data + pos;
        const uint8_t* newline = static_cast<const uint8_t*>(std::memchr(begin, '\n', remaining()));
        size_t length = newline ? static_cast<size_t>(newline - begin) : remaining();
        pos += length + (newline ? 1u : 0u);
        std::string line(reinterpret_cast<const char*>(begin), length);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return line;
    }

    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error(std::string(message) + " (at byte " + std::to_string(pos) +
                                 " of " + std::to_string(size) + ")");
    }

    void require(size_t count, const char* message) const {
        if (remaining() < count) {
            fail(message);
        }
    }
};

struct RGBEScaleTable {
    float scale[256];
    RGBEScaleTable() {
        scale[0] = 0.0f;
        for (int e = 1; e < 256; ++e) {
            scale[e] = std::ldexp(1.0f, e - (128 + 8));
        }
    }
};

void convertScanline(const unsigned char* scanline, int width, float4* dst) {
    static const RGBEScaleTable table;
    for (int x = 0; x < width; ++x) {
        unsigned char r = scanline[x + 0 * width];
        unsigned char g = scanline[x + 1 * width];
        unsigned char b = scanline[x + 2 * width];
        unsigned char e = scanline[x + 3 * width];
        if (e) {
            float f = table.scale[e];
            dst[x].x = r * f;
            dst[x].y = g * f;
            dst[x].z = b * f;
            dst[x].w = 1.0f;
        } else {
            dst[x].x = dst[x].y = dst[x].z = 0.0f;
            dst[x].w = 1.0f;
        }
    }
}

void decodeScanline(ByteReader& in, int width, unsigned char* scanline) {
    in.require(4, "Unexpected EOF reading HDRI scanline header");
    const uint8_t* scanlineHeader = in.data + in.pos;
    in.pos += 4;

    bool rle = false;
    if (scanlineHeader[0] == 2 && scanlineHeader[1] == 2) {
        int scanlineWidth = (int(scanlineHeader[2]) << 8) | int(scanlineHeader[3]);
        if (scanlineWidth == width) {
            rle = true;
        }
    }

    if (!rle) {
        size_t w = static_cast<size_t>(width);
        scanline[0] = scanlineHeader[0];
        scanline[w] = scanlineHeader[1];
        scanline[2 * w] = scanlineHeader[2];
        scanline[3 * w] = scanlineHeader[3];
        size_t remaining = (w - 1) * 4u;
        in.require(remaining, "Unexpected EOF reading legacy HDRI scanline");
        std::memcpy(scanline + 4, in.data + in.pos, remaining);
        in.pos += remaining;
        for (size_t i = 0; i < remaining / 4u; ++i) {
            scanline[(i + 1) + 0 * w] = scanline[4 + i * 4 + 0];
            scanline[(i + 1) + 1 * w] = scanline[4 + i * 4 + 1];
            scanline[(i + 1) + 2 * w] = scanline[4 + i * 4 + 2];
            scanline[(i + 1) + 3 * w] = scanline[4 + i * 4 + 3];
        }
        return;
    }

    // Runs that overshoot the channel still consume their bytes but are clipped
    for (int channel = 0; channel < 4; ++channel) {
        unsigned char* dst = scanline + static_cast<size_t>(channel) * static_cast<size_t>(width);
        int index = 0;
        while (index < width) {
            in.require(1, "Unexpected EOF while decoding HDRI RLE");
            unsigned char code = in.data[in.pos++];
            if (code > 128) {
                int count = code - 128;
                in.require(1, "Unexpected EOF in HDRI RLE run");
                unsigned char value = in.data[in.pos++];
                std::memset(dst + index, value, static_cast<size_t>(std::min(count, width - index)));
                index += count;
            } else {
                int count = code;
                if (count == 0) {
                    in.fail("Zero-length literal in HDRI RLE");
                }
                in.require(static_cast<size_t>(count), "Unexpected EOF in HDRI RLE literal");
                std::memcpy(dst + index, in.data + in.pos, static_cast<size_t>(std::min(count, width - index)));
                in.pos += static_cast<size_t>(count);
                index += count;
            }
        }
    }
}

} // namespace

int downscaledHeight(int srcWidth, int srcHeight, int dstWidth) {
    double height = std::round(static_cast<double>(srcHeight) * dstWidth / srcWidth);
    return static_cast<int>(std::max(1.0, std::min(static_cast<double>(srcHeight), height)));
}

HDRImage decodeHDRImage(const uint8_t* data, size_t size, int maxWidth) {
    ByteReader in{data, size};

    std::string header = in.readLine();
    if (header.rfind("#?", 0) != 0) {
        throw std::runtime_error("Invalid HDRI header (missing #?): " + header);
    }

    for (;;) {
        if (in.atEnd()) {
            throw std::runtime_error("Unexpected EOF while reading HDRI header");
        }
        std::string line = in.readLine();
        if (line.empty()) {
            break;
        }
    }

    std::string resolution = in.readLine();
    if (resolution.empty()) {
        throw std::runtime_error("Missing resolution line in HDRI");
    }

    int width = 0;
    int height = 0;
    char axis1 = 0, axis2 = 0;
    char sign1 = 0, sign2 = 0;
    if (sscanf(resolution.c_str(), "%c%c %d %c%c %d", &sign1, &axis1, &height, &sign2, &axis2, &width) != 6) {
        throw std::runtime_error("Failed to parse HDRI resolution string: " + resolution);
    }
    if ((axis1 != 'Y' && axis1 != 'y') || (axis2 != 'X' && axis2 != 'x')) {
        throw std::runtime_error("Only -Y +X orientation is supported, got: " + resolution);
    }
    if (sign1 != '-' || sign2 != '+') {
        throw std::runtime_error("Unsupported HDRI orientation: " + resolution);
    }
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("HDRI has invalid dimensions: " + resolution);
    }

    const size_t rowPixels = static_cast<size_t>(width);
    std::vector<unsigned char> scanline(rowPixels * 4u);

    HDRImage image;
    image.sourceWidth = width;
    image.sourceHeight = height;

    if (!needsDownscale(width, maxWidth)) {
        image.width = width;
        image.height = height;
        image.pixels.resize(rowPixels * static_cast<size_t>(height));
        for (int y = 0; y < height; ++y) {
            decodeScanline(in, width, scanline.data());
            convertScanline(scanline.data(), width, image.pixels.data() + static_cast<size_t>(y) * rowPixels);
        }
        return image;
    }

    image.width = maxWidth;
    image.height = downscaledHeight(width, height, maxWidth);
    BoxDownscaler downscaler(width, height, image.width, image.height);
    std::vector<float4> row(rowPixels);
    for (int y = 0; y < height; ++y) {
        decodeScanline(in, width, scanline.data());
        convertScanline(scanline.data(), width, row.data());
        downscaler.addRow(y, row.data());
    }
    image.pixels = downscaler.finish();
    return image;
}

HDRImage loadHDRImage(const std::filesystem::path& path, int maxWidth) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open HDRI file: " + path.string());
    }
    std::streamoff size = file.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to read HDRI file: " + path.string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Failed to read HDRI file: " + path.string());
    }
    file.close();

    return decodeHDRImage(bytes.data(), bytes.size(), maxWidth);
}

HDRImage downscaleHDRImage(const HDRImage& src, int maxWidth) {
    if (!needsDownscale(src.width, maxWidth)) {
        return src;
    }
    HDRImage out;
    out.sourceWidth = src.sourceWidth;
    out.sourceHeight = src.sourceHeight;
    out.width = maxWidth;
    out.height = downscaledHeight(src.width, src.height, maxWidth);
    BoxDownscaler downscaler(src.width, src.height, out.width, out.height);
    for (int y = 0; y < src.height; ++y) {
        downscaler.addRow(y, src.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(src.width));
    }
    out.pixels = downscaler.finish();
    return out;
}
