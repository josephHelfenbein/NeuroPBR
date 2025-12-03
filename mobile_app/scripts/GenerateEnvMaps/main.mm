// GenerateEnvMaps - Command-line tool to precompute environment maps using Metal
// Uses the same Metal shaders as the iOS app

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

#include <fstream>
#include <vector>
#include <string>
#include <cmath>

// Include the EnvironmentPrefilter from the iOS project
#import "../../ios/MetalRenderer/EnvironmentPrefilter.h"

#pragma mark - HDR Loading

struct HDRPixel {
    float r, g, b, a;
};

struct HDRImage {
    int width = 0;
    int height = 0;
    std::vector<HDRPixel> pixels;
};

HDRImage LoadHDR(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open HDRI file");
    }

    auto readLine = [&file]() {
        std::string line;
        std::getline(file, line);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return line;
    };

    std::string header = readLine();
    if (header.rfind("#?", 0) != 0) {
        throw std::runtime_error("Invalid HDRI header");
    }

    for (;;) {
        if (!file) {
            throw std::runtime_error("Unexpected EOF while reading HDRI header");
        }
        std::string line = readLine();
        if (line.empty()) break;
    }

    std::string resolution = readLine();
    if (resolution.empty()) {
        throw std::runtime_error("Missing resolution line in HDRI");
    }

    int width = 0, height = 0;
    char axis1 = 0, axis2 = 0, sign1 = 0, sign2 = 0;
    if (sscanf(resolution.c_str(), "%c%c %d %c%c %d", &sign1, &axis1, &height, &sign2, &axis2, &width) != 6) {
        throw std::runtime_error("Failed to parse HDRI resolution string");
    }

    HDRImage image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));

    std::vector<unsigned char> scanline(static_cast<size_t>(width) * 4u);

    for (int y = 0; y < height; ++y) {
        unsigned char scanlineHeader[4];
        if (!file.read(reinterpret_cast<char*>(scanlineHeader), 4)) {
            throw std::runtime_error("Unexpected EOF reading HDRI scanline header");
        }

        bool rle = false;
        if (scanlineHeader[0] == 2 && scanlineHeader[1] == 2) {
            int scanlineWidth = (int(scanlineHeader[2]) << 8) | int(scanlineHeader[3]);
            if (scanlineWidth == width) {
                rle = true;
            }
        }

        if (!rle) {
            scanline[0] = scanlineHeader[0];
            scanline[width] = scanlineHeader[1];
            scanline[2 * width] = scanlineHeader[2];
            scanline[3 * width] = scanlineHeader[3];
            size_t remaining = static_cast<size_t>(width - 1) * 4u;
            if (!file.read(reinterpret_cast<char*>(scanline.data() + 4), static_cast<std::streamsize>(remaining))) {
                throw std::runtime_error("Unexpected EOF reading legacy HDRI scanline");
            }
            for (size_t i = 0; i < remaining / 4u; ++i) {
                scanline[(i + 1) + 0 * width] = scanline[4 + i * 4 + 0];
                scanline[(i + 1) + 1 * width] = scanline[4 + i * 4 + 1];
                scanline[(i + 1) + 2 * width] = scanline[4 + i * 4 + 2];
                scanline[(i + 1) + 3 * width] = scanline[4 + i * 4 + 3];
            }
        } else {
            for (int channel = 0; channel < 4; ++channel) {
                int index = 0;
                while (index < width) {
                    unsigned char code;
                    file.read(reinterpret_cast<char*>(&code), 1);
                    if (!file) {
                        throw std::runtime_error("Unexpected EOF while decoding HDRI RLE");
                    }
                    if (code > 128) {
                        int count = code - 128;
                        unsigned char value;
                        file.read(reinterpret_cast<char*>(&value), 1);
                        for (int i = 0; i < count; ++i) {
                            scanline[channel * width + index++] = value;
                        }
                    } else {
                        int count = code;
                        file.read(reinterpret_cast<char*>(scanline.data() + channel * width + index), count);
                        index += count;
                    }
                }
            }
        }

        for (int x = 0; x < width; ++x) {
            unsigned char r = scanline[x + 0 * width];
            unsigned char g = scanline[x + 1 * width];
            unsigned char b = scanline[x + 2 * width];
            unsigned char e = scanline[x + 3 * width];

            HDRPixel& dst = image.pixels[static_cast<size_t>(y) * width + x];
            if (e) {
                float f = std::ldexp(1.0f, int(e) - (128 + 8));
                dst.r = r * f;
                dst.g = g * f;
                dst.b = b * f;
                dst.a = 1.0f;
            } else {
                dst.r = dst.g = dst.b = 0.0f;
                dst.a = 1.0f;
            }
        }
    }

    return image;
}

#pragma mark - KTX Writing

static const uint8_t KTX_IDENTIFIER[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

// OpenGL constants
static const uint32_t GL_RGBA = 0x1908;
static const uint32_t GL_HALF_FLOAT = 0x140B;
static const uint32_t GL_RGBA16F = 0x881A;

uint16_t floatToHalf(float f) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = bits & 0x7FFFFF;
    
    if (exp == 128) { // Inf or NaN
        return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
    }
    if (exp > 15) { // Overflow
        return (sign << 15) | 0x7C00;
    }
    if (exp < -14) { // Underflow
        if (exp < -24) return sign << 15;
        mant = (mant | 0x800000) >> (1 - 14 - exp);
        return (sign << 15) | (mant >> 13);
    }
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

bool saveCubemapKTX(id<MTLTexture> texture, const std::string& path) {
    NSUInteger size = texture.width;
    NSUInteger mipCount = texture.mipmapLevelCount;
    MTLPixelFormat srcFormat = texture.pixelFormat;
    
    // Determine source format properties
    bool isFloat32 = (srcFormat == MTLPixelFormatRGBA32Float);
    NSUInteger srcBytesPerPixel = isFloat32 ? 16 : 8; // RGBA32Float = 16, RGBA16Float = 8
    NSUInteger dstBytesPerPixel = 8; // Always output as RGBA16Float
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        NSLog(@"Failed to open file for writing: %s", path.c_str());
        return false;
    }
    
    // Write identifier
    file.write(reinterpret_cast<const char*>(KTX_IDENTIFIER), 12);
    
    // Write header - always output as RGBA16F
    uint32_t endianness = 0x04030201;
    uint32_t glType = GL_HALF_FLOAT;
    uint32_t glTypeSize = 2;
    uint32_t glFormat = GL_RGBA;
    uint32_t glInternalFormat = GL_RGBA16F;
    uint32_t glBaseInternalFormat = GL_RGBA;
    uint32_t pixelWidth = (uint32_t)size;
    uint32_t pixelHeight = (uint32_t)size;
    uint32_t pixelDepth = 0;
    uint32_t numberOfArrayElements = 0;
    uint32_t numberOfFaces = 6;
    uint32_t numberOfMipmapLevels = (uint32_t)mipCount;
    uint32_t bytesOfKeyValueData = 0;
    
    file.write(reinterpret_cast<const char*>(&endianness), 4);
    file.write(reinterpret_cast<const char*>(&glType), 4);
    file.write(reinterpret_cast<const char*>(&glTypeSize), 4);
    file.write(reinterpret_cast<const char*>(&glFormat), 4);
    file.write(reinterpret_cast<const char*>(&glInternalFormat), 4);
    file.write(reinterpret_cast<const char*>(&glBaseInternalFormat), 4);
    file.write(reinterpret_cast<const char*>(&pixelWidth), 4);
    file.write(reinterpret_cast<const char*>(&pixelHeight), 4);
    file.write(reinterpret_cast<const char*>(&pixelDepth), 4);
    file.write(reinterpret_cast<const char*>(&numberOfArrayElements), 4);
    file.write(reinterpret_cast<const char*>(&numberOfFaces), 4);
    file.write(reinterpret_cast<const char*>(&numberOfMipmapLevels), 4);
    file.write(reinterpret_cast<const char*>(&bytesOfKeyValueData), 4);
    
    // Read back texture data and write to file
    for (NSUInteger mip = 0; mip < mipCount; ++mip) {
        NSUInteger mipSize = MAX(1, size >> mip);
        NSUInteger srcBytesPerRow = mipSize * srcBytesPerPixel;
        NSUInteger dstBytesPerRow = mipSize * dstBytesPerPixel;
        // Align bytesPerRow to 256-byte boundary to avoid AGX warnings
        NSUInteger alignedSrcBytesPerRow = ((srcBytesPerRow + 255) / 256) * 256;
        NSUInteger dstBytesPerFace = mipSize * mipSize * dstBytesPerPixel;
        uint32_t imageSize = (uint32_t)(dstBytesPerFace * 6);
        
        file.write(reinterpret_cast<const char*>(&imageSize), 4);
        
        std::vector<uint8_t> alignedFaceData(alignedSrcBytesPerRow * mipSize);
        std::vector<uint8_t> faceData(dstBytesPerFace);
        
        for (NSUInteger face = 0; face < 6; ++face) {
            // Get texture data with aligned row stride
            [texture getBytes:alignedFaceData.data()
                  bytesPerRow:alignedSrcBytesPerRow
                bytesPerImage:alignedSrcBytesPerRow * mipSize
                   fromRegion:MTLRegionMake2D(0, 0, mipSize, mipSize)
                  mipmapLevel:mip
                        slice:face];
            
            // Convert and copy to output buffer
            for (NSUInteger row = 0; row < mipSize; ++row) {
                const uint8_t* srcRow = alignedFaceData.data() + row * alignedSrcBytesPerRow;
                uint8_t* dstRow = faceData.data() + row * dstBytesPerRow;
                
                if (isFloat32) {
                    // Convert RGBA32Float to RGBA16Float
                    const float* srcPixels = reinterpret_cast<const float*>(srcRow);
                    uint16_t* dstPixels = reinterpret_cast<uint16_t*>(dstRow);
                    for (NSUInteger x = 0; x < mipSize; ++x) {
                        dstPixels[x * 4 + 0] = floatToHalf(srcPixels[x * 4 + 0]);
                        dstPixels[x * 4 + 1] = floatToHalf(srcPixels[x * 4 + 1]);
                        dstPixels[x * 4 + 2] = floatToHalf(srcPixels[x * 4 + 2]);
                        dstPixels[x * 4 + 3] = floatToHalf(srcPixels[x * 4 + 3]);
                    }
                } else {
                    // Already RGBA16Float, just copy
                    memcpy(dstRow, srcRow, dstBytesPerRow);
                }
            }
            
            file.write(reinterpret_cast<const char*>(faceData.data()), dstBytesPerFace);
        }
        
        // Padding to 4-byte boundary
        uint32_t padding = (4 - (imageSize % 4)) % 4;
        if (padding > 0) {
            uint8_t zeros[4] = {0, 0, 0, 0};
            file.write(reinterpret_cast<const char*>(zeros), padding);
        }
    }
    
    return true;
}

bool saveBRDFLUT(id<MTLTexture> texture, const std::string& path) {
    NSUInteger size = texture.width;
    NSUInteger bytesPerPixel = 4; // RG16F
    NSUInteger bytesPerRow = size * bytesPerPixel;
    // Align bytesPerRow to 256-byte boundary to avoid AGX warnings
    NSUInteger alignedBytesPerRow = ((bytesPerRow + 255) / 256) * 256;
    
    std::vector<uint8_t> alignedData(alignedBytesPerRow * size);
    std::vector<uint8_t> data(size * size * bytesPerPixel);
    
    [texture getBytes:alignedData.data()
          bytesPerRow:alignedBytesPerRow
           fromRegion:MTLRegionMake2D(0, 0, size, size)
          mipmapLevel:0];
    
    // Copy to tightly packed buffer
    for (NSUInteger row = 0; row < size; ++row) {
        memcpy(data.data() + row * bytesPerRow,
               alignedData.data() + row * alignedBytesPerRow,
               bytesPerRow);
    }
    
    // Save as KTX format (2D texture, single mip level)
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        NSLog(@"Failed to open file for writing: %s", path.c_str());
        return false;
    }
    
    // KTX header (same structure as cubemaps)
    uint8_t identifier[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    file.write(reinterpret_cast<const char*>(identifier), 12);
    
    uint32_t endianness = 0x04030201;
    file.write(reinterpret_cast<const char*>(&endianness), 4);
    
    // OpenGL format for RG16F
    uint32_t glType = 0x140B;  // GL_HALF_FLOAT
    uint32_t glTypeSize = 2;
    uint32_t glFormat = 0x8227;  // GL_RG
    uint32_t glInternalFormat = 0x822F;  // GL_RG16F
    uint32_t glBaseInternalFormat = 0x8227;  // GL_RG
    
    file.write(reinterpret_cast<const char*>(&glType), 4);
    file.write(reinterpret_cast<const char*>(&glTypeSize), 4);
    file.write(reinterpret_cast<const char*>(&glFormat), 4);
    file.write(reinterpret_cast<const char*>(&glInternalFormat), 4);
    file.write(reinterpret_cast<const char*>(&glBaseInternalFormat), 4);
    
    uint32_t pixelWidth = (uint32_t)size;
    uint32_t pixelHeight = (uint32_t)size;
    uint32_t pixelDepth = 0;
    uint32_t numberOfArrayElements = 0;
    uint32_t numberOfFaces = 1;  // Not a cubemap
    uint32_t numberOfMipmapLevels = 1;
    uint32_t bytesOfKeyValueData = 0;
    
    file.write(reinterpret_cast<const char*>(&pixelWidth), 4);
    file.write(reinterpret_cast<const char*>(&pixelHeight), 4);
    file.write(reinterpret_cast<const char*>(&pixelDepth), 4);
    file.write(reinterpret_cast<const char*>(&numberOfArrayElements), 4);
    file.write(reinterpret_cast<const char*>(&numberOfFaces), 4);
    file.write(reinterpret_cast<const char*>(&numberOfMipmapLevels), 4);
    file.write(reinterpret_cast<const char*>(&bytesOfKeyValueData), 4);
    
    // Single mip level
    uint32_t imageSize = (uint32_t)data.size();
    file.write(reinterpret_cast<const char*>(&imageSize), 4);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    
    return true;
}

#pragma mark - Main

void printUsage(const char* programName) {
    fprintf(stderr, "Usage: %s [options]\n", programName);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i, --input <dir>      Input directory containing HDR files (default: ../assets/hdris)\n");
    fprintf(stderr, "  -o, --output <dir>     Output directory for generated maps (default: ../assets/env_maps)\n");
    fprintf(stderr, "  --face-size <n>        Size of environment cubemap faces (default: 512)\n");
    fprintf(stderr, "  --irradiance-size <n>  Size of irradiance cubemap faces (default: 64)\n");
    fprintf(stderr, "  --prefilter-size <n>   Base size of prefiltered cubemap (default: 256)\n");
    fprintf(stderr, "  --specular-samples <n> Samples for specular prefiltering (default: 1024)\n");
    fprintf(stderr, "  --diffuse-samples <n>  Samples for diffuse irradiance (default: 512)\n");
    fprintf(stderr, "  -h, --help             Show this help message\n");
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        // Default parameters
        NSString *inputDir = @"../assets/hdris";
        NSString *outputDir = @"../assets/env_maps";
        NSUInteger faceSize = 512;
        NSUInteger irradianceSize = 64;
        NSUInteger prefilterSize = 256;
        NSUInteger specularSamples = 1024;
        NSUInteger diffuseSamples = 512;
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "-i" || arg == "--input") {
                if (i + 1 < argc) inputDir = [NSString stringWithUTF8String:argv[++i]];
            } else if (arg == "-o" || arg == "--output") {
                if (i + 1 < argc) outputDir = [NSString stringWithUTF8String:argv[++i]];
            } else if (arg == "--face-size") {
                if (i + 1 < argc) faceSize = atoi(argv[++i]);
            } else if (arg == "--irradiance-size") {
                if (i + 1 < argc) irradianceSize = atoi(argv[++i]);
            } else if (arg == "--prefilter-size") {
                if (i + 1 < argc) prefilterSize = atoi(argv[++i]);
            } else if (arg == "--specular-samples") {
                if (i + 1 < argc) specularSamples = atoi(argv[++i]);
            } else if (arg == "--diffuse-samples") {
                if (i + 1 < argc) diffuseSamples = atoi(argv[++i]);
            } else if (arg == "-h" || arg == "--help") {
                printUsage(argv[0]);
                return 0;
            }
        }
        
        // Resolve paths
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *cwd = [fm currentDirectoryPath];
        
        if (![inputDir hasPrefix:@"/"]) {
            inputDir = [cwd stringByAppendingPathComponent:inputDir];
        }
        if (![outputDir hasPrefix:@"/"]) {
            outputDir = [cwd stringByAppendingPathComponent:outputDir];
        }
        
        inputDir = [inputDir stringByStandardizingPath];
        outputDir = [outputDir stringByStandardizingPath];
        
        NSLog(@"Input directory: %@", inputDir);
        NSLog(@"Output directory: %@", outputDir);
        
        // Create output directory
        NSError *error = nil;
        if (![fm createDirectoryAtPath:outputDir withIntermediateDirectories:YES attributes:nil error:&error]) {
            NSLog(@"Failed to create output directory: %@", error.localizedDescription);
            return 1;
        }
        
        // Find HDR files
        NSArray<NSString*> *contents = [fm contentsOfDirectoryAtPath:inputDir error:&error];
        if (!contents) {
            NSLog(@"Failed to read input directory: %@", error.localizedDescription);
            return 1;
        }
        
        NSMutableArray<NSString*> *hdrFiles = [NSMutableArray array];
        for (NSString *file in contents) {
            if ([[file.pathExtension lowercaseString] isEqualToString:@"hdr"]) {
                [hdrFiles addObject:file];
            }
        }
        
        if (hdrFiles.count == 0) {
            NSLog(@"No HDR files found in %@", inputDir);
            return 1;
        }
        
        NSLog(@"Found %lu HDR file(s)", (unsigned long)hdrFiles.count);
        
        // Initialize Metal
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Metal is not supported on this device");
            return 1;
        }
        NSLog(@"Using Metal device: %@", device.name);
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        
        // Load the Metal library from the iOS project
        NSString *metalLibPath = [[[NSBundle mainBundle] bundlePath] stringByAppendingPathComponent:@"default.metallib"];
        id<MTLLibrary> library = nil;
        
        // Try loading from bundle first
        library = [device newDefaultLibrary];
        
        if (!library) {
            // Try compiling the shader source directly
            NSString *shaderPath = @"../../ios/MetalRenderer/EnvironmentPrefilter.metal";
            if (![shaderPath hasPrefix:@"/"]) {
                shaderPath = [cwd stringByAppendingPathComponent:shaderPath];
            }
            shaderPath = [shaderPath stringByStandardizingPath];
            
            NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath encoding:NSUTF8StringEncoding error:&error];
            if (!shaderSource) {
                NSLog(@"Failed to load shader source: %@", error.localizedDescription);
                return 1;
            }
            
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            library = [device newLibraryWithSource:shaderSource options:options error:&error];
            if (!library) {
                NSLog(@"Failed to compile Metal shaders: %@", error.localizedDescription);
                return 1;
            }
        }
        
        NSLog(@"Metal library loaded successfully");
        
        // Create the prefilter
        NPBREnvironmentPrefilter *prefilter = [[NPBREnvironmentPrefilter alloc] initWithDevice:device
                                                                                       library:library
                                                                                  commandQueue:commandQueue];
        
        // Generate and save BRDF LUT once
        NSString *brdfPath = [outputDir stringByAppendingPathComponent:@"brdf_lut.ktx"];
        if (![fm fileExistsAtPath:brdfPath]) {
            NSLog(@"Generating BRDF LUT...");
            id<MTLTexture> brdfLUT = [prefilter sharedBRDFLUT];
            if (brdfLUT) {
                // Need to copy to shared storage for reading
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:brdfLUT.pixelFormat
                                                                                               width:brdfLUT.width
                                                                                              height:brdfLUT.height
                                                                                           mipmapped:NO];
                desc.usage = MTLTextureUsageShaderRead;
                desc.storageMode = MTLStorageModeShared;
                id<MTLTexture> sharedBRDF = [device newTextureWithDescriptor:desc];
                
                id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                [blit copyFromTexture:brdfLUT
                          sourceSlice:0
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(brdfLUT.width, brdfLUT.height, 1)
                            toTexture:sharedBRDF
                     destinationSlice:0
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                
                if (saveBRDFLUT(sharedBRDF, brdfPath.UTF8String)) {
                    NSLog(@"Saved: %@", brdfPath);
                }
            }
        } else {
            NSLog(@"BRDF LUT already exists: %@", brdfPath);
        }
        
        // Process each HDR file
        MTKTextureLoader *textureLoader = [[MTKTextureLoader alloc] initWithDevice:device];
        
        for (NSString *hdrFile in hdrFiles) {
            @autoreleasepool {
                NSString *hdrPath = [inputDir stringByAppendingPathComponent:hdrFile];
                NSString *baseName = [hdrFile stringByDeletingPathExtension];
                
                NSLog(@"\nProcessing: %@", baseName);
                
                // Load HDR image
                NSLog(@"  Loading HDR...");
                HDRImage hdrImage;
                try {
                    hdrImage = LoadHDR(hdrPath.UTF8String);
                } catch (const std::exception& e) {
                    NSLog(@"  Failed to load HDR: %s", e.what());
                    continue;
                }
                NSLog(@"  HDR size: %dx%d", hdrImage.width, hdrImage.height);
                
                // Create Metal texture from HDR
                MTLTextureDescriptor *hdrDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                   width:hdrImage.width
                                                                                                  height:hdrImage.height
                                                                                               mipmapped:NO];
                hdrDesc.usage = MTLTextureUsageShaderRead;
                hdrDesc.storageMode = MTLStorageModeShared;
                id<MTLTexture> hdrTexture = [device newTextureWithDescriptor:hdrDesc];
                
                [hdrTexture replaceRegion:MTLRegionMake2D(0, 0, hdrImage.width, hdrImage.height)
                              mipmapLevel:0
                                withBytes:hdrImage.pixels.data()
                              bytesPerRow:hdrImage.width * sizeof(HDRPixel)];
                
                // Generate environment products using the Metal prefilter
                NSLog(@"  Generating environment maps (this may take a moment)...");
                NSError *genError = nil;
                NPBREnvironmentProducts *products = [prefilter generateFromHDRTexture:hdrTexture
                                                                             faceSize:faceSize
                                                                       irradianceSize:irradianceSize
                                                                      specularSamples:specularSamples
                                                                       diffuseSamples:diffuseSamples
                                                                                error:&genError];
                
                if (!products) {
                    NSLog(@"  Failed to generate environment maps: %@", genError.localizedDescription);
                    continue;
                }
                
                // Copy textures to shared storage for saving
                auto copyToShared = ^id<MTLTexture>(id<MTLTexture> src) {
                    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:src.pixelFormat
                                                                                                       size:src.width
                                                                                                  mipmapped:(src.mipmapLevelCount > 1)];
                    desc.mipmapLevelCount = src.mipmapLevelCount;
                    desc.usage = MTLTextureUsageShaderRead;
                    desc.storageMode = MTLStorageModeShared;
                    id<MTLTexture> dst = [device newTextureWithDescriptor:desc];
                    
                    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    for (NSUInteger mip = 0; mip < src.mipmapLevelCount; ++mip) {
                        NSUInteger mipSize = MAX(1, src.width >> mip);
                        for (NSUInteger face = 0; face < 6; ++face) {
                            [blit copyFromTexture:src
                                      sourceSlice:face
                                      sourceLevel:mip
                                     sourceOrigin:MTLOriginMake(0, 0, 0)
                                       sourceSize:MTLSizeMake(mipSize, mipSize, 1)
                                        toTexture:dst
                                 destinationSlice:face
                                 destinationLevel:mip
                                destinationOrigin:MTLOriginMake(0, 0, 0)];
                        }
                    }
                    [blit endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    return dst;
                };
                
                // Save environment cubemap
                NSString *envPath = [outputDir stringByAppendingPathComponent:
                                     [NSString stringWithFormat:@"%@_env.ktx", baseName]];
                NSLog(@"  Saving environment cubemap...");
                id<MTLTexture> sharedEnv = copyToShared(products.environment);
                if (saveCubemapKTX(sharedEnv, envPath.UTF8String)) {
                    NSLog(@"  Saved: %@", envPath);
                }
                
                // Save irradiance cubemap
                NSString *irrPath = [outputDir stringByAppendingPathComponent:
                                     [NSString stringWithFormat:@"%@_irradiance.ktx", baseName]];
                NSLog(@"  Saving irradiance cubemap...");
                id<MTLTexture> sharedIrr = copyToShared(products.irradiance);
                if (saveCubemapKTX(sharedIrr, irrPath.UTF8String)) {
                    NSLog(@"  Saved: %@", irrPath);
                }
                
                // Save prefiltered cubemap
                NSString *pfPath = [outputDir stringByAppendingPathComponent:
                                    [NSString stringWithFormat:@"%@_prefiltered.ktx", baseName]];
                NSLog(@"  Saving prefiltered cubemap (%lu mip levels)...", (unsigned long)products.mipLevelCount);
                id<MTLTexture> sharedPf = copyToShared(products.prefiltered);
                if (saveCubemapKTX(sharedPf, pfPath.UTF8String)) {
                    NSLog(@"  Saved: %@", pfPath);
                }
                
                NSLog(@"  Done: %@", baseName);
            }
        }
        
        NSLog(@"\n==================================================");
        NSLog(@"Processed %lu environment(s)", (unsigned long)hdrFiles.count);
        NSLog(@"Output directory: %@", outputDir);
        
        return 0;
    }
}
