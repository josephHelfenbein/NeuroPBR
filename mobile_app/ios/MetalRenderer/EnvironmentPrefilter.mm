#import "EnvironmentPrefilter.h"
#import <MetalKit/MetalKit.h>

static const NSUInteger kDefaultFaceSize = 512;
static const NSUInteger kDefaultIrradianceSize = 512;
static const NSUInteger kDefaultSpecularSamples = 1024;
static const NSUInteger kDefaultDiffuseSamples = 512;
static const NSUInteger kBRDFLUTSize = 512;

// KTX file format constants
static const uint8_t KTX_IDENTIFIER[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};

#pragma pack(push, 1)
typedef struct {
    uint8_t identifier[12];
    uint32_t endianness;
    uint32_t glType;
    uint32_t glTypeSize;
    uint32_t glFormat;
    uint32_t glInternalFormat;
    uint32_t glBaseInternalFormat;
    uint32_t pixelWidth;
    uint32_t pixelHeight;
    uint32_t pixelDepth;
    uint32_t numberOfArrayElements;
    uint32_t numberOfFaces;
    uint32_t numberOfMipmapLevels;
    uint32_t bytesOfKeyValueData;
} KTXHeader;
#pragma pack(pop)

@interface NPBREnvironmentProducts ()
@end

@implementation NPBREnvironmentProducts
@end

@interface NPBREnvironmentPrefilter ()
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLComputePipelineState> equirectToCubePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> specularPipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> irradiancePipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> brdfPipeline;
@property(nonatomic, strong) id<MTLTexture> cachedBRDFLUT;
@end

@implementation NPBREnvironmentPrefilter

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  commandQueue:(id<MTLCommandQueue>)queue {
    if ((self = [super init])) {
        _device = device;
        _library = library;
        _commandQueue = queue;

        NSError *error = nil;
        _equirectToCubePipeline = [self pipelineNamed:@"equirectangularToCubemapKernel" error:&error];
        NSAssert(_equirectToCubePipeline, @"Failed to create equirectangular pipeline: %@", error);
        _specularPipeline = [self pipelineNamed:@"prefilterSpecularKernel" error:&error];
        NSAssert(_specularPipeline, @"Failed to create specular pipeline: %@", error);
        _irradiancePipeline = [self pipelineNamed:@"convolveDiffuseKernel" error:&error];
        NSAssert(_irradiancePipeline, @"Failed to create irradiance pipeline: %@", error);
        _brdfPipeline = [self pipelineNamed:@"precomputeBRDFKernel" error:&error];
        NSAssert(_brdfPipeline, @"Failed to create BRDF pipeline: %@", error);
    }
    return self;
}

- (id<MTLTexture>)sharedBRDFLUT {
    if (!_cachedBRDFLUT) {
        NSError *error = nil;
        _cachedBRDFLUT = [self generateBRDFLUTWithSize:kBRDFLUTSize error:&error];
        if (!_cachedBRDFLUT) {
            NSLog(@"[Neuropbr] Failed to generate BRDF LUT: %@", error.localizedDescription);
        }
    }
    return _cachedBRDFLUT;
}

- (nullable NPBREnvironmentProducts *)generateFromHDRTexture:(id<MTLTexture>)hdrTexture
                                                    faceSize:(NSUInteger)faceSize
                                              irradianceSize:(NSUInteger)irradianceSize
                                             specularSamples:(NSUInteger)specularSamples
                                             diffuseSamples:(NSUInteger)diffuseSamples
                                                       error:(NSError *__autoreleasing  _Nullable *)error {
    if (!hdrTexture) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-1 userInfo:@{NSLocalizedDescriptionKey : @"Missing HDR texture"}];
        }
        return nil;
    }

    faceSize = faceSize ? faceSize : kDefaultFaceSize;
    irradianceSize = irradianceSize ? irradianceSize : kDefaultIrradianceSize;
    specularSamples = specularSamples ? specularSamples : kDefaultSpecularSamples;
    diffuseSamples = diffuseSamples ? diffuseSamples : kDefaultDiffuseSamples;

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"EnvironmentPrefilter";

    id<MTLTexture> envCubemap = [self cubeTextureWithSize:faceSize mipmapped:YES format:hdrTexture.pixelFormat];
    [self encodeEquirectangularHDR:hdrTexture toCubemap:envCubemap commandBuffer:commandBuffer];

    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    blit.label = @"GenerateEnvMips";
    [blit generateMipmapsForTexture:envCubemap];
    [blit endEncoding];

    NSUInteger mipLevels = (NSUInteger)floor(log2((double)faceSize)) + 1;
    id<MTLTexture> specularCubemap = [self cubeTextureWithSize:faceSize mipmapped:YES format:hdrTexture.pixelFormat mipLevels:mipLevels];
    [self encodePrefilterSpecularFrom:envCubemap toTarget:specularCubemap samples:specularSamples commandBuffer:commandBuffer];

    id<MTLTexture> irradianceCubemap = [self cubeTextureWithSize:irradianceSize mipmapped:NO format:hdrTexture.pixelFormat];
    [self encodeIrradianceFrom:envCubemap toTarget:irradianceCubemap samples:diffuseSamples commandBuffer:commandBuffer];

    id<MTLTexture> brdfLUT = [self sharedBRDFLUT];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    NPBREnvironmentProducts *products = [[NPBREnvironmentProducts alloc] init];
    products.environment = envCubemap;
    products.prefiltered = specularCubemap;
    products.irradiance = irradianceCubemap;
    products.brdf = brdfLUT;
    products.mipLevelCount = mipLevels;
    return products;
}

#pragma mark - Encoding helpers

- (id<MTLComputePipelineState>)pipelineNamed:(NSString *)name error:(NSError **)error {
    id<MTLFunction> function = [_library newFunctionWithName:name];
    if (!function) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-2 userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithFormat:@"Missing Metal function %@", name]}];
        }
        return nil;
    }
    return [_device newComputePipelineStateWithFunction:function error:error];
}

- (id<MTLTexture>)cubeTextureWithSize:(NSUInteger)size mipmapped:(BOOL)mipmapped format:(MTLPixelFormat)format {
    NSUInteger mipLevels = mipmapped ? ((NSUInteger)floor(log2((double)size)) + 1) : 1;
    return [self cubeTextureWithSize:size mipmapped:mipmapped format:format mipLevels:mipLevels];
}

- (id<MTLTexture>)cubeTextureWithSize:(NSUInteger)size mipmapped:(BOOL)mipmapped format:(MTLPixelFormat)format mipLevels:(NSUInteger)mipLevels {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:size mipmapped:mipmapped];
    desc.mipmapLevelCount = mipLevels;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:desc];
}

- (id<MTLTexture>)cubeArrayViewForTexture:(id<MTLTexture>)texture mipLevel:(NSUInteger)mipLevel levelCount:(NSUInteger)levelCount {
    if (!texture || mipLevel >= texture.mipmapLevelCount || levelCount == 0) {
        return nil;
    }
    NSRange levelRange = NSMakeRange(mipLevel, MIN(levelCount, texture.mipmapLevelCount - mipLevel));
    NSRange sliceRange = NSMakeRange(0, 6);
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:MTLTextureType2DArray
                                           levels:levelRange
                                           slices:sliceRange];
}

- (void)encodeEquirectangularHDR:(id<MTLTexture>)hdrTexture toCubemap:(id<MTLTexture>)cubemap commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLTexture> cubeView = [self cubeArrayViewForTexture:cubemap mipLevel:0 levelCount:1];
    if (!cubeView) {
        return;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"EquirectangularToCube";
    [encoder setComputePipelineState:_equirectToCubePipeline];
    [encoder setTexture:hdrTexture atIndex:0];
    [encoder setTexture:cubeView atIndex:1];

    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
    MTLSize threadgroups = MTLSizeMake((cubemap.width + 7) / 8, (cubemap.height + 7) / 8, 6);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

- (void)encodePrefilterSpecularFrom:(id<MTLTexture>)envCube toTarget:(id<MTLTexture>)prefilteredCube samples:(NSUInteger)samples commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    for (NSUInteger level = 0; level < prefilteredCube.mipmapLevelCount; ++level) {
        if (level == 0) {
            id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
            blit.label = @"CopyLevel0";
            for (int slice = 0; slice < 6; ++slice) {
                [blit copyFromTexture:envCube
                          sourceSlice:slice
                          sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0)
                           sourceSize:MTLSizeMake(envCube.width, envCube.height, 1)
                            toTexture:prefilteredCube
                     destinationSlice:slice
                     destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            [blit endEncoding];
            continue;
        }

        id<MTLTexture> mipView = [self cubeArrayViewForTexture:prefilteredCube mipLevel:level levelCount:1];
        if (!mipView) {
            continue;
        }
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        encoder.label = [NSString stringWithFormat:@"SpecularPrefilter_L%lu", (unsigned long)level];
        [encoder setComputePipelineState:_specularPipeline];
        [encoder setTexture:envCube atIndex:0];
        [encoder setTexture:mipView atIndex:1];

        struct PrefilterUniforms {
            uint32_t faceSize;
            uint32_t sampleCount;
            float roughness;
            uint32_t mipLevel;
        } uniforms;
        NSUInteger mipFace = MAX((NSUInteger)1, prefilteredCube.width >> level);
        uniforms.faceSize = (uint32_t)mipFace;
        uniforms.sampleCount = (uint32_t)samples;
        uniforms.roughness = prefilteredCube.mipmapLevelCount > 1 ? (float)level / (float)(prefilteredCube.mipmapLevelCount - 1) : 0.0f;
        uniforms.mipLevel = (uint32_t)level;
        [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];

        MTLSize tg = MTLSizeMake(8, 8, 1);
        MTLSize grid = MTLSizeMake((uniforms.faceSize + 7) / 8, (uniforms.faceSize + 7) / 8, 6);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [encoder endEncoding];
    }
}

- (void)encodeIrradianceFrom:(id<MTLTexture>)envCube toTarget:(id<MTLTexture>)irradianceCube samples:(NSUInteger)samples commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLTexture> irrView = [self cubeArrayViewForTexture:irradianceCube mipLevel:0 levelCount:1];
    if (!irrView) {
        return;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"DiffuseIrradiance";
    [encoder setComputePipelineState:_irradiancePipeline];
    [encoder setTexture:envCube atIndex:0];
    [encoder setTexture:irrView atIndex:1];

    struct IrradianceUniforms {
        uint32_t faceSize;
        uint32_t sampleCount;
    } uniforms;
    uniforms.faceSize = (uint32_t)irradianceCube.width;
    uniforms.sampleCount = (uint32_t)samples;
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];

    MTLSize tg = MTLSizeMake(8, 8, 1);
    MTLSize grid = MTLSizeMake((irradianceCube.width + 7) / 8, (irradianceCube.height + 7) / 8, 6);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];
}

- (id<MTLTexture>)generateBRDFLUTWithSize:(NSUInteger)size error:(NSError **)error {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:size height:size mipmapped:NO];
    desc.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> lut = [_device newTextureWithDescriptor:desc];

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"BRDFLUT";
    [encoder setComputePipelineState:_brdfPipeline];
    [encoder setTexture:lut atIndex:0];
    struct {
        uint32_t width;
        uint32_t height;
    } uniforms = { (uint32_t)size, (uint32_t)size };
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];
    MTLSize tg = MTLSizeMake(8, 8, 1);
    MTLSize grid = MTLSizeMake((size + 7) / 8, (size + 7) / 8, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    return lut;
}

#pragma mark - KTX Loading

- (MTLPixelFormat)metalPixelFormatFromGLFormat:(uint32_t)glInternalFormat glType:(uint32_t)glType {
    // GL_RGBA16F = 0x881A, GL_HALF_FLOAT = 0x140B
    // GL_RGBA32F = 0x8814, GL_FLOAT = 0x1406
    // GL_RG16F = 0x822F
    
    if (glInternalFormat == 0x881A || (glInternalFormat == 0x1908 && glType == 0x140B)) {
        return MTLPixelFormatRGBA16Float;
    }
    if (glInternalFormat == 0x8814 || (glInternalFormat == 0x1908 && glType == 0x1406)) {
        return MTLPixelFormatRGBA32Float;
    }
    if (glInternalFormat == 0x822F) {
        return MTLPixelFormatRG16Float;
    }
    // Default to RGBA16Float
    return MTLPixelFormatRGBA16Float;
}

- (NSUInteger)bytesPerPixelForFormat:(MTLPixelFormat)format {
    switch (format) {
        case MTLPixelFormatRGBA16Float: return 8;
        case MTLPixelFormatRGBA32Float: return 16;
        case MTLPixelFormatRG16Float: return 4;
        case MTLPixelFormatRG32Float: return 8;
        default: return 8;
    }
}

- (nullable id<MTLTexture>)loadKTXCubemap:(NSString *)path error:(NSError **)error {
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:error];
    if (!data) {
        return nil;
    }
    
    if (data.length < sizeof(KTXHeader)) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-10 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX file too small"}];
        }
        return nil;
    }
    
    const uint8_t *bytes = (const uint8_t *)data.bytes;
    KTXHeader header;
    memcpy(&header, bytes, sizeof(KTXHeader));
    
    // Verify identifier
    if (memcmp(header.identifier, KTX_IDENTIFIER, 12) != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-11 
                                     userInfo:@{NSLocalizedDescriptionKey : @"Invalid KTX identifier"}];
        }
        return nil;
    }
    
    // Check endianness
    BOOL needsSwap = (header.endianness != 0x04030201);
    if (needsSwap) {
        // For simplicity, we don't support byte-swapped files
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-12 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX endianness not supported"}];
        }
        return nil;
    }
    
    if (header.numberOfFaces != 6) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-13 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX file is not a cubemap"}];
        }
        return nil;
    }
    
    MTLPixelFormat format = [self metalPixelFormatFromGLFormat:header.glInternalFormat glType:header.glType];
    NSUInteger bytesPerPixel = [self bytesPerPixelForFormat:format];
    NSUInteger size = header.pixelWidth;
    NSUInteger mipCount = MAX(1, header.numberOfMipmapLevels);
    
    // Create cubemap texture
    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format 
                                                                                       size:size 
                                                                                  mipmapped:(mipCount > 1)];
    desc.mipmapLevelCount = mipCount;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    
    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
    if (!texture) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-14 
                                     userInfo:@{NSLocalizedDescriptionKey : @"Failed to create cubemap texture"}];
        }
        return nil;
    }
    
    // Read mip levels
    NSUInteger offset = sizeof(KTXHeader) + header.bytesOfKeyValueData;
    
    for (NSUInteger mip = 0; mip < mipCount; ++mip) {
        if (offset + 4 > data.length) break;
        
        uint32_t imageSize;
        memcpy(&imageSize, bytes + offset, 4);
        offset += 4;
        
        NSUInteger mipSize = MAX(1, size >> mip);
        NSUInteger bytesPerRow = mipSize * bytesPerPixel;
        NSUInteger bytesPerFace = mipSize * mipSize * bytesPerPixel;
        
        for (NSUInteger face = 0; face < 6; ++face) {
            if (offset + bytesPerFace > data.length) break;
            
            [texture replaceRegion:MTLRegionMake2D(0, 0, mipSize, mipSize)
                       mipmapLevel:mip
                             slice:face
                         withBytes:bytes + offset
                       bytesPerRow:bytesPerRow
                     bytesPerImage:bytesPerFace];
            
            offset += bytesPerFace;
        }
        
        // Align to 4-byte boundary
        NSUInteger padding = (4 - (imageSize % 4)) % 4;
        offset += padding;
    }
    
    return texture;
}

- (nullable id<MTLTexture>)loadTexture2D:(NSString *)path error:(NSError **)error {
    // Check if this is a KTX file
    NSString *ext = path.pathExtension.lowercaseString;
    if ([ext isEqualToString:@"ktx"]) {
        return [self loadKTX2DTexture:path error:error];
    }
    
    // Use MTKTextureLoader for PNG/image files
    MTKTextureLoader *loader = [[MTKTextureLoader alloc] initWithDevice:_device];
    
    NSDictionary *options = @{
        MTKTextureLoaderOptionSRGB : @NO,
        MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
        MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared)
    };
    
    NSURL *url = [NSURL fileURLWithPath:path];
    id<MTLTexture> texture = [loader newTextureWithContentsOfURL:url options:options error:error];
    
    return texture;
}

- (nullable id<MTLTexture>)loadKTX2DTexture:(NSString *)path error:(NSError **)error {
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:error];
    if (!data) {
        return nil;
    }
    
    if (data.length < sizeof(KTXHeader)) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-20 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX 2D file too small"}];
        }
        return nil;
    }
    
    const uint8_t *bytes = (const uint8_t *)data.bytes;
    KTXHeader header;
    memcpy(&header, bytes, sizeof(KTXHeader));
    
    // Verify identifier
    if (memcmp(header.identifier, KTX_IDENTIFIER, 12) != 0) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-21 
                                     userInfo:@{NSLocalizedDescriptionKey : @"Invalid KTX 2D identifier"}];
        }
        return nil;
    }
    
    // Check endianness
    BOOL needsSwap = (header.endianness != 0x04030201);
    if (needsSwap) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-22 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX 2D endianness not supported"}];
        }
        return nil;
    }
    
    MTLPixelFormat format = [self metalPixelFormatFromGLFormat:header.glInternalFormat glType:header.glType];
    NSUInteger bytesPerPixel = [self bytesPerPixelForFormat:format];
    NSUInteger width = header.pixelWidth;
    NSUInteger height = header.pixelHeight > 0 ? header.pixelHeight : 1;
    
    // Create 2D texture
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format 
                                                                                    width:width 
                                                                                   height:height 
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    
    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
    if (!texture) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-23 
                                     userInfo:@{NSLocalizedDescriptionKey : @"Failed to create 2D texture"}];
        }
        return nil;
    }
    
    // Read first mip level
    NSUInteger offset = sizeof(KTXHeader) + header.bytesOfKeyValueData;
    
    if (offset + 4 > data.length) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-24 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX 2D file truncated"}];
        }
        return nil;
    }
    
    uint32_t imageSize;
    memcpy(&imageSize, bytes + offset, 4);
    offset += 4;
    
    NSUInteger bytesPerRow = width * bytesPerPixel;
    
    if (offset + width * height * bytesPerPixel > data.length) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-25 
                                     userInfo:@{NSLocalizedDescriptionKey : @"KTX 2D image data truncated"}];
        }
        return nil;
    }
    
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0
                 withBytes:bytes + offset
               bytesPerRow:bytesPerRow];
    
    return texture;
}

- (nullable NPBREnvironmentProducts *)loadPrecomputedEnvironment:(NSString *)environmentPath
                                                  irradiancePath:(NSString *)irradiancePath
                                                 prefilteredPath:(NSString *)prefilteredPath
                                                        brdfPath:(nullable NSString *)brdfPath
                                                           error:(NSError **)error {
    NSError *loadError = nil;
    
    // Load environment cubemap
    id<MTLTexture> envTexture = [self loadKTXCubemap:environmentPath error:&loadError];
    if (!envTexture) {
        NSLog(@"[Neuropbr] Failed to load environment cubemap: %@", loadError.localizedDescription);
        if (error) *error = loadError;
        return nil;
    }
    
    // Load irradiance cubemap
    id<MTLTexture> irrTexture = [self loadKTXCubemap:irradiancePath error:&loadError];
    if (!irrTexture) {
        NSLog(@"[Neuropbr] Failed to load irradiance cubemap: %@", loadError.localizedDescription);
        if (error) *error = loadError;
        return nil;
    }
    
    // Load prefiltered cubemap
    id<MTLTexture> prefilteredTexture = [self loadKTXCubemap:prefilteredPath error:&loadError];
    if (!prefilteredTexture) {
        NSLog(@"[Neuropbr] Failed to load prefiltered cubemap: %@", loadError.localizedDescription);
        if (error) *error = loadError;
        return nil;
    }
    
    // Load or generate BRDF LUT
    id<MTLTexture> brdfTexture = nil;
    if (brdfPath.length > 0) {
        brdfTexture = [self loadTexture2D:brdfPath error:&loadError];
        if (!brdfTexture) {
            NSLog(@"[Neuropbr] Failed to load BRDF LUT, will generate: %@", loadError.localizedDescription);
        }
    }
    if (!brdfTexture) {
        brdfTexture = [self sharedBRDFLUT];
    }
    
    NPBREnvironmentProducts *products = [[NPBREnvironmentProducts alloc] init];
    products.environment = envTexture;
    products.irradiance = irrTexture;
    products.prefiltered = prefilteredTexture;
    products.brdf = brdfTexture;
    products.mipLevelCount = prefilteredTexture.mipmapLevelCount;
    
    NSLog(@"[Neuropbr] Loaded precomputed environment: env=%lux%lu, irr=%lux%lu, pf=%lux%lu mips=%lu",
          (unsigned long)envTexture.width, (unsigned long)envTexture.height,
          (unsigned long)irrTexture.width, (unsigned long)irrTexture.height,
          (unsigned long)prefilteredTexture.width, (unsigned long)prefilteredTexture.height,
          (unsigned long)products.mipLevelCount);
    
    return products;
}

@end
