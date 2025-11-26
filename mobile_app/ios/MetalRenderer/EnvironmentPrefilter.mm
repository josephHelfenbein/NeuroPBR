#import "EnvironmentPrefilter.h"
#import <MetalKit/MetalKit.h>

static const NSUInteger kDefaultFaceSize = 512;
static const NSUInteger kDefaultIrradianceSize = 512;
static const NSUInteger kDefaultSpecularSamples = 1024;
static const NSUInteger kDefaultDiffuseSamples = 512;
static const NSUInteger kBRDFLUTSize = 512;

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

@end
