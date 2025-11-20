#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <Flutter/Flutter.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>
#import <MobileCoreServices/MobileCoreServices.h>
#import <simd/simd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#import "Renderer.hpp"
#import "EnvironmentPrefilter.h"

using namespace neuropbr;

namespace {

struct GPUFrameUniforms {
    float viewProjection[16];
    float invView[16];
    float cameraPosTime[4];
    float resolutionExposure[4];
    float iblParams[4];
    float toneMapping[4];
};

struct GPUMaterialUniforms {
    float baseTint[4];
    float scalars[4];
    float featureToggles[4];
};

constexpr MTLPixelFormat kColorFormat = MTLPixelFormatBGRA8Unorm;

uint64_t HandleForTexture(id<MTLTexture> texture) {
    return reinterpret_cast<uint64_t>((__bridge const void *)texture);
}

PreviewChannel PreviewChannelFromNumber(NSNumber *value) {
    if (!value) {
        return PreviewChannel::Final;
    }
    switch (value.unsignedIntegerValue) {
    case 1:
        return PreviewChannel::Albedo;
    case 2:
        return PreviewChannel::Roughness;
    case 3:
        return PreviewChannel::Metallic;
    case 4:
        return PreviewChannel::Normal;
    default:
        return PreviewChannel::Final;
    }
}

MaterialSlot MaterialSlotFromString(NSString *key) {
    NSString *lower = key.lowercaseString;
    if ([lower isEqualToString:@"albedo"]) {
        return MaterialSlot::Albedo;
    }
    if ([lower isEqualToString:@"normal"]) {
        return MaterialSlot::Normal;
    }
    if ([lower isEqualToString:@"roughness"]) {
        return MaterialSlot::Roughness;
    }
    if ([lower isEqualToString:@"metallic"]) {
        return MaterialSlot::Metallic;
    }
    return MaterialSlot::Albedo;
}

MTLPixelFormat PixelFormatFromString(NSString *value, MTLPixelFormat fallback) {
    if (!value.length) {
        return fallback;
    }
    NSString *lower = value.lowercaseString;
    if ([lower isEqualToString:@"rgba32float"]) {
        return MTLPixelFormatRGBA32Float;
    }
    if ([lower isEqualToString:@"rg32float"]) {
        return MTLPixelFormatRG32Float;
    }
    if ([lower isEqualToString:@"r32float"]) {
        return MTLPixelFormatR32Float;
    }
    if ([lower isEqualToString:@"rgba16float"]) {
        return MTLPixelFormatRGBA16Float;
    }
    if ([lower isEqualToString:@"rgba8unorm"]) {
        return MTLPixelFormatRGBA8Unorm;
    }
    if ([lower isEqualToString:@"rg8unorm"]) {
        return MTLPixelFormatRG8Unorm;
    }
    if ([lower isEqualToString:@"r8unorm"]) {
        return MTLPixelFormatR8Unorm;
    }
    if ([lower isEqualToString:@"rg16float"]) {
        return MTLPixelFormatRG16Float;
    }
    if ([lower isEqualToString:@"r16float"]) {
        return MTLPixelFormatR16Float;
    }
    return fallback;
}

uint32_t ChannelsFromFormat(MTLPixelFormat format, NSNumber *overrideValue) {
    if (overrideValue) {
        return overrideValue.unsignedIntValue;
    }
    switch (format) {
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR16Float:
        return 1;
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRG32Float:
        return 2;
    default:
        return 4;
    }
}

bool IsHDRFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA16Float:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA32Float:
        return true;
    default:
        return false;
    }
}

NSUInteger BytesPerPixel(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Unorm:
        return 1;
    case MTLPixelFormatRG8Unorm:
        return 2;
    case MTLPixelFormatRGBA8Unorm:
        return 4;
    case MTLPixelFormatR16Float:
        return 2;
    case MTLPixelFormatRG16Float:
        return 4;
    case MTLPixelFormatRG32Float:
        return 8;
    case MTLPixelFormatR32Float:
        return 4;
    case MTLPixelFormatRGBA16Float:
        return 8;
    case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        return 4;
    }
}

float DegreesToRadians(float degrees) { return degrees * (3.14159265359f / 180.0f); }

} // namespace

@interface NPBRMetalRendererBridge : NSObject <FlutterTexture>

- (instancetype)initWithRegistrar:(id<FlutterTextureRegistrar>)registrar;
- (BOOL)initializeWithSize:(CGSize)size error:(NSError **)error;
- (void)shutdown;

@property(nonatomic, readonly) int64_t textureId;
@property(nonatomic, readonly) NSString *deviceName;

- (void)setCamera:(const CameraParameters &)camera;
- (void)setLighting:(const LightingControls &)lighting;
- (void)setPreview:(const PreviewControls &)preview;
- (void)setEnvironment:(const EnvironmentControls &)environment;
- (BOOL)applyEnvironmentDictionary:(NSDictionary *)args error:(NSError **)error;

- (void)loadMaterial:(uint32_t)materialId payloads:(NSDictionary<NSString *, NSDictionary *> *)payloads;
- (void)updateMaterial:(uint32_t)materialId slot:(MaterialSlot)slot payload:(NSDictionary *)payload;

- (BOOL)renderMaterial:(uint32_t)materialId;
- (NSData *)snapshotPNG;

- (void)setFrameCallback:(void (^)(uint64_t textureHandle))callback;
- (void)setErrorCallback:(void (^)(NSError *error))callback;

@end

@implementation NPBRMetalRendererBridge {
    id<FlutterTextureRegistrar> _registrar;
    int64_t _textureId;

    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    id<MTLLibrary> _library;
    id<MTLRenderPipelineState> _pipeline;

    id<MTLTexture> _colorTarget;
    CVPixelBufferRef _pixelBuffer;

    id<MTLBuffer> _frameUniforms;
    id<MTLBuffer> _materialUniforms;

    Renderer *_renderer;

    std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> _defaultBindings;

    NSMutableDictionary<NSNumber *, id<MTLTexture>> *_textureCache;
    std::mutex _textureMutex;

    dispatch_queue_t _renderQueue;

    void (^_frameCallback)(uint64_t textureHandle);
    void (^_errorCallback)(NSError *error);

    id<MTLTexture> _fallbackEnv;
    id<MTLTexture> _fallbackIrradiance;
    id<MTLTexture> _fallbackPrefilter;
    id<MTLTexture> _fallbackBRDF;
    id<MTLTexture> _fallbackAlbedo;
    id<MTLTexture> _fallbackNormal;
    id<MTLTexture> _fallbackRoughness;
    id<MTLTexture> _fallbackMetallic;
    EnvironmentControls _defaultEnvironment;

    MTKTextureLoader *_textureLoader;
    NSString *_deviceName;
    NPBREnvironmentPrefilter *_prefilter;
}

- (instancetype)initWithRegistrar:(id<FlutterTextureRegistrar>)registrar {
    if ((self = [super init])) {
        _registrar = registrar;
        _textureId = -1;
        _textureCache = [NSMutableDictionary dictionary];
        _renderQueue = dispatch_queue_create("com.neuropbr.renderer", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)dealloc {
    [self shutdown];
}

- (BOOL)initializeWithSize:(CGSize)size error:(NSError *__autoreleasing  _Nullable *)error {
    if (_renderer) {
        return YES;
    }

    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBRMetal" code:-1 userInfo:@{NSLocalizedDescriptionKey : @"Metal device unavailable"}];
        }
        return NO;
    }

    _deviceName = _device.name;
    _textureLoader = [[MTKTextureLoader alloc] initWithDevice:_device];
    _commandQueue = [_device newCommandQueue];
    _library = [_device newDefaultLibrary];

    if (!_library) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBRMetal" code:-2 userInfo:@{NSLocalizedDescriptionKey : @"Failed to load Metal library"}];
        }
        return NO;
    }

    _prefilter = [[NPBREnvironmentPrefilter alloc] initWithDevice:_device library:_library commandQueue:_commandQueue];

    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.label = @"NeuropbrPipeline";
    pipelineDescriptor.vertexFunction = [_library newFunctionWithName:@"fullscreen_vertex"];
    pipelineDescriptor.fragmentFunction = [_library newFunctionWithName:@"pbr_fragment"];
    pipelineDescriptor.colorAttachments[0].pixelFormat = kColorFormat;
    pipelineDescriptor.sampleCount = 1;

    NSError *pipelineError = nil;
    _pipeline = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&pipelineError];
    if (!_pipeline) {
        if (error) {
            *error = pipelineError ?: [NSError errorWithDomain:@"NPBRMetal" code:-3 userInfo:@{NSLocalizedDescriptionKey : @"Failed to create pipeline"}];
        }
        return NO;
    }

    NSDictionary *cvOptions = @{
        (__bridge NSString *)kCVPixelBufferMetalCompatibilityKey : @YES,
        (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey : @{}
    };

    CVReturn cvStatus = CVPixelBufferCreate(kCFAllocatorDefault,
                                            static_cast<size_t>(size.width),
                                            static_cast<size_t>(size.height),
                                            kCVPixelFormatType_32BGRA,
                                            (__bridge CFDictionaryRef)cvOptions,
                                            &_pixelBuffer);
    if (cvStatus != kCVReturnSuccess) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBRMetal" code:-4 userInfo:@{NSLocalizedDescriptionKey : @"Failed to allocate pixel buffer"}];
        }
        return NO;
    }

    IOSurfaceRef surface = CVPixelBufferGetIOSurface(_pixelBuffer);
    MTLTextureDescriptor *colorDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kColorFormat width:(NSUInteger)size.width height:(NSUInteger)size.height mipmapped:NO];
    colorDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    colorDesc.storageMode = MTLStorageModeShared;
    colorDesc.resourceOptions = MTLResourceStorageModeShared;
    colorDesc.iosurface = surface;
    colorDesc.iosurfacePlane = 0;
    _colorTarget = [_device newTextureWithDescriptor:colorDesc];
    _colorTarget.label = @"NeuropbrRenderTarget";

    RendererConfig config{};
    config.width = static_cast<uint32_t>(size.width);
    config.height = static_cast<uint32_t>(size.height);

    _renderer = npbrCreateRenderer(config);
    _frameUniforms = [_device newBufferWithLength:sizeof(GPUFrameUniforms) options:MTLResourceStorageModeShared];
    _materialUniforms = [_device newBufferWithLength:sizeof(GPUMaterialUniforms) options:MTLResourceStorageModeShared];

    [self buildFallbackTextures];
    npbrSetEnvironment(_renderer, _defaultEnvironment);

    if (_textureId < 0) {
        _textureId = [_registrar registerTexture:self];
    }
    return YES;
}

- (void)shutdown {
    if (_textureId >= 0 && _registrar) {
        [_registrar unregisterTexture:_textureId];
        _textureId = -1;
    }
    if (_pixelBuffer) {
        CVPixelBufferRelease(_pixelBuffer);
        _pixelBuffer = nullptr;
    }
    _colorTarget = nil;
    _pipeline = nil;
    _library = nil;
    _commandQueue = nil;
    _device = nil;
    _textureLoader = nil;

    if (_renderer) {
        npbrDestroyRenderer(_renderer);
        _renderer = nullptr;
    }

    [_textureCache removeAllObjects];
}

- (int64_t)textureId { return _textureId; }

- (NSString *)deviceName { return _deviceName ?: @"Unknown"; }

- (void)setCamera:(const CameraParameters &)camera { npbrSetCamera(_renderer, camera); }

- (void)setLighting:(const LightingControls &)lighting { npbrSetLighting(_renderer, lighting); }

- (void)setPreview:(const PreviewControls &)preview { npbrSetPreview(_renderer, preview); }

- (void)setEnvironment:(const EnvironmentControls &)environment { npbrSetEnvironment(_renderer, environment); }

- (void)loadMaterial:(uint32_t)materialId payloads:(NSDictionary<NSString *, NSDictionary *> *)payloads {
    std::array<TextureBinding, static_cast<size_t>(MaterialSlot::COUNT)> bindings = _defaultBindings;
    [payloads enumerateKeysAndObjectsUsingBlock:^(NSString *key, NSDictionary *payload, BOOL *stop) {
        MaterialSlot slot = MaterialSlotFromString(key);
        TextureBinding binding = [self bindingFromPayload:payload slot:slot];
        if (binding.handle) {
            bindings[static_cast<size_t>(slot)] = binding;
        }
    }];
    npbrUpsertMaterial(_renderer, materialId, bindings);
}

- (void)updateMaterial:(uint32_t)materialId slot:(MaterialSlot)slot payload:(NSDictionary *)payload {
    TextureBinding binding = [self bindingFromPayload:payload slot:slot];
    if (binding.handle) {
        npbrUpdateMaterialTexture(_renderer, materialId, slot, binding);
    }
}

- (BOOL)renderMaterial:(uint32_t)materialId {
    __block BOOL success = NO;
    dispatch_sync(_renderQueue, ^{
        if (!_renderer) {
            return;
        }
        FrameDescriptor descriptor{};
        if (!npbrBuildFrame(_renderer, materialId, descriptor)) {
            return;
        }
        [self writeUniforms:descriptor];
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        commandBuffer.label = @"NeuropbrFrame";

        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = _colorTarget;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

        id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
        encoder.label = @"NeuropbrEncoder";
        [encoder setRenderPipelineState:_pipeline];
        [encoder setFragmentBuffer:_frameUniforms offset:0 atIndex:0];
        [encoder setFragmentBuffer:_materialUniforms offset:0 atIndex:1];

        id<MTLTexture> env = [self textureForHandle:descriptor.environment.environmentHandle fallback:_fallbackEnv];
        id<MTLTexture> irradiance = [self textureForHandle:descriptor.environment.irradianceHandle fallback:_fallbackIrradiance];
        id<MTLTexture> prefiltered = [self textureForHandle:descriptor.environment.prefilteredHandle fallback:_fallbackPrefilter];
        id<MTLTexture> brdf = [self textureForHandle:descriptor.environment.brdfHandle fallback:_fallbackBRDF];

        [encoder setFragmentTexture:env atIndex:0];
        [encoder setFragmentTexture:irradiance atIndex:1];
        [encoder setFragmentTexture:prefiltered atIndex:2];
        [encoder setFragmentTexture:brdf atIndex:3];

        id<MTLTexture> albedo = [self textureForHandle:descriptor.textures[static_cast<size_t>(MaterialSlot::Albedo)].handle fallback:_fallbackAlbedo];
        id<MTLTexture> normal = [self textureForHandle:descriptor.textures[static_cast<size_t>(MaterialSlot::Normal)].handle fallback:_fallbackNormal];
        id<MTLTexture> roughness = [self textureForHandle:descriptor.textures[static_cast<size_t>(MaterialSlot::Roughness)].handle fallback:_fallbackRoughness];
        id<MTLTexture> metallic = [self textureForHandle:descriptor.textures[static_cast<size_t>(MaterialSlot::Metallic)].handle fallback:_fallbackMetallic];

        [encoder setFragmentTexture:albedo atIndex:4];
        [encoder setFragmentTexture:normal atIndex:5];
        [encoder setFragmentTexture:roughness atIndex:6];
        [encoder setFragmentTexture:metallic atIndex:7];

        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [encoder endEncoding];

        __weak typeof(self) weakSelf = self;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            __strong typeof(self) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            if (buffer.status == MTLCommandBufferStatusCompleted) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    [strongSelf->_registrar textureFrameAvailable:strongSelf->_textureId];
                    if (strongSelf->_frameCallback) {
                        strongSelf->_frameCallback(HandleForTexture(strongSelf->_colorTarget));
                    }
                });
            } else if (buffer.status == MTLCommandBufferStatusError && strongSelf->_errorCallback) {
                NSError *commandError = buffer.error ?: [NSError errorWithDomain:@"NPBRMetal" code:-10 userInfo:@{NSLocalizedDescriptionKey : @"Unknown Metal error"}];
                strongSelf->_errorCallback(commandError);
            }
        }];

        [commandBuffer commit];
        success = YES;
    });
    return success;
}

- (NSData *)snapshotPNG {
    if (!_colorTarget) {
        return nil;
    }
    NSUInteger width = _colorTarget.width;
    NSUInteger height = _colorTarget.height;
    NSUInteger bytesPerRow = width * 4;
    id<MTLBuffer> readback = [_device newBufferWithLength:bytesPerRow * height options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit copyFromTexture:_colorTarget
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:MTLOriginMake(0, 0, 0)
              sourceSize:MTLSizeMake(width, height, 1)
               toBuffer:readback
      destinationOffset:0
 destinationBytesPerRow:bytesPerRow
destinationBytesPerImage:bytesPerRow * height];
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    uint8_t *bytes = (uint8_t *)readback.contents;
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(bytes, width, height, 8, bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGImageRef cgImage = CGBitmapContextCreateImage(context);

    NSMutableData *data = [NSMutableData data];
    CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)data, kUTTypePNG, 1, nullptr);
    if (destination) {
        CGImageDestinationAddImage(destination, cgImage, nullptr);
        CGImageDestinationFinalize(destination);
        CFRelease(destination);
    }

    CGImageRelease(cgImage);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);

    return data;
}

- (void)setFrameCallback:(void (^)(uint64_t))callback { _frameCallback = [callback copy]; }

- (void)setErrorCallback:(void (^)(NSError *))callback { _errorCallback = [callback copy]; }

#pragma mark - FlutterTexture

- (CVPixelBufferRef)copyPixelBuffer {
    if (!_pixelBuffer) {
        return nil;
    }
    CVPixelBufferRetain(_pixelBuffer);
    return _pixelBuffer;
}

#pragma mark - Helpers

- (void)buildFallbackTextures {
    _fallbackEnv = [self solidColorCube:MTLPixelFormatRGBA32Float value:vector_float4{0.1f, 0.1f, 0.1f, 1.0f}];
    _fallbackIrradiance = _fallbackEnv;
    _fallbackPrefilter = _fallbackEnv;
    _fallbackBRDF = [self solidColor2D:MTLPixelFormatRG32Float value:vector_float4{0.5f, 0.5f, 0.0f, 0.0f} width:256 height:256];

    TextureBinding envBinding = [self bindingForTexture:_fallbackEnv channels:4 hdr:YES];
    TextureBinding irrBinding = [self bindingForTexture:_fallbackIrradiance channels:4 hdr:YES];
    TextureBinding preBinding = [self bindingForTexture:_fallbackPrefilter channels:4 hdr:YES];
    TextureBinding brdfBinding = [self bindingForTexture:_fallbackBRDF channels:2 hdr:NO];

    _defaultEnvironment.environmentId = 0;
    _defaultEnvironment.environmentHandle = envBinding.handle;
    _defaultEnvironment.irradianceHandle = irrBinding.handle;
    _defaultEnvironment.prefilteredHandle = preBinding.handle;
    _defaultEnvironment.brdfHandle = brdfBinding.handle;
    _defaultEnvironment.lodCount = 1.0f;

    _fallbackAlbedo = [self solidColor2D:MTLPixelFormatRGBA8Unorm value:vector_float4{1.0f, 1.0f, 1.0f, 1.0f} width:1 height:1];
    _fallbackNormal = [self solidColor2D:MTLPixelFormatRGBA8Unorm value:vector_float4{0.5f, 0.5f, 1.0f, 1.0f} width:1 height:1];
    _fallbackRoughness = [self solidColor2D:MTLPixelFormatRGBA8Unorm value:vector_float4{0.5f, 0.5f, 0.5f, 1.0f} width:1 height:1];
    _fallbackMetallic = [self solidColor2D:MTLPixelFormatRGBA8Unorm value:vector_float4{0.0f, 0.0f, 0.0f, 1.0f} width:1 height:1];

    _defaultBindings[static_cast<size_t>(MaterialSlot::Albedo)] = [self bindingForTexture:_fallbackAlbedo channels:4 hdr:NO];
    _defaultBindings[static_cast<size_t>(MaterialSlot::Normal)] = [self bindingForTexture:_fallbackNormal channels:4 hdr:NO];
    _defaultBindings[static_cast<size_t>(MaterialSlot::Roughness)] = [self bindingForTexture:_fallbackRoughness channels:4 hdr:NO];
    _defaultBindings[static_cast<size_t>(MaterialSlot::Metallic)] = [self bindingForTexture:_fallbackMetallic channels:4 hdr:NO];
}

- (id<MTLTexture>)solidColorCube:(MTLPixelFormat)format value:(vector_float4)value {

- (id<MTLTexture>)solidColor2D:(MTLPixelFormat)format value:(vector_float4)value width:(NSUInteger)width height:(NSUInteger)height {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];
    NSUInteger bytesPerPixel = BytesPerPixel(format);
    NSUInteger bytesPerRow = bytesPerPixel * width;
    std::vector<uint8_t> data(bytesPerRow * height, 0);

    for (NSUInteger y = 0; y < height; ++y) {
        uint8_t *row = data.data() + y * bytesPerRow;
        for (NSUInteger x = 0; x < width; ++x) {
            if (format == MTLPixelFormatRGBA8Unorm) {
                row[x * 4 + 0] = static_cast<uint8_t>(value.z * 255.0f);
                row[x * 4 + 1] = static_cast<uint8_t>(value.y * 255.0f);
                row[x * 4 + 2] = static_cast<uint8_t>(value.x * 255.0f);
                row[x * 4 + 3] = static_cast<uint8_t>(value.w * 255.0f);
            } else if (format == MTLPixelFormatRG32Float) {
                float *dst = reinterpret_cast<float *>(row) + x * 2;
                dst[0] = value.x;
                dst[1] = value.y;
            } else if (format == MTLPixelFormatR32Float) {
                float *dst = reinterpret_cast<float *>(row) + x;
                dst[0] = value.x;
            } else {
                float *dst = reinterpret_cast<float *>(row) + x * 4;
                dst[0] = value.x;
                dst[1] = value.y;
                dst[2] = value.z;
                dst[3] = value.w;
            }
        }
    }

    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:data.data() bytesPerRow:bytesPerRow];
    return texture;
}

- (id<MTLTexture>)solidColorCube:(MTLPixelFormat)format value:(vector_float4)value {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:format size:1 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [_device newTextureWithDescriptor:desc];

    vector_float4 values[6];
    for (int i = 0; i < 6; ++i) {
        values[i] = value;
    }
    NSUInteger bytesPerPixel = BytesPerPixel(format);
    for (NSUInteger face = 0; face < 6; ++face) {
        [texture replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
                    mipmapLevel:0
                          slice:face
                      withBytes:&values[face]
                    bytesPerRow:bytesPerPixel
                  bytesPerImage:bytesPerPixel];
    }
    return texture;
}

- (TextureBinding)bindingForTexture:(id<MTLTexture>)texture channels:(uint32_t)channels hdr:(BOOL)hdr {
    if (!texture) {
        TextureBinding binding{};
        return binding;
    }
    TextureBinding binding{};
    binding.handle.value = HandleForTexture(texture);
    binding.info.width = (uint32_t)texture.width;
    binding.info.height = (uint32_t)texture.height;
    binding.info.channels = channels;
    binding.info.isHDR = hdr;

    std::lock_guard<std::mutex> guard(_textureMutex);
    _textureCache[@(binding.handle.value)] = texture;
    return binding;
}

- (id<MTLTexture>)textureForHandle:(TextureHandle)handle fallback:(id<MTLTexture>)fallback {
    if (!handle.value) {
        return fallback;
    }
    std::lock_guard<std::mutex> guard(_textureMutex);
    id<MTLTexture> texture = _textureCache[@(handle.value)];
    return texture ?: fallback;
}

- (TextureBinding)bindingFromPayload:(NSDictionary *)payload slot:(MaterialSlot)slot {
    if (!payload.count) {
        return _defaultBindings[static_cast<size_t>(slot)];
    }

    NSString *path = payload[@"path"];
    NSData *bytes = payload[@"bytes"];
    NSNumber *widthValue = payload[@"width"];
    NSNumber *heightValue = payload[@"height"];
    NSNumber *handleValue = payload[@"metalHandle"];
    NSString *formatString = payload[@"format"];
    NSNumber *channelsOverride = payload[@"channels"];
    NSNumber *isCubeValue = payload[@"isCube"];

    id<MTLTexture> texture = nil;
    if (handleValue) {
        TextureHandle handle{};
        handle.value = handleValue.unsignedLongLongValue;
        texture = [self textureForHandle:handle fallback:nil];
    }

    uint32_t channels = channelsOverride ? channelsOverride.unsignedIntValue : 4;
    BOOL hdr = NO;

    if (!texture && path.length) {
        NSMutableDictionary *options = [@{ MTKTextureLoaderOptionSRGB : @NO,
                                           MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead) } mutableCopy];
        if (isCubeValue.boolValue) {
            options[MTKTextureLoaderOptionCubeLayout] = MTKTextureLoaderCubeLayoutVertical;
        }
        NSError *loadError = nil;
        texture = [_textureLoader newTextureWithContentsOfURL:[NSURL fileURLWithPath:path] options:options error:&loadError];
        if (!texture) {
            NSLog(@"[Neuropbr] Failed to load %@: %@", path, loadError.localizedDescription);
            return _defaultBindings[static_cast<size_t>(slot)];
        }
        channels = ChannelsFromFormat(texture.pixelFormat, channelsOverride);
        hdr = IsHDRFormat(texture.pixelFormat);
    } else if (!texture && bytes && widthValue && heightValue) {
        MTLPixelFormat format = PixelFormatFromString(formatString, MTLPixelFormatRGBA32Float);
        channels = ChannelsFromFormat(format, channelsOverride);
        hdr = IsHDRFormat(format);
        texture = [self textureFromBytes:bytes width:widthValue.unsignedIntegerValue height:heightValue.unsignedIntegerValue format:format];
    }

    if (!texture) {
        return _defaultBindings[static_cast<size_t>(slot)];
    }

    return [self bindingForTexture:texture channels:channels hdr:hdr];
}

- (TextureBinding)bindingFromFile:(NSString *)path
                      fallback:(id<MTLTexture>)fallback
                         usage:(MTLTextureUsage)usage
                      expectCube:(BOOL)expectCube
                     outTexture:(id<MTLTexture> __strong *)outTexture
                           error:(NSError **)error {
    id<MTLTexture> texture = fallback;
    if (path.length) {
        NSMutableDictionary *options = [@{ MTKTextureLoaderOptionSRGB : @NO,
                                           MTKTextureLoaderOptionTextureUsage : @(usage) } mutableCopy];
        if (expectCube) {
            options[MTKTextureLoaderOptionCubeLayout] = MTKTextureLoaderCubeLayoutVertical;
        }
        NSError *loadError = nil;
        texture = [_textureLoader newTextureWithContentsOfURL:[NSURL fileURLWithPath:path] options:options error:&loadError];
        if (!texture) {
            if (error) {
                *error = loadError;
            }
            return TextureBinding{};
        }
    }
    if (!texture) {
        return TextureBinding{};
    }
    if (outTexture) {
        *outTexture = texture;
    }
    return [self bindingForTexture:texture channels:ChannelsFromFormat(texture.pixelFormat, nil) hdr:IsHDRFormat(texture.pixelFormat)];
}

- (BOOL)isHDRFilePath:(NSString *)path {
    if (!path.length) {
        return NO;
    }
    static NSSet<NSString *> *extensions = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        extensions = [NSSet setWithObjects:@"hdr", @"hdri", @"exr", @"pfm", nil];
    });
    NSString *ext = path.pathExtension.lowercaseString;
    return [extensions containsObject:ext];
}

- (id<MTLTexture>)loadHDRTextureFromValue:(id)value error:(NSError **)error {
    if ([value isKindOfClass:[NSString class]]) {
        NSString *path = (NSString *)value;
        if (!path.length) {
            return nil;
        }
        NSDictionary *options = @{ MTKTextureLoaderOptionSRGB : @NO,
                                    MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
                                    MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModePrivate) };
        NSError *loadError = nil;
        id<MTLTexture> texture = [_textureLoader newTextureWithContentsOfURL:[NSURL fileURLWithPath:path]
                                                                    options:options
                                                                      error:&loadError];
        if (!texture && error) {
            *error = loadError;
        }
        return texture;
    }
    if ([value isKindOfClass:[NSDictionary class]]) {
        NSDictionary *payload = (NSDictionary *)value;
        NSNumber *handleValue = payload[@"metalHandle"];
        if (handleValue) {
            TextureHandle handle{};
            handle.value = handleValue.unsignedLongLongValue;
            return [self textureForHandle:handle fallback:nil];
        }
        NSString *path = payload[@"path"];
        if (path.length) {
            return [self loadHDRTextureFromValue:path error:error];
        }
        NSData *bytes = payload[@"bytes"];
        NSNumber *widthValue = payload[@"width"];
        NSNumber *heightValue = payload[@"height"];
        NSString *formatString = payload[@"format"];
        if (bytes && widthValue && heightValue) {
            MTLPixelFormat format = PixelFormatFromString(formatString, MTLPixelFormatRGBA32Float);
            return [self textureFromBytes:bytes
                                     width:widthValue.unsignedIntegerValue
                                    height:heightValue.unsignedIntegerValue
                                     format:format];
        }
    }
    if (error && !*error) {
        *error = [NSError errorWithDomain:@"NPBREnvironment" code:-5 userInfo:@{NSLocalizedDescriptionKey : @"Unsupported HDR payload"}];
    }
    return nil;
}

- (BOOL)generateEnvironmentFromHDRValue:(id)value
                               faceSize:(NSUInteger)faceSize
                         irradianceSize:(NSUInteger)irradianceSize
                        specularSamples:(NSUInteger)specularSamples
                        diffuseSamples:(NSUInteger)diffuseSamples
                               controls:(EnvironmentControls &)controls
                                  error:(NSError **)error {
    if (!_prefilter) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment"
                                          code:-6
                                      userInfo:@{NSLocalizedDescriptionKey : @"Prefilter pipeline unavailable"}];
        }
        return NO;
    }

    NSError *loadError = nil;
    id<MTLTexture> hdrTexture = [self loadHDRTextureFromValue:value error:&loadError];
    if (!hdrTexture) {
        if (error) {
            *error = loadError;
        }
        return NO;
    }

    NSError *prefilterError = nil;
    NPBREnvironmentProducts *products = [_prefilter generateFromHDRTexture:hdrTexture
                                                                  faceSize:faceSize
                                                            irradianceSize:irradianceSize
                                                           specularSamples:specularSamples
                                                           diffuseSamples:diffuseSamples
                                                                     error:&prefilterError];
    if (!products) {
        if (error) {
            *error = prefilterError;
        }
        return NO;
    }

    TextureBinding envBinding = [self bindingForTexture:products.environment channels:4 hdr:YES];
    TextureBinding irrBinding = [self bindingForTexture:products.irradiance channels:4 hdr:YES];
    TextureBinding preBinding = [self bindingForTexture:products.prefiltered channels:4 hdr:YES];
    TextureBinding brdfBinding = [self bindingForTexture:products.brdf channels:2 hdr:NO];

    if (!envBinding.handle.value || !preBinding.handle.value || !irrBinding.handle.value || !brdfBinding.handle.value) {
        if (error) {
            *error = [NSError errorWithDomain:@"NPBREnvironment" code:-7 userInfo:@{NSLocalizedDescriptionKey : @"Failed to bind generated environment textures"}];
        }
        return NO;
    }

    controls.environmentHandle = envBinding.handle;
    controls.irradianceHandle = irrBinding.handle;
    controls.prefilteredHandle = preBinding.handle;
    controls.brdfHandle = brdfBinding.handle;
    controls.lodCount = products.mipLevelCount > 1 ? static_cast<float>(products.mipLevelCount - 1) : 1.0f;

    return YES;
}

- (BOOL)applyEnvironmentDictionary:(NSDictionary *)args error:(NSError **)error {
    NSNumber *environmentId = args[@"environmentId"] ?: @(0);
    NSString *envPath = args[@"environment"];
    NSString *irrPath = args[@"irradiance"];
    NSString *prefilterPath = args[@"prefiltered"];
    NSString *brdfPath = args[@"brdf"];
    id hdrValue = args[@"hdr"] ?: args[@"hdri"];

    BOOL shouldGenerate = hdrValue != nil;
    if (!shouldGenerate && envPath.length && (!irrPath.length || !prefilterPath.length) && [self isHDRFilePath:envPath]) {
        hdrValue = envPath;
        shouldGenerate = YES;
    }

    EnvironmentControls controls{};
    controls.environmentId = environmentId.unsignedIntValue;

    if (shouldGenerate) {
        NSUInteger faceSize = [args[@"faceSize"] unsignedIntegerValue];
        NSUInteger irradianceSize = [args[@"irradianceSize"] unsignedIntegerValue];
        NSUInteger specularSamples = [args[@"specularSamples"] unsignedIntegerValue];
        NSUInteger diffuseSamples = [args[@"diffuseSamples"] unsignedIntegerValue];
        if (![self generateEnvironmentFromHDRValue:hdrValue
                                          faceSize:faceSize
                                    irradianceSize:irradianceSize
                                   specularSamples:specularSamples
                                   diffuseSamples:diffuseSamples
                                          controls:controls
                                             error:error]) {
            return NO;
        }
        [self setEnvironment:controls];
        return YES;
    }

    id<MTLTexture> envTexture = nil;
    id<MTLTexture> irrTexture = nil;
    id<MTLTexture> prefilteredTexture = nil;
    id<MTLTexture> brdfTexture = nil;

    NSError *loadError = nil;
    TextureBinding envBinding = [self bindingFromFile:envPath fallback:_fallbackEnv usage:MTLTextureUsageShaderRead expectCube:YES outTexture:&envTexture error:&loadError];
    if (!envBinding.handle.value && envPath.length) {
        if (error) *error = loadError;
        return NO;
    }

    TextureBinding irrBinding = [self bindingFromFile:irrPath fallback:_fallbackIrradiance usage:MTLTextureUsageShaderRead expectCube:YES outTexture:&irrTexture error:&loadError];
    if (!irrBinding.handle.value && irrPath.length) {
        if (error) *error = loadError;
        return NO;
    }

    TextureBinding preBinding = [self bindingFromFile:prefilterPath fallback:_fallbackPrefilter usage:MTLTextureUsageShaderRead expectCube:YES outTexture:&prefilteredTexture error:&loadError];
    if (!preBinding.handle.value && prefilterPath.length) {
        if (error) *error = loadError;
        return NO;
    }

    TextureBinding brdfBinding = [self bindingFromFile:brdfPath fallback:_fallbackBRDF usage:MTLTextureUsageShaderRead expectCube:NO outTexture:&brdfTexture error:&loadError];
    if (!brdfBinding.handle.value && brdfPath.length) {
        if (error) *error = loadError;
        return NO;
    }

    controls.environmentHandle = envBinding.handle;
    controls.irradianceHandle = irrBinding.handle;
    controls.prefilteredHandle = preBinding.handle;
    controls.brdfHandle = brdfBinding.handle;
    if (prefilteredTexture && prefilteredTexture.mipmapLevelCount > 1) {
        controls.lodCount = static_cast<float>(prefilteredTexture.mipmapLevelCount - 1);
    } else {
        controls.lodCount = 1.0f;
    }

    [self setEnvironment:controls];
    return YES;
}

    NSData *bytes = payload[@"bytes"];
    NSNumber *widthValue = payload[@"width"];
    NSNumber *heightValue = payload[@"height"];
    NSNumber *handleValue = payload[@"metalHandle"];
    NSString *formatString = payload[@"format"];

    id<MTLTexture> texture = nil;
    if (handleValue) {
        texture = [self textureForHandle:{handleValue.unsignedLongLongValue} fallback:nil];
    }

    BOOL isHDR = NO;
    uint32_t channels = 4;

    if (!texture && path.length) {
        NSURL *url = [NSURL fileURLWithPath:path];
        NSError *loadError = nil;
        texture = [_textureLoader newTextureWithContentsOfURL:url
                                                     options:@{ MTKTextureLoaderOptionSRGB : @NO,
                                                                MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead)
                                                     }
                                                       error:&loadError];
        if (!texture) {
            NSLog(@"[Neuropbr] Failed to load texture %@: %@", path, loadError.localizedDescription);
            return _defaultBindings[static_cast<size_t>(slot)];
        }
        channels = ChannelsFromFormat(texture.pixelFormat, payload[@"channels"]);
        isHDR = IsHDRFormat(texture.pixelFormat);
    } else if (!texture && bytes && widthValue && heightValue) {
        MTLPixelFormat format = PixelFormatFromString(formatString, MTLPixelFormatRGBA32Float);
        channels = ChannelsFromFormat(format, payload[@"channels"]);
        isHDR = IsHDRFormat(format);
        texture = [self textureFromBytes:bytes width:widthValue.unsignedIntegerValue height:heightValue.unsignedIntegerValue format:format];
    }

    if (!texture) {
        return _defaultBindings[static_cast<size_t>(slot)];
    }

    return [self bindingForTexture:texture channels:channels hdr:isHDR];
}

- (id<MTLTexture>)textureFromBytes:(NSData *)data width:(NSUInteger)width height:(NSUInteger)height format:(MTLPixelFormat)format {
    MTLTextureDescriptor *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> texture = [_device newTextureWithDescriptor:descriptor];
    NSUInteger bytesPerRow = BytesPerPixel(format) * width;
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 withBytes:data.bytes bytesPerRow:bytesPerRow];
    return texture;
}

- (void)writeUniforms:(const FrameDescriptor &)descriptor {
    GPUFrameUniforms frame{};
    memcpy(frame.viewProjection, descriptor.frame.viewProjection.data(), sizeof(float) * 16);
    memcpy(frame.invView, descriptor.frame.invView.data(), sizeof(float) * 16);
    memcpy(frame.cameraPosTime, descriptor.frame.cameraPosTime.data(), sizeof(float) * 4);
    memcpy(frame.resolutionExposure, descriptor.frame.resolutionExposure.data(), sizeof(float) * 4);
    memcpy(frame.iblParams, descriptor.frame.iblParams.data(), sizeof(float) * 4);
    memcpy(frame.toneMapping, descriptor.frame.toneMapping.data(), sizeof(float) * 4);
    memcpy(_frameUniforms.contents, &frame, sizeof(GPUFrameUniforms));

    GPUMaterialUniforms material{};
    memcpy(material.baseTint, descriptor.material.baseTint.data(), sizeof(float) * 4);
    memcpy(material.scalars, descriptor.material.scalars.data(), sizeof(float) * 4);
    memcpy(material.featureToggles, descriptor.material.featureToggles.data(), sizeof(float) * 4);
    memcpy(_materialUniforms.contents, &material, sizeof(GPUMaterialUniforms));
}

@end

@interface NeuropbrMetalRendererPlugin : NSObject <FlutterPlugin>
@end

@implementation NeuropbrMetalRendererPlugin {
    FlutterMethodChannel *_channel;
    NPBRMetalRendererBridge *_bridge;
}

+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar> *)registrar {
    FlutterMethodChannel *channel = [FlutterMethodChannel methodChannelWithName:@"neuropbr_renderer" binaryMessenger:[registrar messenger]];
    NeuropbrMetalRendererPlugin *instance = [[NeuropbrMetalRendererPlugin alloc] initWithChannel:channel registrar:registrar];
    [registrar addMethodCallDelegate:instance channel:channel];
}

- (instancetype)initWithChannel:(FlutterMethodChannel *)channel registrar:(NSObject<FlutterPluginRegistrar> *)registrar {
    if ((self = [super init])) {
        _channel = channel;
        _bridge = [[NPBRMetalRendererBridge alloc] initWithRegistrar:[registrar textures]];
        __weak typeof(self) weakSelf = self;
        [_bridge setFrameCallback:^(uint64_t textureHandle) {
            __strong typeof(self) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            [strongSelf->_channel invokeMethod:@"onFrameRendered" arguments:@{ @"textureId" : @(strongSelf->_bridge.textureId) }];
        }];
        [_bridge setErrorCallback:^(NSError *error) {
            __strong typeof(self) strongSelf = weakSelf;
            if (!strongSelf) {
                return;
            }
            [strongSelf->_channel invokeMethod:@"onRendererError" arguments:@{ @"message" : error.localizedDescription ?: @"Unknown" }];
        }];
    }
    return self;
}

- (void)handleMethodCall:(FlutterMethodCall *)call result:(FlutterResult)result {
    NSDictionary *args = call.arguments;
    if ([call.method isEqualToString:@"initRenderer"]) {
        [self handleInit:args result:result];
    } else if ([call.method isEqualToString:@"setCamera"]) {
        [self handleSetCamera:args result:result];
    } else if ([call.method isEqualToString:@"setLighting"]) {
        [self handleSetLighting:args result:result];
    } else if ([call.method isEqualToString:@"setPreview"]) {
        [self handleSetPreview:args result:result];
    } else if ([call.method isEqualToString:@"loadMaterial"]) {
        [self handleLoadMaterial:args result:result];
    } else if ([call.method isEqualToString:@"updateMaterial"]) {
        [self handleUpdateMaterial:args result:result];
    } else if ([call.method isEqualToString:@"setEnvironment"]) {
        [self handleSetEnvironment:args result:result];
    } else if ([call.method isEqualToString:@"renderFrame"]) {
        [self handleRender:args result:result];
    } else if ([call.method isEqualToString:@"exportFrame"]) {
        NSData *png = [_bridge snapshotPNG];
        result(png ? png : [FlutterError errorWithCode:@"snapshot_failed" message:@"Snapshot unavailable" details:nil]);
    } else {
        result(FlutterMethodNotImplemented);
    }
}

- (void)handleInit:(NSDictionary *)args result:(FlutterResult)result {
    NSNumber *width = args[@"width"];
    NSNumber *height = args[@"height"];
    if (!width || !height) {
        result([FlutterError errorWithCode:@"invalid_args" message:@"width/height required" details:nil]);
        return;
    }
    NSError *error = nil;
    if (![_bridge initializeWithSize:CGSizeMake(width.doubleValue, height.doubleValue) error:&error]) {
        result([FlutterError errorWithCode:@"init_failed" message:error.localizedDescription details:nil]);
        return;
    }
    result(@{ @"textureId" : @(_bridge.textureId),
              @"device" : _bridge.deviceName ?: @"Unknown" });
}

- (void)handleSetCamera:(NSDictionary *)args result:(FlutterResult)result {
    CameraParameters camera{};
    [self fillFloat3:args[@"position"] target:camera.position defaultValue:0.0f];
    [self fillFloat3:args[@"target"] target:camera.target defaultValue:0.0f];
    BOOL hasUp = [args[@"up"] isKindOfClass:NSArray.class];
    [self fillFloat3:args[@"up"] target:camera.up defaultValue:0.0f];
    if (!hasUp) {
        camera.up[0] = 0.0f;
        camera.up[1] = 1.0f;
        camera.up[2] = 0.0f;
    }
    camera.fovY = args[@"fov"] ? args[@"fov"].floatValue : 45.0f;
    camera.nearZ = args[@"near"] ? args[@"near"].floatValue : 0.01f;
    camera.farZ = args[@"far"] ? args[@"far"].floatValue : 100.0f;
    [_bridge setCamera:camera];
    result(nil);
}

- (void)handleSetLighting:(NSDictionary *)args result:(FlutterResult)result {
    LightingControls lighting{};
    lighting.exposure = args[@"exposure"] ? args[@"exposure"].floatValue : 0.0f;
    lighting.intensity = args[@"intensity"] ? args[@"intensity"].floatValue : 1.0f;
    lighting.rotation = args[@"rotation"] ? args[@"rotation"].floatValue : 0.0f;
    [_bridge setLighting:lighting];
    result(nil);
}

- (void)handleSetPreview:(NSDictionary *)args result:(FlutterResult)result {
    PreviewControls controls{};
    [self fillFloat3:args[@"tint"] target:controls.baseColorTint defaultValue:1.0f];
    controls.roughnessMultiplier = args[@"roughnessMultiplier"] ? args[@"roughnessMultiplier"].floatValue : 1.0f;
    controls.metallicMultiplier = args[@"metallicMultiplier"] ? args[@"metallicMultiplier"].floatValue : 1.0f;
    controls.enableNormalMap = args[@"enableNormal"] ? args[@"enableNormal"].boolValue : true;
    controls.showNormals = args[@"showNormals"] ? args[@"showNormals"].boolValue : false;
    controls.showWireframe = args[@"showWireframe"] ? args[@"showWireframe"].boolValue : false;
    controls.channel = PreviewChannelFromNumber(args[@"channel"]);
    NSNumber *tone = args[@"toneMapping"];
    if (tone) {
        controls.toneMapping = tone.unsignedIntegerValue == static_cast<uint32_t>(ToneMapping::Filmic) ? ToneMapping::Filmic
                                                                                                         : ToneMapping::ACES;
    }
    [_bridge setPreview:controls];
    result(nil);
}

- (void)handleLoadMaterial:(NSDictionary *)args result:(FlutterResult)result {
    NSNumber *materialId = args[@"materialId"];
    NSDictionary *textures = args[@"textures"];
    if (!materialId || ![textures isKindOfClass:NSDictionary.class]) {
        result([FlutterError errorWithCode:@"invalid_material" message:@"materialId/textures missing" details:nil]);
        return;
    }
    [_bridge loadMaterial:materialId.unsignedIntValue payloads:textures];
    result(nil);
}

- (void)handleUpdateMaterial:(NSDictionary *)args result:(FlutterResult)result {
    NSNumber *materialId = args[@"materialId"];
    NSString *slotKey = args[@"slot"];
    NSDictionary *payload = args[@"payload"];
    if (!materialId || !slotKey) {
        result([FlutterError errorWithCode:@"invalid_material" message:@"materialId/slot required" details:nil]);
        return;
    }
    [_bridge updateMaterial:materialId.unsignedIntValue slot:MaterialSlotFromString(slotKey) payload:payload ?: @{}];
    result(nil);
}

- (void)handleSetEnvironment:(NSDictionary *)args result:(FlutterResult)result {
    NSError *error = nil;
    if (![_bridge applyEnvironmentDictionary:args error:&error]) {
        result([FlutterError errorWithCode:@"environment_failed" message:error.localizedDescription ?: @"Environment error" details:nil]);
        return;
    }
    result(nil);
}

- (void)handleRender:(NSDictionary *)args result:(FlutterResult)result {
    NSNumber *materialId = args[@"materialId"];
    if (!materialId) {
        result([FlutterError errorWithCode:@"invalid_material" message:@"materialId required" details:nil]);
        return;
    }
    BOOL ok = [_bridge renderMaterial:materialId.unsignedIntValue];
    result(ok ? @YES : [FlutterError errorWithCode:@"render_failed" message:@"Renderer unavailable" details:nil]);
}

- (void)fillFloat3:(NSArray *)source target:(float[3])target defaultValue:(float)value {
    if (![source isKindOfClass:NSArray.class] || source.count < 3) {
        target[0] = value;
        target[1] = value;
        target[2] = value;
        return;
    }
    target[0] = [source[0] floatValue];
    target[1] = [source[1] floatValue];
    target[2] = [source[2] floatValue];
}

@end
