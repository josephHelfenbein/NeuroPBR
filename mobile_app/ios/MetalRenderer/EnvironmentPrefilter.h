#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface NPBREnvironmentProducts : NSObject
@property(nonatomic, strong) id<MTLTexture> environment;
@property(nonatomic, strong) id<MTLTexture> irradiance;
@property(nonatomic, strong) id<MTLTexture> prefiltered;
@property(nonatomic, strong) id<MTLTexture> brdf;
@property(nonatomic, assign) NSUInteger mipLevelCount;
@end

@interface NPBREnvironmentPrefilter : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device
                       library:(id<MTLLibrary>)library
                  commandQueue:(id<MTLCommandQueue>)queue;

/// Generate environment products from an HDR texture (live computation)
- (nullable NPBREnvironmentProducts *)generateFromHDRTexture:(id<MTLTexture>)hdrTexture
                                                    faceSize:(NSUInteger)faceSize
                                              irradianceSize:(NSUInteger)irradianceSize
                                             specularSamples:(NSUInteger)specularSamples
                                             diffuseSamples:(NSUInteger)diffuseSamples
                                                       error:(NSError **)error;

/// Load precomputed environment maps from disk (KTX files)
- (nullable NPBREnvironmentProducts *)loadPrecomputedEnvironment:(NSString *)environmentPath
                                                  irradiancePath:(NSString *)irradiancePath
                                                 prefilteredPath:(NSString *)prefilteredPath
                                                        brdfPath:(nullable NSString *)brdfPath
                                                           error:(NSError **)error;

/// Load a KTX cubemap texture from disk
- (nullable id<MTLTexture>)loadKTXCubemap:(NSString *)path error:(NSError **)error;

/// Load a 2D texture (for BRDF LUT) from disk
- (nullable id<MTLTexture>)loadTexture2D:(NSString *)path error:(NSError **)error;

- (id<MTLTexture>)sharedBRDFLUT;

@end

NS_ASSUME_NONNULL_END
