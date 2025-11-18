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

- (nullable NPBREnvironmentProducts *)generateFromHDRTexture:(id<MTLTexture>)hdrTexture
                                                    faceSize:(NSUInteger)faceSize
                                              irradianceSize:(NSUInteger)irradianceSize
                                             specularSamples:(NSUInteger)specularSamples
                                             diffuseSamples:(NSUInteger)diffuseSamples
                                                       error:(NSError **)error;

- (id<MTLTexture>)sharedBRDFLUT;

@end

NS_ASSUME_NONNULL_END
