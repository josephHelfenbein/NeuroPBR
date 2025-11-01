#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <utils.cuh>
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" __global__
void equirectangularToCubemap(cudaTextureObject_t hdrTex, cudaSurfaceObject_t cubemapSurface, int faceSize);

extern "C" __global__
void prefilterSpecularCubemap(cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface, int faceSize, float roughness, unsigned int sampleCount);

extern "C" __global__
void convolveDiffuseIrradiance(cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface, int faceSize, unsigned int sampleCount);

extern "C" __global__
void computeEnvironmentBrightness(cudaTextureObject_t envCubemap,
								  int faceSize,
								  float horizonMinY,
								  float horizonMaxY,
								  float zenithMinY,
								  float* blockAccum);

extern "C" __global__
void reduceEnvironmentBrightness(const float* blockAccum,
								 int blockCount,
								 float* outAccum);

void launchEquirectangularToCubemap(dim3 gridDim, dim3 blockDim,
									cudaTextureObject_t hdrTex, cudaSurfaceObject_t cubemapSurface,
									int faceSize);

void launchPrefilterSpecularCubemap(dim3 gridDim, dim3 blockDim,
									cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface,
									int faceSize, float roughness, unsigned int sampleCount);

void launchConvolveDiffuseIrradiance(dim3 gridDim, dim3 blockDim,
									 cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface,
									 int faceSize, unsigned int sampleCount);

void launchComputeEnvironmentBrightness(dim3 gridDim, dim3 blockDim,
										cudaTextureObject_t envCubemap,
										int faceSize,
										float horizonMinY,
										float horizonMaxY,
										float zenithMinY,
										float* blockAccum);

void launchReduceEnvironmentBrightness(const float* blockAccum,
									   int blockCount,
									   float* outAccum);