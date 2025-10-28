#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <utils.cuh>
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" __global__
void precomputeBRDF(cudaSurfaceObject_t brdfLutSurf, int W, int H);

void launchPrecomputeBRDF(dim3 gridDim, dim3 blockDim,
						  cudaSurfaceObject_t brdfLutSurf, int width, int height);