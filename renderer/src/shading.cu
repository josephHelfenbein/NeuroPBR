#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "utils.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


extern "C" __global__
void shadeKernel(cudaTextureObject_t envTex,
                  cudaTextureObject_t irradianceTex,
                  cudaTextureObject_t brdfLutTex,
                  const float* albedo, const float* normal,
                  const float* roughness, const float* metallic,
                  float* outRGBA, int W, int H,
                  float camX, float camY, float camZ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    float ax = albedo[3*idx+0], ay = albedo[3*idx+1], az = albedo[3*idx+2];
    float nx = normal[3*idx+0], ny = normal[3*idx+1], nz = normal[3*idx+2];

    float nlen = sqrtf(nx*nx + ny*ny + nz*nz) + 1e-8f;
    nx /= nlen;
    ny /= nlen;
    nz /= nlen;

    float r = roughness[idx];
    float m = metallic[idx];

    float px = ((x + 0.5f) / W - 0.5f) * 1.0f;
    float py = ((y + 0.5f) / H - 0.5f) * 1.0f;
    float pz = 0.0f;
    float vx = camX - px, vy = camY - py, vz = camZ - pz;
    float vlen = sqrtf(vx*vx+vy*vy+vz*vz)+1e-8f;
    vx/=vlen;
    vy/=vlen;
    vz/=vlen;

    float NdotV = fmaxf(nx*vx + ny*vy + nz*vz, 0.0f);
    float dotVN = vx*nx + vy*ny + vz*nz;
    float rx = vx - 2.0f * dotVN * nx;
    float ry = vy - 2.0f * dotVN * ny;
    float rz = vz - 2.0f * dotVN * nz;

    float F0x = 0.04f*(1.0f - m) + ax * m;
    float F0y = 0.04f*(1.0f - m) + ay * m;
    float F0z = 0.04f*(1.0f - m) + az * m;

    float iu, iv;
    dirToLatLong(nx, ny, nz, &iu, &iv);
    float4 irrSample = tex2D<float4>(irradianceTex, iu, iv);
    float irradiance_r = irrSample.x;
    float irradiance_g = irrSample.y;
    float irradiance_b = irrSample.z;

    float3 diffuse;
    diffuse.x = irradiance_r * ax * (1.0f - m) / M_PI;
    diffuse.y = irradiance_g * ay * (1.0f - m) / M_PI;
    diffuse.z = irradiance_b * az * (1.0f - m) / M_PI;

    float eu, ev;
    dirToLatLong(rx, ry, rz, &eu, &ev);
    float4 envSample = tex2D<float4>(envTex, eu, ev);
    float envx = envSample.x;
    float envy = envSample.y;
    float envz = envSample.z;

    float lutU = NdotV;
    float lutV = r;
    float lutx = tex2D<float2>(brdfLutTex, lutU, lutV).x;
    float luty = tex2D<float2>(brdfLutTex, lutU, lutV).y;

    float spec_r = envx * (F0x * lutx + luty);
    float spec_g = envy * (F0y * lutx + luty);
    float spec_b = envz * (F0z * lutx + luty);

    float outR = diffuse.x + spec_r;
    float outG = diffuse.y + spec_g;
    float outB = diffuse.z + spec_b;

    outRGBA[4*idx + 0] = outR;
    outRGBA[4*idx + 1] = outG;
    outRGBA[4*idx + 2] = outB;
    outRGBA[4*idx + 3] = 1.0f;
}
