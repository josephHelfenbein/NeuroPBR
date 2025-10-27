#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ inline void dirToLatLong(float dx, float dy, float dz, float *u, float *v) {
    float theta = atan2f(dz, dx);
    float clamp_y = fmaxf(-1.0f, fminf(1.0f, dy));
    float phi = asinf(clamp_y);

    *u = 0.5f + theta * (1.0f / (2.0f * M_PI));
    *v = 0.5f - phi * (1.0f / M_PI);

    *u = *u - floorf(*u);
    if (*v < 0.0f) *v = 0.0f;
    if (*v > 1.0f) *v = 1.0f;
}

extern "C" __global__
void shadeKernel(cudaTextureObject_t env_tex,
                  cudaTextureObject_t irradiance_tex,
                  cudaTextureObject_t brdf_lut_tex,
                  const float* albedo, const float* normal_map,
                  const float* roughness, const float* metallic,
                  float* out_rgba, int W, int H,
                  float cam_x, float cam_y, float cam_z) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx = y * W + x;

    float ax = albedo[3*idx+0], ay = albedo[3*idx+1], az = albedo[3*idx+2];
    float nx = normal_map[3*idx+0], ny = normal_map[3*idx+1], nz = normal_map[3*idx+2];

    float nlen = sqrtf(nx*nx + ny*ny + nz*nz) + 1e-8f;
    nx /= nlen;
    ny /= nlen;
    nz /= nlen;

    float r = roughness[idx];
    float m = metallic[idx];

    float px = ((x + 0.5f) / W - 0.5f) * 1.0f;
    float py = ((y + 0.5f) / H - 0.5f) * 1.0f;
    float pz = 0.0f;
    float vx = cam_x - px, vy = cam_y - py, vz = cam_z - pz;
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
    float4 irrSample = tex2D<float4>(irradiance_tex, iu, iv);
    float irradiance_r = irrSample.x;
    float irradiance_g = irrSample.y;
    float irradiance_b = irrSample.z;

    float3 diffuse;
    diffuse.x = irradiance_r * ax * (1.0f - m) / M_PI;
    diffuse.y = irradiance_g * ay * (1.0f - m) / M_PI;
    diffuse.z = irradiance_b * az * (1.0f - m) / M_PI;

    float eu, ev;
    dirToLatLong(rx, ry, rz, &eu, &ev);
    float4 envSample = tex2D<float4>(env_tex, eu, ev);
    float envx = envSample.x;
    float envy = envSample.y;
    float envz = envSample.z;

    float lut_u = NdotV;
    float lut_v = r;
    float lutx = tex2D<float2>(brdf_lut_tex, lut_u, lut_v).x;
    float luty = tex2D<float2>(brdf_lut_tex, lut_u, lut_v).y;

    float spec_r = envx * (F0x * lutx + luty);
    float spec_g = envy * (F0y * lutx + luty);
    float spec_b = envz * (F0z * lutx + luty);

    float out_r = diffuse.x + spec_r;
    float out_g = diffuse.y + spec_g;
    float out_b = diffuse.z + spec_b;

    out_rgba[4*idx + 0] = out_r;
    out_rgba[4*idx + 1] = out_g;
    out_rgba[4*idx + 2] = out_b;
    out_rgba[4*idx + 3] = 1.0f;
}
