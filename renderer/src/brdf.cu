#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__device__ inline float3 makeNormalized(const float3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z) + 1e-8f;
    return make_float3(v.x/len, v.y/len, v.z/len);
}

__device__ inline float dot3(const float3 a, const float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 cross3(const float3 a, const float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ inline float radicalInverseVdC(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

__device__ inline float2 hammersley(unsigned int i, unsigned int N) {
    return make_float2(float(i) / float(N), radicalInverseVdC(i));
}

__device__ inline float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0f * M_PI * Xi.x;
    float cosTheta = sqrtf((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
    float3 up = fabsf(N.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tangent = makeNormalized(cross3(up, N));
    float3 bitangent = cross3(N, tangent);
    float3 sampleVec = make_float3(
        tangent.x * H.x + bitangent.x * H.y + N.x * H.z,
        tangent.y * H.x + bitangent.y * H.y + N.y * H.z,
        tangent.z * H.x + bitangent.z * H.y + N.z * H.z
    );
    return makeNormalized(sampleVec);
}

__device__ float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    float denom = NdotV * (1.0f - k) + k;
    return NdotV / denom;
}

__device__ float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotL = fmaxf(dot3(N, L), 0.0f);
    float NdotV = fmaxf(dot3(N, V), 0.0f);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    return ggx1 * ggx2;
}

__device__ float2 integrateBRDF(float NdotV, float roughness) {
    float3 V = make_float3(sqrtf(1.0f - NdotV * NdotV), 0.0f, NdotV);
    float A = 0.0f;
    float B = 0.0f;
    float3 N = make_float3(0.0f, 0.0f, 1.0f);
    const unsigned int SAMPLE_COUNT = 1024u;
    for(unsigned int i=0u; i<SAMPLE_COUNT; i++){
        float2 Xi = hammersley(i, SAMPLE_COUNT);
        float3 H = importanceSampleGGX(Xi, N, roughness);
        float VdotH = fmaxf(dot3(V, H), 0.0f);
        float3 L = make_float3(
            2.0f * VdotH * H.x - V.x,
            2.0f * VdotH * H.y - V.y,
            2.0f * VdotH * H.z - V.z
        );
        L = makeNormalized(L);
        float NdotL = fmaxf(L.z, 0.0f);
        float NdotH = fmaxf(H.z, 0.0f);
        if(NdotL > 0.0f){
            float G = geometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = powf(1.0f - VdotH, 5.0f);
            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(SAMPLE_COUNT);
    B /= float(SAMPLE_COUNT);
    return make_float2(A, B);
}

extern "C" __global__
void precomputeBRDF(cudaSurfaceObject_t brdf_lut_surf, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float NdotV = (float(x) + 0.5f) / float(W);
    float roughness = (float(y) + 0.5f) / float(H);

    float2 integratedBRDF = integrateBRDF(NdotV, roughness);

    surf2Dwrite(integratedBRDF, brdf_lut_surf, x * (int)sizeof(float2), y);
}