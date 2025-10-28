#include "brdf.cuh"

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
void precomputeBRDF(cudaSurfaceObject_t brdfLutSurf, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float NdotV = (float(x) + 0.5f) / float(W);
    float roughness = (float(y) + 0.5f) / float(H);

    float2 integratedBRDF = integrateBRDF(NdotV, roughness);

    surf2Dwrite(integratedBRDF, brdfLutSurf, x * (int)sizeof(float2), y);
}

void launchPrecomputeBRDF(dim3 gridDim, dim3 blockDim,
                          cudaSurfaceObject_t brdfLutSurf, int width, int height) {
    precomputeBRDF<<<gridDim, blockDim>>>(brdfLutSurf, width, height);
}