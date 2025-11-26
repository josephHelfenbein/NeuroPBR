#include <metal_stdlib>
using namespace metal;

constant float kPi = 3.14159265358979323846264338327950288;
#define M_PI kPi

namespace {

float3 normalizeSafe(float3 v) {
    float len = length(v);
    return len > 0.0f ? v / len : float3(0.0f, 0.0f, 0.0f);
}

float3 faceUvToDir(uint face, float u, float v) {
    switch (face) {
        case 0: return normalizeSafe(float3(1.0f, -v, -u));
        case 1: return normalizeSafe(float3(-1.0f, -v, u));
        case 2: return normalizeSafe(float3(u, 1.0f, v));
        case 3: return normalizeSafe(float3(u, -1.0f, -v));
        case 4: return normalizeSafe(float3(u, -v, 1.0f));
        default: return normalizeSafe(float3(-u, -v, -1.0f));
    }
}

float2 dirToLatLong(float3 dir) {
    dir = normalizeSafe(dir);
    float theta = atan2(dir.z, dir.x);
    float phi = asin(clamp(dir.y, -1.0f, 1.0f));
    float u = 0.5f + theta * (1.0f / (2.0f * M_PI));
    float v = 0.5f - phi * (1.0f / M_PI);
    return float2(fract(u), clamp(v, 0.0f, 1.0f));
}

float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

float2 hammersley(uint i, uint N) {
    return float2(float(i) / float(N), radicalInverseVdC(i));
}

float random(float2 co) {
    return fract(sin(dot(co, float2(12.9898, 78.233))) * 43758.5453);
}

void buildTangentBasis(float3 N, thread float3 &T, thread float3 &B) {
    float sign = N.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = float3(b, sign + N.y * N.y * a, -N.y);
}

float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0f * float(M_PI) * Xi.x;
    float cosTheta = sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    float3 T, B;
    buildTangentBasis(N, T, B);
    float3 sample = T * H.x + B * H.y + N * H.z;
    return normalizeSafe(sample);
}

float3 cosineSampleHemisphere(float2 Xi, float3 N) {
    float r = sqrt(Xi.x);
    float phi = 2.0f * float(M_PI) * Xi.y;
    float x = r * cos(phi);
    float y = r * sin(phi);
    float z = sqrt(max(0.0f, 1.0f - Xi.x));
    float3 T, B;
    buildTangentBasis(N, T, B);
    float3 sample = T * x + B * y + N * z;
    return normalizeSafe(sample);
}

} // namespace

struct PrefilterUniforms {
    uint faceSize;
    uint sampleCount;
    float roughness;
    uint mipLevel;
};

struct IrradianceUniforms {
    uint faceSize;
    uint sampleCount;
};

kernel void equirectangularToCubemapKernel(texture2d<float, access::sample> hdrTexture [[texture(0)]],
                                           texture2d_array<float, access::write> cubeFaces [[texture(1)]],
                                           uint3 gid [[thread_position_in_grid]]) {
    uint faceSize = cubeFaces.get_width();
    if (gid.x >= faceSize || gid.y >= faceSize || gid.z >= cubeFaces.get_array_size()) {
        return;
    }
    
    constexpr sampler hdrSampler(coord::normalized, s_address::repeat, t_address::clamp_to_edge, filter::linear);
    
    float2 uv = (float2(gid.xy) + 0.5f) / float(faceSize);
    float2 uvRemapped = float2(uv * 2.0f - 1.0f);
    float3 dir = faceUvToDir(gid.z, uvRemapped.x, uvRemapped.y);
    float2 latlong = dirToLatLong(dir);
    float4 color = hdrTexture.sample(hdrSampler, latlong);
    
    // Sanitize input to avoid blue pixels/NaNs
    if (isnan(color.x) || isinf(color.x)) color.x = 0.0f;
    if (isnan(color.y) || isinf(color.y)) color.y = 0.0f;
    if (isnan(color.z) || isinf(color.z)) color.z = 0.0f;
    
    color.w = 1.0f; // Force alpha to 1.0
    cubeFaces.write(color, uint2(gid.xy), gid.z);
}

float distributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    return a2 / (M_PI * denom * denom);
}

kernel void prefilterSpecularKernel(texturecube<float, access::sample> envMap [[texture(0)]],
                                    texture2d_array<float, access::write> prefilteredFaces [[texture(1)]],
                                    constant PrefilterUniforms &uniforms [[buffer(0)]],
                                    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.faceSize || gid.y >= uniforms.faceSize || gid.z >= prefilteredFaces.get_array_size()) {
        return;
    }
    float2 uv = (float2(gid.xy) + 0.5f) / float(uniforms.faceSize);
    float2 uvRemapped = float2(uv * 2.0f - 1.0f);
    float3 N = faceUvToDir(gid.z, uvRemapped.x, uvRemapped.y);
    float3 R = N;
    float3 V = R;

    constexpr sampler cubeSampler(filter::linear, mip_filter::linear);
    float3 prefiltered = float3(0.0f);
    float totalWeight = 0.0f;
    for (uint i = 0; i < uniforms.sampleCount; ++i) {
        float2 Xi = hammersley(i, uniforms.sampleCount);
        float3 H = importanceSampleGGX(Xi, N, uniforms.roughness);
        float3 L = normalizeSafe(2.0f * dot(V, H) * H - V);
        float NdotL = clamp(dot(N, L), 0.0f, 1.0f);
        if (NdotL > 0.0f) {
            // PDF based mip level selection to reduce fireflies
            float NdotH = clamp(dot(N, H), 0.0f, 1.0f);
            float HdotV = clamp(dot(H, V), 0.0f, 1.0f);
            float D = distributionGGX(NdotH, uniforms.roughness);
            float pdf = (D * NdotH) / (4.0f * HdotV) + 0.0001f;
            
            float resolution = float(uniforms.faceSize);
            float saTexel = 4.0f * M_PI / (6.0f * resolution * resolution);
            float saSample = 1.0f / (float(uniforms.sampleCount) * pdf + 0.0001f);
            float mipLevel = uniforms.roughness == 0.0f ? 0.0f : 0.5f * log2(saSample / saTexel);

            float3 sampleColor = envMap.sample(cubeSampler, L, level(mipLevel)).xyz;
            // Sanitize
            if (isnan(sampleColor.x) || isinf(sampleColor.x)) sampleColor.x = 0.0f;
            if (isnan(sampleColor.y) || isinf(sampleColor.y)) sampleColor.y = 0.0f;
            if (isnan(sampleColor.z) || isinf(sampleColor.z)) sampleColor.z = 0.0f;

            prefiltered += sampleColor * NdotL;
            totalWeight += NdotL;
        }
    }
    float inv = totalWeight > 0.0f ? 1.0f / totalWeight : 0.0f;
    float4 outColor = float4(prefiltered * inv, 1.0f);
    prefilteredFaces.write(outColor, uint2(gid.xy), gid.z);
}

kernel void convolveDiffuseKernel(texturecube<float, access::sample> envMap [[texture(0)]],
                                  texture2d_array<float, access::write> irradianceFaces [[texture(1)]],
                                  constant IrradianceUniforms &uniforms [[buffer(0)]],
                                  uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.faceSize || gid.y >= uniforms.faceSize || gid.z >= irradianceFaces.get_array_size()) {
        return;
    }
    float2 uv = (float2(gid.xy) + 0.5f) / float(uniforms.faceSize);
    float2 uvRemapped = float2(uv * 2.0f - 1.0f);
    float3 N = faceUvToDir(gid.z, uvRemapped.x, uvRemapped.y);

    constexpr sampler cubeSampler(filter::linear, mip_filter::linear);
    
    float envWidth = float(envMap.get_width());
    float targetWidth = sqrt(max(1.0f, float(uniforms.sampleCount)));
    float mipLevel = max(0.0f, log2(envWidth / targetWidth));

    float3 irradiance = float3(0.0f);
    float rand = random(uv);
    for (uint i = 0; i < uniforms.sampleCount; ++i) {
        float2 Xi = hammersley(i, uniforms.sampleCount);
        Xi.y = fract(Xi.y + rand);
        float3 L = cosineSampleHemisphere(Xi, N);
        float NdotL = clamp(dot(N, L), 0.0f, 1.0f);
        if (NdotL > 0.0f) {
            // Use calculated mip level
            float3 sampleColor = envMap.sample(cubeSampler, L, level(mipLevel)).xyz;
            // Sanitize
            if (isnan(sampleColor.x) || isinf(sampleColor.x)) sampleColor.x = 0.0f;
            if (isnan(sampleColor.y) || isinf(sampleColor.y)) sampleColor.y = 0.0f;
            if (isnan(sampleColor.z) || isinf(sampleColor.z)) sampleColor.z = 0.0f;

            irradiance += sampleColor;
        }
    }
    float scale = float(M_PI) / float(uniforms.sampleCount);
    float4 outColor = float4(irradiance * scale, 1.0f);
    irradianceFaces.write(outColor, uint2(gid.xy), gid.z);
}

struct BRDFUniforms {
    uint width;
    uint height;
};

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    float denom = NdotV * (1.0f - k) + k;
    return NdotV / denom;
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotL = clamp(dot(N, L), 0.0f, 1.0f);
    float NdotV = clamp(dot(N, V), 0.0f, 1.0f);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    return ggx1 * ggx2;
}

float2 integrateBRDF(float NdotV, float roughness) {
    float3 V = float3(sqrt(max(0.0f, 1.0f - NdotV * NdotV)), 0.0f, NdotV);
    float A = 0.0f;
    float B = 0.0f;
    float3 N = float3(0.0f, 0.0f, 1.0f);
    const uint sampleCount = 1024u;
    for (uint i = 0; i < sampleCount; ++i) {
        float2 Xi = hammersley(i, sampleCount);
        float3 H = importanceSampleGGX(Xi, N, roughness);
        float3 L = normalizeSafe(2.0f * dot(V, H) * H - V);
        float NdotL = clamp(L.z, 0.0f, 1.0f);
        float NdotH = clamp(H.z, 0.0f, 1.0f);
        float VdotH = clamp(dot(V, H), 0.0f, 1.0f);
        if (NdotL > 0.0f) {
            float G = geometrySmith(N, V, L, roughness);
            float G_Vis = (G * VdotH) / max(NdotH * NdotV, 1e-4f);
            float Fc = pow(1.0f - VdotH, 5.0f);
            A += (1.0f - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    float inv = 1.0f / float(sampleCount);
    return float2(A * inv, B * inv);
}

kernel void precomputeBRDFKernel(texture2d<float, access::write> brdfLut [[texture(0)]],
                                 constant BRDFUniforms &uniforms [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }
    float NdotV = (float(gid.x) + 0.5f) / float(uniforms.width);
    float roughness = (float(gid.y) + 0.5f) / float(uniforms.height);
    float2 result = integrateBRDF(NdotV, roughness);
    brdfLut.write(float4(result, 0.0f, 1.0f), gid);
}
