#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {
    float4x4 viewProjection;
    float4x4 invView;
    float4 cameraPosTime;
    float4 resolutionExposure; // xy = size, z = exposure
    float4 iblParams; // x intensity, y rotation, z lodCount
    float4 toneMapping; // x tone enum, y exposure
};

struct MaterialUniforms {
    float4 baseTint;
    float4 scalars; // x rough mul, y metal mul, z unused, w channel
    float4 featureToggle; // x normal, y unused, z show normals, w wireframe
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

constant float3x3 kTBN = float3x3(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));

vertex VertexOut fullscreen_vertex(uint vertexID [[vertex_id]]) {
    const float2 quad[4] = {
        float2(-1.0, -1.0),
        float2(1.0, -1.0),
        float2(-1.0, 1.0),
        float2(1.0, 1.0)
    };
    VertexOut out;
    out.position = float4(quad[vertexID], 0.0, 1.0);
    out.uv = float2(0.5, 0.5) + 0.5 * quad[vertexID];
    return out;
}

float3 applyNormalMap(float3 normalSample) {
    float3 n = normalSample * 2.0 - 1.0;
    return normalize(kTBN * n);
}

float distributionGGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(denom * denom * 3.14159265, 1e-4);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float g1 = NdotV / max(NdotV * (1.0 - k) + k, 1e-4);
    float g2 = NdotL / max(NdotL * (1.0 - k) + k, 1e-4);
    return g1 * g2;
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 toneMap(float3 color, constant FrameUniforms &frame) {
    float exposure = exp2(frame.toneMapping.y);
    color *= exposure;
    if (frame.toneMapping.x < 0.5) {
        // ACES approximation
        const float a = 2.51;
        const float b = 0.03;
        const float c = 2.43;
        const float d = 0.59;
        const float e = 0.14;
        color = (color * (a * color + b)) / (color * (c * color + d) + e);
    } else {
        // Filmic
        color = (color * (6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);
    }
    return pow(max(color, float3(0.0)), float3(1.0 / 2.2));
}

fragment float4 pbr_fragment(
    VertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(0)]],
    constant MaterialUniforms &material [[buffer(1)]],
    textureCube<float> envMap [[texture(0)]],
    textureCube<float> irradianceMap [[texture(1)]],
    textureCube<float> prefilteredMap [[texture(2)]],
    texture2d<float> brdfLUT [[texture(3)]],
    texture2d<float> albedoMap [[texture(4)]],
    texture2d<float> normalMap [[texture(5)]],
    texture2d<float> roughnessMap [[texture(6)]],
    texture2d<float> metallicMap [[texture(7)]]
) {
    constexpr sampler linearSampler(filter::linear, address::repeat);
    constexpr sampler clampSampler(filter::linear, address::clamp_to_edge);

    float2 uv = in.uv;

    float3 albedo = albedoMap.sample(linearSampler, uv).rgb * material.baseTint.rgb;
    float roughness = clamp(roughnessMap.sample(linearSampler, uv).r * material.scalars.x, 0.02, 1.0);
    float metallic = clamp(metallicMap.sample(linearSampler, uv).r * material.scalars.y, 0.0, 1.0);

    float3 N = float3(0.0, 0.0, 1.0);
    if (material.featureToggle.x > 0.5) {
        float3 mapN = normalMap.sample(linearSampler, uv).rgb;
        N = applyNormalMap(mapN);
    }

    float3 V = normalize(float3(0.0, 0.0, 1.0));
    float3 R = reflect(-V, N);

    float3 F0 = mix(float3(0.04), albedo, metallic);

    float3 irradiance = irradianceMap.sample(clampSampler, N).rgb;
    float3 diffuse = irradiance * albedo;

    float lod = roughness * frame.iblParams.z;
    float3 prefiltered = prefilteredMap.sample(clampSampler, R, level(lod)).rgb;
    float2 brdf = brdfLUT.sample(clampSampler, float2(max(dot(N, V), 0.0), roughness)).rg;
    float3 specular = prefiltered * (F0 * brdf.x + brdf.y);

    float3 color = (diffuse * (1.0 - metallic) + specular) * frame.iblParams.x;

    // Channel inspector overrides
    if (material.scalars.w > 0.5) {
        if (material.scalars.w < 1.5) color = albedo;
        else if (material.scalars.w < 2.5) color = float3(roughness);
        else if (material.scalars.w < 3.5) color = float3(metallic);
        else color = N * 0.5 + 0.5;
    }

    if (material.featureToggle.z > 0.5) {
        color = N * 0.5 + 0.5;
    }

    if (material.featureToggle.w > 0.5) {
        float grid = abs(fract(uv * 20.0 - 0.5) - 0.5) / fwidth(uv * 20.0);
        grid = smoothstep(0.0, 1.0, grid);
        color = mix(color, float3(1.0, 0.18, 0.18), clamp(1.0 - grid, 0.0, 1.0));
    }

    color = toneMap(color, frame);
    return float4(color, 1.0);
}
