#include <metal_stdlib>
using namespace metal;

struct FrameUniforms {
    float4x4 cameraToWorld;
    float4x4 worldToCamera;
    float4x4 projection;
    float4 cameraPosFov; // xyz = pos, w = fov
    float4 resolutionExposure; // xy = size, z = exposure
    float4 iblParams; // x intensity, y rotation, z lodCount
    float4 toneMapping; // x tone enum, y exposure
};

struct MaterialUniforms {
    float4 baseTint;
    float4 scalars; // x rough mul, y metal mul, z unused, w channel
    float4 featureToggle; // x normal, y unused, z show normals, w wireframe
};

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
};

struct BackgroundVertexOut {
    float4 position [[position]];
    float2 uv;
};

constant float3x3 kTBN = float3x3(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), float3(0.0, 0.0, 1.0));
constant float kPi = 3.14159265359;

vertex BackgroundVertexOut fullscreen_vertex(uint vertexID [[vertex_id]]) {
    const float2 quad[4] = {
        float2(-1.0, -1.0),
        float2(1.0, -1.0),
        float2(-1.0, 1.0),
        float2(1.0, 1.0)
    };
    BackgroundVertexOut out;
    out.position = float4(quad[vertexID], 1.0, 1.0); // Depth = 1.0 (far plane)
    out.uv = float2(0.5, 0.5) + 0.5 * quad[vertexID];
    return out;
}

vertex VertexOut standard_vertex(
    VertexIn in [[stage_in]],
    constant FrameUniforms &frame [[buffer(1)]]
) {
    VertexOut out;
    float4 worldPos = float4(in.position, 1.0);
    out.worldPos = worldPos.xyz;
    out.normal = in.normal;
    out.uv = in.uv;
    out.position = frame.projection * frame.worldToCamera * worldPos;
    return out;
}

float3 applyNormalMap(float3 normalSample, float3x3 tbn) {
    float3 n = normalSample * 2.0 - 1.0;
    return normalize(tbn * n);
}

float3x3 computeTBN(float3 N, float3 p, float2 uv) {
    // get edge vectors of the pixel triangle
    float3 dp1 = dfdx(p);
    float3 dp2 = dfdy(p);
    float2 duv1 = dfdx(uv);
    float2 duv2 = dfdy(uv);

    // solve the linear system
    float3 dp2perp = cross(dp2, N);
    float3 dp1perp = cross(N, dp1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // construct a scale-invariant frame 
    float invmax = rsqrt(max(dot(T,T), dot(B,B)));
    return float3x3(T * invmax, B * invmax, N);
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

fragment float4 background_fragment(
    BackgroundVertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(0)]],
    texturecube<float> envMap [[texture(0)]]
) {
    constexpr sampler linearSampler(filter::linear, address::repeat, mip_filter::linear);
    
    float2 ndc = in.uv * 2.0 - 1.0;
    float aspect = frame.resolutionExposure.x / frame.resolutionExposure.y;
    float fov = frame.cameraPosFov.w;
    float tanHalfFov = tan(fov * 0.5);

    // Camera looks down -Z. Right is +X. Up is +Y.
    float3 rayDirCam = normalize(float3(ndc.x * aspect * tanHalfFov, ndc.y * tanHalfFov, -1.0));
    float3 rayDir = normalize((frame.cameraToWorld * float4(rayDirCam, 0.0)).xyz);
    
    float3 color = envMap.sample(linearSampler, rayDir).rgb * frame.iblParams.x;
    color = toneMap(color, frame);
    return float4(color, 1.0);
}

fragment float4 pbr_fragment(
    VertexOut in [[stage_in]],
    constant FrameUniforms &frame [[buffer(0)]],
    constant MaterialUniforms &material [[buffer(1)]],
    texturecube<float> envMap [[texture(0)]],
    texturecube<float> irradianceMap [[texture(1)]],
    texturecube<float> prefilteredMap [[texture(2)]],
    texture2d<float> brdfLUT [[texture(3)]],
    texture2d<float> albedoMap [[texture(4)]],
    texture2d<float> normalMap [[texture(5)]],
    texture2d<float> roughnessMap [[texture(6)]],
    texture2d<float> metallicMap [[texture(7)]]
) {
    constexpr sampler linearSampler(filter::linear, address::repeat, mip_filter::linear);
    constexpr sampler clampSampler(filter::linear, address::clamp_to_edge, mip_filter::linear);

    float3 N = normalize(in.normal);
    float2 uv = in.uv;
    
    // TBN
    float3x3 TBN = computeTBN(N, in.worldPos, uv);

    float3 albedo = albedoMap.sample(linearSampler, uv).rgb * material.baseTint.rgb;
    float roughness = clamp(roughnessMap.sample(linearSampler, uv).r * material.scalars.x, 0.02, 1.0);
    float metallic = clamp(metallicMap.sample(linearSampler, uv).r * material.scalars.y, 0.0, 1.0);

    float3 shadingN = N;
    if (material.featureToggle.x > 0.5) {
        float3 mapN = normalMap.sample(linearSampler, uv).rgb;
        shadingN = applyNormalMap(mapN, TBN);
    }

    float3 V = normalize(frame.cameraPosFov.xyz - in.worldPos);
    float3 R = reflect(-V, shadingN);

    float3 F0 = mix(float3(0.04), albedo, metallic);

    float3 irradiance = irradianceMap.sample(clampSampler, shadingN).rgb;
    float3 diffuse = irradiance * albedo;

    // Map roughness to valid mip levels (0 to count-1) to avoid out-of-bounds sampling
    float lod = roughness * max(frame.iblParams.z - 1.0, 0.0);
    float3 prefiltered = prefilteredMap.sample(clampSampler, R, level(lod)).rgb;
    float2 brdf = brdfLUT.sample(clampSampler, float2(max(dot(shadingN, V), 0.0), roughness)).rg;
    float3 specular = prefiltered * (F0 * brdf.x + brdf.y);

    float3 color = (diffuse * (1.0 - metallic) + specular) * frame.iblParams.x;

    // Channel inspector overrides
    if (material.scalars.w > 0.5) {
        if (material.scalars.w < 1.5) color = albedo;
        else if (material.scalars.w < 2.5) color = float3(roughness);
        else if (material.scalars.w < 3.5) color = float3(metallic);
        else color = shadingN * 0.5 + 0.5;
    }

    if (material.featureToggle.z > 0.5) {
        color = shadingN * 0.5 + 0.5;
    }

    if (material.featureToggle.w > 0.5) {
        float grid = abs(fract(uv.x * 20.0 - 0.5) - 0.5) / fwidth(uv.x * 20.0);
        float grid2 = abs(fract(uv.y * 10.0 - 0.5) - 0.5) / fwidth(uv.y * 10.0);
        float g = min(grid, grid2);
        g = smoothstep(0.0, 1.0, g);
        color = mix(color, float3(1.0, 0.18, 0.18), clamp(1.0 - g, 0.0, 1.0));
    }

    color = toneMap(color, frame);
    return float4(color, 1.0);
}