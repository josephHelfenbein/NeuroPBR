#include "prefilter.cuh"

extern "C" __global__
void equirectangularToCubemap(cudaTextureObject_t hdr_tex,
							    cudaSurfaceObject_t cubemap_surface,
								int faceSize) {
	int face = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (face >= 6 || x >= faceSize || y >= faceSize) return;

	float u = (float(x) + 0.5f) / float(faceSize);
	float v = (float(y) + 0.5f) / float(faceSize);
	float2 uv = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);

	float3 dir = faceUvToDir(face, uv.x, uv.y);
	float texU, texV;
	dirToLatLong(dir, &texU, &texV);

	float4 color = tex2D<float4>(hdr_tex, texU, texV);
	surf2DLayeredwrite(color, cubemap_surface, x * (int) sizeof(float4), y, face);
}

extern "C" __global__
void prefilterSpecularCubemap(cudaTextureObject_t envCubemap,
								cudaSurfaceObject_t outSurface,
								int faceSize,
								float roughness,
								unsigned int sampleCount) {
	int face = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (face >= 6 || x >= faceSize || y >= faceSize) return;

	float u = (float(x) + 0.5f) / float(faceSize);
	float v = (float(y) + 0.5f) / float(faceSize);
	float2 uv = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);
	float3 N = faceUvToDir(face, uv.x, uv.y);
	float3 R = N;
	float3 V = R;

	float3 prefiltered = make_float3(0.0f, 0.0f, 0.0f);
	float totalWeight = 0.0f;

	for (unsigned int i = 0u; i < sampleCount; ++i) {
		float2 Xi = hammersley(i, sampleCount);
		float3 H = importanceSampleGGX(Xi, N, roughness);
		float3 L = makeNormalized(make_float3(2.0f * dot3(V, H) * H.x - V.x,
											  2.0f * dot3(V, H) * H.y - V.y,
											  2.0f * dot3(V, H) * H.z - V.z));
		float NdotL = fmaxf(dot3(N, L), 0.0f);
		if (NdotL > 0.0f) {
			float4 sample = texCubemap<float4>(envCubemap, L.x, L.y, L.z);
			prefiltered.x += sample.x * NdotL;
			prefiltered.y += sample.y * NdotL;
			prefiltered.z += sample.z * NdotL;
			totalWeight += NdotL;
		}
	}

	if (totalWeight > 0.0f) {
		float inv = 1.0f / totalWeight;
		prefiltered.x *= inv;
		prefiltered.y *= inv;
		prefiltered.z *= inv;
	}

	surf2DLayeredwrite(make_float4(prefiltered.x, prefiltered.y, prefiltered.z, 1.0f),
					   outSurface, x * (int) sizeof(float4), y, face);
}

extern "C" __global__
void convolveDiffuseIrradiance(cudaTextureObject_t envCubemap,
								 cudaSurfaceObject_t outSurface,
								 int faceSize,
								 unsigned int sampleCount) {
	int face = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (face >= 6 || x >= faceSize || y >= faceSize) return;

	float u = (float(x) + 0.5f) / float(faceSize);
	float v = (float(y) + 0.5f) / float(faceSize);
	float2 uv = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);
	float3 N = faceUvToDir(face, uv.x, uv.y);

	float3 irradiance = make_float3(0.0f, 0.0f, 0.0f);

	for (unsigned int i = 0u; i < sampleCount; ++i) {
		float2 Xi = hammersley(i, sampleCount);
		float3 L = cosineSampleHemisphere(Xi, N);
		float NdotL = fmaxf(dot3(N, L), 0.0f);
		if (NdotL > 0.0f) {
			float4 sample = texCubemap<float4>(envCubemap, L.x, L.y, L.z);
			irradiance.x += sample.x;
			irradiance.y += sample.y;
			irradiance.z += sample.z;
		}
	}

	float scale = M_PI / float(sampleCount);
	irradiance.x *= scale;
	irradiance.y *= scale;
	irradiance.z *= scale;

	surf2DLayeredwrite(make_float4(irradiance.x, irradiance.y, irradiance.z, 1.0f),
					   outSurface, x * (int) sizeof(float4), y, face);
}

void launchEquirectangularToCubemap(dim3 gridDim, dim3 blockDim,
							 cudaTextureObject_t hdrTex, cudaSurfaceObject_t cubemapSurface,
							 int faceSize) {
	equirectangularToCubemap<<<gridDim, blockDim>>>(hdrTex, cubemapSurface, faceSize);
}

void launchPrefilterSpecularCubemap(dim3 gridDim, dim3 blockDim,
							   cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface,
							   int faceSize, float roughness, unsigned int sampleCount) {
	prefilterSpecularCubemap<<<gridDim, blockDim>>>(envCubemap, outSurface, faceSize, roughness, sampleCount);
}

void launchConvolveDiffuseIrradiance(dim3 gridDim, dim3 blockDim,
								 cudaTextureObject_t envCubemap, cudaSurfaceObject_t outSurface,
								 int faceSize, unsigned int sampleCount) {
	convolveDiffuseIrradiance<<<gridDim, blockDim>>>(envCubemap, outSurface, faceSize, sampleCount);
}

