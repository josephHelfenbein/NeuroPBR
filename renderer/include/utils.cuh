#pragma once
#include <cuda_runtime.h>
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

__device__ inline float areaElement(float x, float y) {
	return atan2f(x * y, sqrtf(x * x + y * y + 1.0f));
}

__device__ inline float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ inline float fractf(float v) {
    return v - floorf(v);
}

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ inline float smoothstepf(float edge0, float edge1, float x) {
    float t = clamp01((x - edge0) / fmaxf(edge1 - edge0, 1e-5f));
    return t * t * (3.0f - 2.0f * t);
}

__device__ inline float dot2(const float2& a, const float2& b) {
    return a.x * b.x + a.y * b.y;
}

__device__ inline float texelSolidAngle(int x, int y, int size) {
	float u0 = (2.0f * float(x) / float(size)) - 1.0f;
	float u1 = (2.0f * float(x + 1) / float(size)) - 1.0f;
	float v0 = (2.0f * float(y) / float(size)) - 1.0f;
	float v1 = (2.0f * float(y + 1) / float(size)) - 1.0f;

	float solidAngle = areaElement(u0, v0) - areaElement(u0, v1)
					 - areaElement(u1, v0) + areaElement(u1, v1);
	return solidAngle;
}

__device__ inline float warpReduceSum(float value) {
	for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
		value += __shfl_down_sync(0xffffffff, value, offset);
	}
	return value;
}

__device__ inline void blockReduce(float input[4]) {
	int linearThread = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
	int lane = linearThread & 31;
	int warp = linearThread >> 5;
	int totalThreads = blockDim.x * blockDim.y * blockDim.z;
	int numWarps = (totalThreads + warpSize - 1) / warpSize;

	float reduced[4];
	for (int i = 0; i < 4; ++i) {
		reduced[i] = warpReduceSum(input[i]);
	}

	__shared__ float shared[4][32];
	if (lane == 0) {
		shared[0][warp] = reduced[0];
		shared[1][warp] = reduced[1];
		shared[2][warp] = reduced[2];
		shared[3][warp] = reduced[3];
	}
	__syncthreads();

	if (warp == 0) {
		float finalValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		for (int i = lane; i < numWarps; i += warpSize) {
			finalValues[0] += shared[0][i];
			finalValues[1] += shared[1][i];
			finalValues[2] += shared[2][i];
			finalValues[3] += shared[3][i];
		}
		for (int i = 0; i < 4; ++i) {
			finalValues[i] = warpReduceSum(finalValues[i]);
		}
		if (lane == 0) {
			input[0] = finalValues[0];
			input[1] = finalValues[1];
			input[2] = finalValues[2];
			input[3] = finalValues[3];
		}
	}
	__syncthreads();
}

__device__ inline float3 faceUvToDir(int face, float u, float v) {
	float3 dir;
	switch (face) {
		case 0: dir = make_float3(1.0f, v, -u); break;
		case 1: dir = make_float3(-1.0f, v, u); break;
		case 2: dir = make_float3(u, 1.0f, -v); break;
		case 3: dir = make_float3(u, -1.0f, v); break;
		case 4: dir = make_float3(u, v, 1.0f); break;
		default: dir = make_float3(-u, v, -1.0f); break;
	}
	return makeNormalized(dir);
}

__device__ inline float radicalInverseVdC(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

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

__device__ inline void dirToLatLong(const float3 dir, float *u, float *v) {
	dirToLatLong(dir.x, dir.y, dir.z, u, v);
}

__device__ inline float2 hammersley(unsigned int i, unsigned int N) {
	return make_float2(float(i) / float(N), radicalInverseVdC(i));
}

__device__ inline void buildTangentBasis(const float3 N, float3 *T, float3 *B) {
	float3 up = fabsf(N.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
	*T = makeNormalized(cross3(up, N));
	*B = cross3(N, *T);
}

__device__ inline float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
	float a = roughness * roughness;
	float phi = 2.0f * M_PI * Xi.x;
	float cosTheta = sqrtf((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
	float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

	float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);
	float3 T, B;
	buildTangentBasis(N, &T, &B);
	float3 sample;
	sample.x = T.x * H.x + B.x * H.y + N.x * H.z;
	sample.y = T.y * H.x + B.y * H.y + N.y * H.z;
	sample.z = T.z * H.x + B.z * H.y + N.z * H.z;
	return makeNormalized(sample);
}

__device__ inline float3 cosineSampleHemisphere(float2 Xi, float3 N) {
	float r = sqrtf(Xi.x);
	float phi = 2.0f * M_PI * Xi.y;
	float x = r * cosf(phi);
	float y = r * sinf(phi);
	float z = sqrtf(fmaxf(0.0f, 1.0f - Xi.x));
	float3 T, B;
	buildTangentBasis(N, &T, &B);
	float3 sample;
	sample.x = T.x * x + B.x * y + N.x * z;
	sample.y = T.y * x + B.y * y + N.y * z;
	sample.z = T.z * x + B.z * y + N.z * z;
	return makeNormalized(sample);
}