#pragma once
#include <cuda_runtime.h>

__global__ void initParticles(int N, float4 *pos, float4 *vel, float4 *acc);
__global__ void calculateForces(int N, float4 *devA, float4 *devX);
__global__ void verletIntegration(int N, float4 *acc, float4 *pos, float4 *vel, float dt);
__global__ void verletIntegration2(int N, float4 *acc, float4 *vel, float dt);
__global__ void setCentripetalVelocities(int N, float4 *pos, float4 *vel, float4 *acc, float3 rCOM);
__global__ void removeDrift(int N, float4 *vel, float3 drift);
