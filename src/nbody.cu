#include "nbody.cuh"
#include <curand_kernel.h>
#include <math.h>
#include <stdio.h>

__global__ void initParticles(int N, float4 *pos, float4 *vel, float4 *acc) {
  // this is a kernel and will be called with params<<<grid_dimensions, block_dimensions>>>
  int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= N) return;

  curandState state;
  curand_init(1234ULL, idx, 0, &state);

  float4 p;
  float spread = (N < 10) ? 5.0 : 50.0; // particles are closer together if smaller N
  float baseMass = (N < 10) ? 100.0f : 10.0f;
  p.x = spread * (curand_uniform(&state) - 0.5f);
  p.y = spread * (curand_uniform(&state) - 0.5f);
  p.z = spread * (curand_uniform(&state) - 0.5f);
  p.w = baseMass * (0.1f + 0.4f * curand_uniform(&state));
  pos[idx] = p;
  vel[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
  acc[idx] = {0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ float3 getAcceleration(float4 &pos_i, float4 &pos_j, float3 &acc) {
  const float EPS2 = 1.0f/150.0f;
  // F = Gm1m2/|r|^2 * r/|r|
  // a1 = Gm2r/|r|^3
  // with softening -> a1 = Gm2r/(|r|^2 + eps^2)^(3/2)
  float3 r;
  float magSquared, val;
  // all threads in a block have access to the same shared memory
  // one shared memory per block
  // must synchronize threads to prevent race condition
    // thread can only continue execution once all threads in block have synchronized
    // must have synchronization at same point for each thread or else the code will deadlock
  r.x = pos_j.x - pos_i.x;
  r.y = pos_j.y - pos_i.y;
  r.z = pos_j.z - pos_i.z;
  magSquared = (r.x*r.x) + (r.y*r.y) + (r.z*r.z) + EPS2;
  val = pos_j.w * rsqrt(magSquared * magSquared * magSquared);
  acc.x += r.x * val;
  acc.y += r.y * val;
  acc.z += r.z * val;
  return acc;
}

__device__ float3 tileCalculation(float4 pos_i, float3 accel, int valid) {
  int i;
  extern __shared__ float4 shPosition[];
  for (i = 0; i < valid; i++) {
    accel = getAcceleration(pos_i, shPosition[i], accel);
  }
  return accel;
}

__global__ void calculateForces(int N, float4 *devA, float4 *devX) {
  extern __shared__ float4 shPosition[]; // var declaration in shared memory
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gtid >= N) return;
  // float G = 6.6743e-11; // can't use for sim purposes. won't show any movement
  float G = 1.0f;
  float3 acc = {0.0f, 0.0f, 0.0f};
  float4 pos_i = devX[gtid];
  // tiles are pxp, where p = blockDim.x
  for (int i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    if (idx < N) shPosition[threadIdx.x] = devX[idx];
    // guards load for last partial tile. remaining threads otherwise have invalid devX[idx] vals
    __syncthreads();
    int valid = min(blockDim.x, N-tile*blockDim.x);
    acc = tileCalculation(pos_i, acc, valid);
    __syncthreads();
  }
  float4 acc4 = {G*acc.x, G*acc.y, G*acc.z, 0.0f}; // for global acceleration var
  devA[gtid] = acc4;
}

__global__ void verletIntegration(int N, float4 *acc, float4 *pos, float4 *vel, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  float4 a = acc[idx];
  float4 p = pos[idx];
  float4 v = vel[idx];

  // position = x_0 + v dt + (1/2) a dt^2
  p.x += (v.x * dt) + (0.5f * a.x * dt * dt);
  p.y += (v.y * dt) + (0.5f * a.y * dt * dt);
  p.z += (v.z * dt) + (0.5f * a.z * dt * dt);

  // half step of velocity = v_0 + (1/2) a * dt
  v.x += 0.5f * a.x * dt;
  v.y += 0.5f * a.y * dt;
  v.z += 0.5f * a.z * dt;

  pos[idx] = p;
  vel[idx] = v;
}

__global__ void verletIntegration2(int N, float4 *acc, float4 *vel, float dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  float4 a = acc[idx];
  float4 v = vel[idx];
  // second half step of velocity after updated acc
  v.x += 0.5f * a.x * dt;
  v.y += 0.5f * a.y * dt;
  v.z += 0.5f * a.z * dt;

  vel[idx] = v;
}

__global__ void setCentripetalVelocities(int N, float4 *pos, float4 *vel, float4 *acc, float3 rCOM) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 r = {pos[idx].x - rCOM.x, pos[idx].y - rCOM.y, pos[idx].z - rCOM.z};
    float rmag = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z);
    if (rmag < 1e-6f) return;

    float3 rhat = {r.x/rmag, r.y/rmag, r.z/rmag};
    float3 a = {acc[idx].x, acc[idx].y, acc[idx].z};
    float ar = fabsf(a.x*rhat.x + a.y*rhat.y + a.z*rhat.z);
    float vc = sqrtf(rmag * ar);  // circular velocity magnitude

    // find tangential vector
    float3 k = {0, 0, 1};  // rotation axis
    float3 t = {k.y*rhat.z - k.z*rhat.y,
                k.z*rhat.x - k.x*rhat.z,
                k.x*rhat.y - k.y*rhat.x};
    float tmag = sqrtf(t.x*t.x + t.y*t.y + t.z*t.z);
    if (tmag < 1e-8f) { t = {1,0,0}; tmag = 1; }
    t.x /= tmag; t.y /= tmag; t.z /= tmag;

    vel[idx].x = vc * t.x;
    vel[idx].y = vc * t.y;
    vel[idx].z = vc * t.z;
}

__global__ void removeDrift(int N, float4 *vel, float3 drift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    vel[idx].x -= drift.x;
    vel[idx].y -= drift.y;
    vel[idx].z -= drift.z;
}
