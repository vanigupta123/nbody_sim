// [ADDED] benchmark.cu — measures calculateForces kernel throughput
// Run: ./benchmark <N>
// Reports: kernel time (ms), GFLOP/s
// Formula: each particle pair does ~20 flops; N^2 pairs total -> 20*N^2 flops per step

#include <stdio.h>
#include <cuda_runtime.h>
#include "nbody.cuh"

int main(int argc, char **argv) {
  int N = (argc > 1) ? atoi(argv[1]) : 4096;
  int warmup = 5;
  int iters  = 20;

  float4 *d_pos, *d_acc;
  cudaMalloc(&d_pos, N * sizeof(float4));
  cudaMalloc(&d_acc, N * sizeof(float4));

  // [ADDED] initialize with dummy data so kernel has something to work on
  // TODO: could call initParticles here instead if you want realistic inputs
  cudaMemset(d_pos, 0, N * sizeof(float4));
  cudaMemset(d_acc, 0, N * sizeof(float4));

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  int shareMem  = sizeof(float4) * blockSize;

  // warmup
  for (int i = 0; i < warmup; i++)
    calculateForces<<<numBlocks, blockSize, shareMem>>>(N, d_acc, d_pos);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++)
    calculateForces<<<numBlocks, blockSize, shareMem>>>(N, d_acc, d_pos);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  float ms_per_step = ms / iters;

  // ~20 flops per pair (subtract, multiply, rsqrt counts as ~4, accumulate)
  double flops = 20.0 * (double)N * (double)N;
  double gflops = (flops / (ms_per_step / 1000.0)) / 1e9;

  printf("N=%d  blockSize=%d  time=%.3f ms/step  perf=%.1f GFLOP/s\n",
         N, blockSize, ms_per_step, gflops);

  cudaFree(d_pos);
  cudaFree(d_acc);
  return 0;
}
