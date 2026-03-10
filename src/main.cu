#include <stdio.h>
#include <ftw.h>
#include <fstream>
#include <sys/stat.h>
#include "nbody.cuh"

int unlink_cb(const char *fpath, const struct stat *sb, int typeflag, struct FTW *ftwbuf) {
  int rv = remove(fpath);
  if (rv) perror(fpath);
  return rv;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("bad input\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int numSteps = atoi(argv[2]);
  float dt = atof(argv[3]);
  bool centripetal = atoi(argv[4]);
  float4 *d_pos, *d_vel, *d_acc;
  cudaMalloc(&d_pos, N*sizeof(float4));
  cudaMalloc(&d_vel, N*sizeof(float4));
  cudaMalloc(&d_acc, N*sizeof(float4));
  // one thread per particle
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1)/blockSize; // same as ceil(N/blockSize)
  initParticles<<<numBlocks, blockSize>>>(N, d_pos, d_vel, d_acc);
  int shareMem = sizeof(float4) * blockSize;
  const char *path = "output/";
  nftw(path, unlink_cb, 64, FTW_DEPTH|FTW_PHYS);
  mkdir(path, S_IRWXU);
  for (int step = 0; step < numSteps; step++) {
    verletIntegration<<<numBlocks, blockSize>>>(N, d_acc, d_pos, d_vel, dt);
    calculateForces<<<numBlocks, blockSize, shareMem>>>(N, d_acc, d_pos);
    verletIntegration2<<<numBlocks, blockSize>>>(N, d_acc, d_vel, dt);
    if (step == 0 && centripetal) {
      printf("modifying motion for tangential velocity etc...\n");
      float3 rCOM = {0,0,0};
      float totalMass = 0.0f;

      float4 *h_vel = new float4[N];
      cudaMemcpy(h_vel, d_vel, N*sizeof(float4), cudaMemcpyDeviceToHost);
      float4 *h_pos = new float4[N];
      cudaMemcpy(h_pos, d_pos, N*sizeof(float4), cudaMemcpyDeviceToHost);

      for (int i = 0; i < N; ++i) {
        rCOM.x += h_pos[i].x * h_pos[i].w;
        rCOM.y += h_pos[i].y * h_pos[i].w;
        rCOM.z += h_pos[i].z * h_pos[i].w;
        totalMass += h_pos[i].w;
      }
      rCOM.x /= totalMass;
      rCOM.y /= totalMass;
      rCOM.z /= totalMass;
      setCentripetalVelocities<<<numBlocks, blockSize>>>(N, d_pos, d_vel, d_acc, rCOM);
      // remove drift velocities
      float3 drift = {0, 0, 0};

      for (int i = 0; i < N; i++) {
          drift.x += h_vel[i].x * h_pos[i].w;
          drift.y += h_vel[i].y * h_pos[i].w;
          drift.z += h_vel[i].z * h_pos[i].w;
      }
      drift.x /= totalMass;
      drift.y /= totalMass;
      drift.z /= totalMass;

      delete[] h_vel;
      delete[] h_pos;
      removeDrift<<<numBlocks, blockSize>>>(N, d_vel, drift);
    }
    if (step % 10 == 0) {
      // record position with a particle id
      float4 *h_pos = new float4[N]; // heap alloc
      cudaMemcpy(h_pos, d_pos, N*sizeof(float4), cudaMemcpyDeviceToHost);
      std::ofstream file("output/output_" + std::to_string(step) + ".csv");
      file << "id,x,y,z,m\n";
      for (int i = 0; i < N; i++) {
        file << i << "," << h_pos[i].x << "," << h_pos[i].y << "," << h_pos[i].z << "," << h_pos[i].w << "\n";
      }
      file.close();
      delete[] h_pos;
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // sample of initial velocities and positions
  float4 s_pos[5], s_vel[5], s_acc[5]; // for 5 particles
  cudaMemcpy(s_pos, d_pos, 5*sizeof(float4), cudaMemcpyDeviceToHost);
  cudaMemcpy(s_vel, d_vel, 5*sizeof(float4), cudaMemcpyDeviceToHost);
  cudaMemcpy(s_acc, d_acc, 5*sizeof(float4), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++) {
    printf("particle with mass %.4f at position (%.2f, %.2f, %.2f) with velocity (%.3e, %.3e, %.3e) and net acc of (%.3e, %.3e, %.3e)\n",
    s_pos[i].w, s_pos[i].x, s_pos[i].y, s_pos[i].z,
    s_vel[i].x, s_vel[i].y, s_vel[i].z,
    s_acc[i].x, s_acc[i].y, s_acc[i].z
    );
  }

  return 0;
}
