#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
  // M is number of rows in A,
  // K is number of columns in A,  number of rows in B
  // N is number of columns in B

  // C is M * N
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

}

void matmul(const float* A_h, const float* B_h, float* C_h, int M, int K, int N) {

}