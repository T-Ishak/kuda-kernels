#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>


__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // blockIdx.x, blockDim.x, threadIdx.x
  // blockIdx.y, blockDim.y, threadIdx.y

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

// m * k times k * n = n * m


int ceil_division(int x, int y) {
  // ceil(x/y)
  return (x + y - 1)/(y);
}

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
  const float* A_d = A.data_ptr<float>();
  const float* B_d = B.data_ptr<float>();

  int M = A.size(0);
  int N = B.size(1);
  int K = A.size(1);

  torch::Tensor output = torch::empty({M, N}, A.options());
  float* C_d = output.data_ptr<float>();

  dim3 blockDim(16, 16);
  dim3 gridDim(ceil_division(N, blockDim.x), ceil_division(M, blockDim.y));

  matmul_kernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);

  return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Matrix multiply (CUDA)");
}