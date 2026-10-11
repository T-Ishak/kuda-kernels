#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[256];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;
    
    if (i < n){
      shared[tid] = input[i];
    }
    else {
      shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int half = blockDim.x / 2; half > 0; half /= 2) {
      if (tid < half) {
          shared[threadIdx.x] += shared[threadIdx.x + half];
      }
      __syncthreads();
  }

    if (tid == 0) {
      atomicAdd(output, shared[0]);
    }
}

void reduce(const float* input_h, float* output_h, int n) {
    int size = n * sizeof(float);
    float *input_d, *output_d;

    cudaMalloc((void**) &input_d, size);
    cudaMalloc((void**) &output_d, sizeof(float));
    cudaMemset(output_d, 0, sizeof(float));

    cudaMemcpy(input_d, input_h, size, cudaMemcpyHostToDevice);

    reduce_kernel<<<ceil(n/256.0), 256>>>(input_d, output_d, n);

    cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(output_d);
}

int main() {
    int n = 1000000;
    float* input = (float*)malloc(n * sizeof(float));
    float output = 0.0f;

    for (int i = 0; i < n; i++) {
        input[i] = 1.0f;
    }

    reduce(input, &output, n);

    printf("Sum: %f (expected %f)\n", output, (float)n);

    free(input);
    return 0;
}