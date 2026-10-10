#include <cuda_runtime.h>
#include <cmath>

__global__ void vecadd_kernel(const float* a, const float* b, float* c, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vecadd(const float* A_h, const float* B_h, float* C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d,*C_d;

  cudaMalloc((void**) &A_d, size);
  cudaMalloc((void**) &B_d, size);
  cudaMalloc((void**) &C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  vecadd_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_d, C_h, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
    int n = 1000;
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    float *c = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    vecadd(a, b, c, n);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            correct = false;
            printf("Mismatch at index %d: got %f, expected %f\n", i, c[i], a[i] + b[i]);
            break;
        }
    }
    if (correct){
      printf("Correct\n");
    }
    else {
      printf("Inorrect\n");
    }

    free(a);
    free(b);
    free(c);
    return 0;
}