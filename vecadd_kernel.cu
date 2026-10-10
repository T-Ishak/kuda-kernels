#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vecadd_kernel(const float* a, const float* b, float* c, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

torch::Tensor vecadd(torch::Tensor a, torch::Tensor b) {
    const float* A_d = a.data_ptr<float>();
    const float* B_d = b.data_ptr<float>();

    torch::Tensor output = torch::empty_like(a);
    float* C_d = output.data_ptr<float>();

    int n = a.numel();

    vecadd_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vecadd", &vecadd, "Vector add (CUDA)");
}