import torch
from torch.utils.cpp_extension import load

kernels = load(name="kernels", sources=["vecadd.cu"])

a = torch.randn(1000, device="cuda")
b = torch.randn(1000, device="cuda")
c = kernels.vecadd(a, b)

print(torch.allclose(c, a + b))