import torch
import torch_dct as dct

torch.set_default_dtype(torch.float64)

x = torch.randn(200)
X = dct.dct(x)
y = dct.idct(X)

sum = torch.abs(x - y).sum()
print(sum)
assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance