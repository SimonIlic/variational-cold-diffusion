import torch
import torch_dct

# get random tensor
x = torch.randn(32, 32)
x = x.to(torch.float64)
freq_x = torch_dct.dct_2d(x, norm='ortho')
x_recon = torch_dct.idct_2d(freq_x, norm='ortho')

print(torch.sum(torch.abs(x - x_recon)))  # tensor(1.1921e-07)
