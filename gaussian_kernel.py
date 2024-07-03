import torch
import torchgeometry as tgm

def gaussian_kernel(batch_size: int, channels: int, kernel_size: int, sigmas: torch.Tensor):
    """ Returns 2D gaussian kernel of batch size with size x size dimensions and standard deviation sigma. Sigmas is a vector with a different sigma for every element in the batch. Final tensor shape is [B, C, size, size]"""
    kernels = []
    for sigma in sigmas:
        kernel = tgm.image.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
        # normalise the kernel
        kernel = kernel / kernel.max()
        kernel = 1 - kernel
        kernels.append(kernel)
    kernels = torch.stack(kernels)
    kernels = kernels.unsqueeze(1).repeat(1, channels, 1, 1)

    return kernels

kernels = gaussian_kernel(2, 3, 51, torch.tensor([1.5, 2.5]))


# plot the gaussian kernels
import matplotlib.pyplot as plt
import numpy as np

kernels = gaussian_kernel(8, 3, 101, 10 * torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 1000]))
kernels = kernels.squeeze().numpy()

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axs.flat[:-1]):
    ax.imshow(kernels[i, 0] - kernels[i+1,0], cmap='hot')
    ax.axis('off')
    ax.set_title(f'Sigma: {(1.5 + i) * 10}')
plt.show()
