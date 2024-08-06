import torch
import torch.nn.functional as F
import torch_dct
import numpy as np

# Define the Discrete Cosine Transform (DCT) and its inverse
def DCT(x):
    return torch_dct.dct_2d(x, norm='ortho')

def IDCT(x):
    return torch_dct.idct_2d(x, norm='ortho')

# Function to compute frequency scaling
def get_frequency_scaling(t, img_dim, sigma_blur_max, min_scale=0.001):
    # Compute dissipation time
    sigma_blur = sigma_blur_max * torch.sin(t * torch.pi / 2) ** 2
    dissipation_time = sigma_blur ** 2 / 2

    # Compute frequencies
    freqs = torch.pi * torch.linspace(0, img_dim - 1, img_dim, device=t.device) / img_dim
    labda = freqs[:, None]**2 + freqs[None, :]**2

    # Compute scaling for frequencies
    scaling = torch.exp(-labda * dissipation_time) * (1 - min_scale)
    scaling = scaling + min_scale
    return scaling

# Function to get alpha and sigma
def get_alpha_sigma(t, img_dim, sigma_blur_max):
    freq_scaling = get_frequency_scaling(t, img_dim, sigma_blur_max)
    a, sigma = get_noise_scaling_cosine(t)
    alpha = a * freq_scaling  # Combine dissipation and scaling
    return alpha, sigma

# Function to obtain noise parameters using a typical cosine schedule
def get_noise_scaling_cosine(t, logsnr_min=-10, logsnr_max=10):
    limit_max = np.arctan(np.exp(-0.5 * logsnr_max))
    limit_min = np.arctan(np.exp(-0.5 * logsnr_min)) - limit_max
    logsnr = -2 * torch.log(torch.tan(limit_min * t + limit_max))

    # Transform logsnr to a, sigma
    return torch.sqrt(F.sigmoid(logsnr)), torch.sqrt(F.sigmoid(-logsnr))

# Function to perform diffusion
def diffuse(x, t, img_dim, sigma_blur_max):

    # reshape t to fit data batch and channels
    if len(x.shape) == 4:
        t = t[:, None, None, None]
    elif len(x.shape) == 3:
        t = t[:, None, None]


    x_freq = DCT(x)
    alpha, sigma = get_alpha_sigma(t, img_dim, sigma_blur_max)
    eps = torch.randn_like(x)

    # Perform diffusion
    z_t = IDCT(alpha * x_freq)
    return z_t, eps, sigma

# Function to perform denoising
def denoise(z_t, t, neural_net, img_dim, encoder_noise, sigma_blur_max, T=1000, delta=1e-8):
    straight_t = t
    # reshape t to fit data batch and channels
    if len(z_t.shape) == 4:
        t = t[:, None, None, None]
    elif len(z_t.shape) == 3:
        t = t[:, None, None]

    alpha_s, sigma_s = get_alpha_sigma(t - 1 / T, img_dim, sigma_blur_max)
    alpha_t, sigma_t = get_alpha_sigma(t, img_dim, sigma_blur_max)

    # Compute helpful coefficients
    alpha_ts = alpha_t / alpha_s
    alpha_st = 1 / alpha_ts
    sigma2_ts = (sigma_t ** 2 - alpha_ts ** 2 * sigma_s ** 2)

    # Denoising variance
    sigma2_denoise = 1 / torch.clip(
        1 / torch.clip(sigma_s ** 2, min=delta) +
        1 / torch.clip(sigma_t ** 2 / alpha_ts ** 2 - sigma_s ** 2, min=delta),
        min=delta)

    # The coefficients for u_t and u_eps
    coeff_term1 = alpha_ts * sigma2_denoise / (sigma2_ts + delta)
    coeff_term2 = alpha_st * sigma2_denoise / torch.clip(sigma_s ** 2, min=delta)

    # Get neural net prediction
    t = straight_t
    hat_eps = neural_net(z_t, None, t, encoder_noise)

    # Compute terms
    u_t = DCT(z_t)
    term1 = IDCT(coeff_term1 * u_t)
    term2 = IDCT(coeff_term2 * (u_t - sigma_t * DCT(hat_eps)))
    mu_denoise = term1 + term2

    # Sample from the denoising distribution
    eps = torch.randn_like(mu_denoise)
    return mu_denoise, IDCT(torch.sqrt(sigma2_denoise) * eps)