""" VAE style encoder for z, that takes x_t, x_{t+1} and t as input. """

import torch
from torch import nn
from torch.nn import functional as F


from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_encoding,
)



class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.latent_dim = latent_dim = config.model.encoder.latent_dim
        self.hidden_dims = hidden_dims = config.model.encoder.hidden_dims
        self.base_dim = base_dim = hidden_dims[0]

        self.img_size = img_size = config.data.image_size

        in_channels = config.data.num_channels * 2 + 1  # x_t, x_{t+1}, t

        self.time_embed = nn.Sequential(
            linear(base_dim, img_size * img_size),
            nn.SiLU(),
            linear(img_size * img_size, img_size * img_size),
        )
        self.data_embed = nn.Conv2d(config.data.num_channels, config.data.num_channels, kernel_size=1)

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    conv_nd(2, in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    normalization(h_dim),
                    nn.SiLU(),
                    nn.Dropout(config.model.dropout)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # encoder should downsample upto a 2x2 image, hence fc is 4 * h_dim
        self.fc_mu = linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = linear(hidden_dims[-1] * 4, latent_dim)
    
    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x_t, x_tp1, scales):
        t_embed = self.time_embed(timestep_encoding(scales, self.base_dim)).view(-1, self.img_size, self.img_size).unsqueeze(1)
        
        x = torch.cat([self.data_embed(x_t), self.data_embed(x_tp1), t_embed], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def sample(self, batch_size):
        """Sample z from prior."""
        z = torch.randn(batch_size, self.latent_dim).to(self.config.device)
        return z, 0, 1
