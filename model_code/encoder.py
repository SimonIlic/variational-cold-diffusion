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
    
    def forward(self, xt, x0, t):
        t_embed = self.time_embed(timestep_encoding(t, self.base_dim)).view(-1, self.img_size, self.img_size).unsqueeze(1)
        
        x = torch.cat([self.data_embed(xt), self.data_embed(x0), t_embed], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def sample(self, batch_size):
        """Sample z from prior."""
        z = torch.randn(batch_size, self.latent_dim).to(self.config.device)
        return z, 0, 1


class ZsEncoder(nn.Module):
    """Encodes zs in several levels of hierarchy."""
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.latent_dim = latent_dim = config.model.encoder.latent_dim
        self.hidden_dims = hidden_dims = config.model.encoder.hidden_dims
        self.base_dim = base_dim = hidden_dims[0]

        self.z_shapes = [()]

        self.img_size = img_size = config.data.image_size

        in_channels = original_in_channels = config.data.num_channels * 2 + 1  # x_t, x_{t+1}, t

        self.time_embed = nn.Sequential(
            linear(base_dim, img_size * img_size),
            nn.SiLU(),
            linear(img_size * img_size, img_size * img_size),
        )
        self.data_embed = nn.Conv2d(config.data.num_channels, config.data.num_channels, kernel_size=1)

        modules = nn.ModuleList([])
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    conv_nd(2, in_channels, h_dim + 2, kernel_size=3, stride=2, padding=1, device=config.device),
                    normalization(h_dim + 2),
                    nn.SiLU(),
                    nn.Dropout(config.model.dropout)
                )
            )
            in_channels = h_dim

        self.encoder = modules

        # first level of local z encoder
        self.first_z_encoder = nn.Sequential(
            conv_nd(2, original_in_channels, 2, kernel_size=1, stride=1, padding=0),
            normalization(2),
            nn.SiLU(),
            nn.Dropout(config.model.dropout)
        )

        # local z encoder
        modules = []
        for module in range(3):
            modules.append(
                nn.Sequential(
                    conv_nd(2, 2, 2, kernel_size=1, stride=1, padding=0),
                    normalization(2),
                    nn.SiLU(),
                )
            )
        modules.append(
            conv_nd(2, 2, 2, kernel_size=1, stride=1, padding=0),
        )
        self.latent_param_encoding = nn.Sequential(*modules)

        # encoder should downsample upto a 2x2 image, hence fc is 4 * h_dim
        self.fc_mu = linear(hidden_dims[-1] * 4 * 4, latent_dim)
        self.fc_var = linear(hidden_dims[-1] * 4 * 4, latent_dim)
    
    def encode(self, x):
        h = x.type(torch.float32)
        latent_params = []

        latent_params.append(self.latent_param_encoding(self.first_z_encoder(h)))

        for layer in self.encoder:
            h = layer(h)
            # peel last two channels for local z encoding
            latent_params.append(self.latent_param_encoding(h[:, -2:, :, :]))
            h = h[:, :-2, :, :]


        h = torch.flatten(h, start_dim=1)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        # current shape is [B, laten_dim]

        # stack together mu, log_var as channels for local z encoding, shape is [B, 2, latent_dim]
        latent_params.append(torch.cat([mu.unsqueeze(1), log_var.unsqueeze(1)], dim=1))

        return latent_params
    
    def reparameterize(self, latent_params):
        zs = []
        for params in latent_params:
            mu, log_var = params[:, 0, ...], params[:, 1, ...]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu
            
            zs.append(z)
        return zs
    
    def forward(self, x_t, x_tp1, scales):
        t_embed = self.time_embed(timestep_encoding(scales, self.base_dim)).view(-1, self.img_size, self.img_size).unsqueeze(1)
        
        x = torch.cat([self.data_embed(x_t), self.data_embed(x_tp1), t_embed], dim=1)
        latent_params = self.encode(x)
        zs = self.reparameterize(latent_params)

        if self.z_shapes is None:
            self.z_shapes = [z.shape[1:] for z in zs]
        return zs, latent_params
    
    def sample(self, batch_size):
        """Sample z from prior."""
        zs = []
        for shape in self.z_shapes:
            z = torch.randn(batch_size, *shape)
            zs.append(z)
        return zs
