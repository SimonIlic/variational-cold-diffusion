# A vae that uses a simple encoder and the diffusion model unet as a decoder.
import torch as th
import torch.nn as nn

from .unet import UNetModel
from .unet import VAEncoder as Encoder

class DiffusionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = UNetModel(config)

    def forward(self, xt, x0, t):
        # sample z from q(z|x_t, x_tp1, t)
        z, latent_params = self.encoder(xt, x0, t)
        # decode x_t from xtp1, z and t
        return self.decoder(xt, t, z), z, latent_params
    
    def sample(self, xt, x0, t, z=None):
        if z is None:
            z, (mu, log_var) = self.encoder(xt, x0, t)
        return self.decoder(xt, t, z)
