# A vae that uses a simple encoder and the diffusion model unet as a decoder.
import torch as th
import torch.nn as nn

from .unet import UNetModel
from .encoder import Encoder

class DiffusionVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = UNetModel(config)

    def forward(self, x_t, x_tp1, t):
        # sample z from q(z|x_t, x_tp1, t)
        z, mu, log_var = self.encoder(x_t, x_tp1, t)
        # decode x_t from xtp1, z and t
        return self.decoder(x_tp1, t, z), z, mu, log_var
    
    def sample(self, x, t, z=None):
        if z is None:
            z, _, _ = self.encoder.sample()
        return self.decoder(x, t, z), z
