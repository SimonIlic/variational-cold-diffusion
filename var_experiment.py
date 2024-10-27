import os
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from scripts import losses
from scripts import sampling
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
import wandb
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import matplotlib.pyplot as plt
from train import create_degrader
from scripts import sampling
from torchvision.utils import make_grid

workdir = "runs/3691322"
n_images = 3

# get config
from configs.config_3691322 import get_config
config = get_config()

# load model
# Initialize model
model = mutils.create_model(config)
optimizer = losses.get_optimizer(config, model.parameters())
ema = ExponentialMovingAverage(
    model.parameters(), decay=config.model.ema_rate)
state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
model_evaluation_fn = mutils.get_model_fn(model, train=False, sample=True if config.model.type == 'vae' else False)

# Create checkpoints directory
checkpoint_dir = os.path.join(workdir, "checkpoints")
# Intermediate checkpoints to resume training
checkpoint_meta_dir = os.path.join(
    workdir, "checkpoints-meta", "checkpoint.pth")
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(checkpoint_meta_dir)).mkdir(
    parents=True, exist_ok=True)
# Resume training when intermediate checkpoints are detected
state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
initial_step = int(state['step'])

# Get the forward process definition
scales = config.model.blur_schedule
degrader = create_degrader(config)

delta = config.model.delta
scales = torch.tensor(config.model.blur_schedule, device=config.device)

initial_sample, original_image = sampling.get_initial_sample(config, degrader, delta, config.eval.batch_size)

same_initial = original_image[6].unsqueeze(0).repeat(n_images, 1, 1, 1)
z = torch.randn(same_initial.shape[0], config.model.encoder.latent_dim, device=config.device)
# use same z for all
z = z[0].unsqueeze(0).repeat(n_images, 1)
print("Resampled z")
for split_t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:

    t = torch.ones(same_initial.shape[0], device=config.device)
    degraded = degrader(same_initial, t)
    result = model_evaluation_fn(degraded, None, t, z)

    config.model.K = 32
    with torch.no_grad():
        # sampling
        # while any t is not zero
        x = degraded.clone()
        x0 = result.clone()
        while torch.any(t >= 0):
            z = torch.randn(same_initial.shape[0], config.model.encoder.latent_dim, device=config.device)
            # split at t=0.5
            if torch.any(t > split_t):
                z = z[0].unsqueeze(0).repeat(n_images, 1)
                split_point = x.clone()
            x0 = model_evaluation_fn(x, None, t, z)
            x[t>0] = x[t>0] - degrader(x0[t>0], t[t>0]) + degrader(x0[t>0], t[t>0] - 1/config.model.K)
            t = t - 1/config.model.K
        sampling_image = x
        print(sampling_image.var(dim=0).mean())
        # caluclate average variance per pixel (averaged over channels)
        # calculate variance per pixel over the batch
        variance = torch.var(sampling_image, dim=0)

    og_grid = make_grid(same_initial, nrow=3)
    degraded_grid = make_grid(degraded, nrow=3)
    result_grid = make_grid(split_point, nrow=3)
    sampling_grid = make_grid(sampling_image, nrow=3)

    # Plot all four images
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    axs[0].imshow(og_grid.permute(1, 2, 0))
    axs[0].set_title('Test Set Image')
    axs[1].imshow(degraded_grid.permute(1, 2, 0))
    axs[1].set_title(r'Degraded $x_T$')
    axs[2].imshow(result_grid.permute(1, 2, 0))
    axs[2].set_title(f'Split point t={split_t}')
    axs[3].imshow(sampling_grid.permute(1, 2, 0))
    axs[3].set_title('Sampling Results')
    # Hide the axis for all subplots
    for ax in axs:
        ax.axis('off')
    #plt.title(f'K={config.model.K}')
    plt.show()
