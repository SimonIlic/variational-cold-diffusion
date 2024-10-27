import torch
import numpy as np
import logging
from scripts import datasets
from model_code.utils import cosine_schedule
import scripts.hoogeboom as hgb
from scripts.utils import save_gif

def get_sampling_fn_noise_forward(config, initial_sample, intermediate_sample_indices, delta, device, share_noise=False, degradation_operator=None):
    # sampler as described in Bansal et al. 2023 (TaCoS)
    K = config.model.K

    def sampler(model):
        intermediate_samples_out = []
        with torch.no_grad():
            u = initial_sample.to(config.device).float()
            if intermediate_sample_indices != None and K in intermediate_sample_indices:
                intermediate_samples_out.append((u, u))
            steps = np.linspace(1, 0, K)
            for i, t in enumerate(steps[:-1]):
                vec_t = torch.ones(
                    initial_sample.shape[0], device=device, dtype=torch.long) * t
                vec_tm1 = torch.ones(
                    initial_sample.shape[0], device=device, dtype=torch.long) * steps[i+1]
                
                # predict reconstruction
                if i == 0:
                    z = torch.randn(config.model.encoder.latent_dim, device=config.device)
                    # expand z to batch size
                    z = z.expand(u.shape[0], -1)
                    reconstructed = model(u, None, vec_t, z)
                else:
                    reconstructed = model(u, reconstructed, vec_t, None)

                a = cosine_schedule(vec_t)
                # reshape to fit channel dimension
                if len(u.shape) == 4:
                    a = a[:, None, None, None]
                else:
                    a = a[:, None, None]
                noise_estimate = (u - a.sqrt() * reconstructed) / (1 - a).sqrt()
                # update step-by-step reconstruction
                u = u - degradation_operator(reconstructed, vec_t, noise_estimate) + degradation_operator(reconstructed, vec_tm1, noise_estimate)
                u = u.float()  # make sure u is in floats

                # noise the reconstruction a bit for sampling variation
                noise = torch.randn_like(u) * delta
                u = u + noise

                # Save trajectory
                if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                    intermediate_samples_out.append((u, reconstructed))
            
            return u, config.model.K, [u for (u, reconstructed) in intermediate_samples_out], [reconstructed for (u, reconstructed) in intermediate_samples_out]

    return sampler

def get_sampling_fn_inverse_heat(config, initial_sample,
                                 intermediate_sample_indices, delta, device,
                                 share_noise=False, degradation_operator=None, resample_z=False):
    """ Returns our inverse heat process sampling function. 
    Arguments: 
    initial_sample: Pytorch Tensor with the initial draw from the prior p(u_K)
    intermediate_sample_indices: list of indices to save (e.g., [0,1,2,3...] or [0,2,4,...])
    delta: Standard deviation of the sampling noise
    share_noise: Whether to use the same noises for all elements in the batch
    """
    K = config.model.K
    blur_schedule = torch.tensor(config.model.blur_schedule, device=config.device)
            
    if config.model.loss_type == 'risannen':
        def sampler(model):

            if share_noise:
                noises = [torch.randn_like(initial_sample[0], dtype=torch.float)[None]
                        for i in range(K)]
            intermediate_samples_out = []
            with torch.no_grad():
                u = initial_sample.to(config.device).float()
                if intermediate_sample_indices != None and K in intermediate_sample_indices:
                    intermediate_samples_out.append((u, u))
                for i in range(K, 0, -1):
                    vec_fwd_steps = torch.ones(
                        initial_sample.shape[0], device=device, dtype=torch.long) * i
                    scales = blur_schedule[vec_fwd_steps]
                    # Predict less blurry mean
                    diff = model(u, scales)
                    u_mean = u + diff
                    # Sampling step
                    if share_noise:
                        noise = noises[i-1]
                    else:
                        noise = torch.randn_like(u)
                    u = u_mean + noise*delta
                    # Save trajectory
                    if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                        intermediate_samples_out.append((u, diff))

                return u_mean, config.model.K, [u for (u, diff) in intermediate_samples_out], [diff for (u, diff) in intermediate_samples_out]
    
    elif config.model.loss_type == 'hoogeboom':
        def sampler(model):
            intermediate_samples_out = []
            with torch.no_grad():
                u = initial_sample.to(config.device).float()
                u = torch.randn_like(u)
                if intermediate_sample_indices != None and K in intermediate_sample_indices:
                    intermediate_samples_out.append((u, u))
                steps = np.linspace(1, 0, K)
                for i, t in enumerate(steps[:-1]):
                    t = torch.ones(
                        initial_sample.shape[0], device=u.device, dtype=torch.long) * t
                    
                    z = torch.randn(config.eval.batch_size, config.model.encoder.latent_dim, device=u.device)
                    u_mean, noise, x0 = hgb.denoise(u, t, model, config.data.image_size, z, config.model.blur_sigma_max, K)

                    u = u_mean + noise

                    # Save trajectory
                    if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                        intermediate_samples_out.append((u, x0))
                    
                return u_mean, config.model.K, [u for (u, mean) in intermediate_samples_out], [mean for (u, mean) in intermediate_samples_out]

            
    elif config.model.loss_type == 'trajectory_matching':
        def sampler(model):
            if share_noise:
                noises = [torch.randn_like(initial_sample[0], dtype=torch.float)[None]
                        for i in range(K)]
                
            intermediate_samples_out = []
            with torch.no_grad():
                u = initial_sample.to(config.device).float()
                if intermediate_sample_indices != None and K in intermediate_sample_indices:
                    intermediate_samples_out.append((u, u))
                for t in np.linspace(1, 0, K+1):
                    vec_fwd_steps = torch.ones(
                        initial_sample.shape[0], device=device, dtype=torch.long) * t
                    # Predict less blurry mean
                    z = torch.randn(config.eval.batch_size, config.model.encoder.latent_dim, device=config.device)
                    prediction = model(u, None, vec_fwd_steps, z)
                    u_mean = prediction
                    # Sampling step
                    if share_noise:
                        noise = noises[i-1]
                    else:
                        noise = torch.randn_like(u)
                    u = u_mean + noise*delta
                    # Save trajectory
                    intermediate_samples_out.append((u, u_mean))

                return u_mean, config.model.K, [u for (u, mean) in intermediate_samples_out], [u_mean for (u, u_mean) in intermediate_samples_out]
    
    elif config.model.loss_type == "bansal" or config.model.loss_type == "variable_t":
        # sampler as described in Bansal et al. 2023 (TaCoS)
        def sampler(model):
            intermediate_samples_out = []
            with torch.no_grad():
                u = initial_sample.to(config.device).float()
                if intermediate_sample_indices != None and K in intermediate_sample_indices:
                    intermediate_samples_out.append((u, u))
                steps = np.linspace(1, 0, K)
                for i, t in enumerate(steps[:-1]):
                    vec_t = torch.ones(
                        initial_sample.shape[0], device=device, dtype=torch.long) * t

                    # predict reconstruction
                    if i == 0:
                        # share noise
                        z = torch.randn(config.model.encoder.latent_dim, device=config.device)
                        # expand z to batch size
                        z = z.expand(u.shape[0], -1)

                        # do not share noise
                        z = torch.randn(config.eval.batch_size, config.model.encoder.latent_dim, device=config.device)
                        print("Sampling with different noise per trajectory")
                        reconstructed = model(u, None, vec_t, z)
                    else:
                        if resample_z:
                            print("New noise sampled at each step")
                            z = torch.randn(config.eval.batch_size, config.model.encoder.latent_dim, device=config.device)
                        else:
                            z = None
                        reconstructed = model(u, reconstructed, vec_t, z)
                    
                    # update step-by-step reconstruction
                    u = u - degradation_operator(reconstructed, vec_t) + degradation_operator(reconstructed, torch.ones(
                        initial_sample.shape[0], device=device, dtype=torch.long) * steps[i+1])
                    u = u.float()  # make sure u is in floats

                    # noise the reconstruction a bit for sampling variation
                    noise = torch.randn_like(u) * delta
                    u = u + noise

                    # Save trajectory
                    if intermediate_sample_indices != None and i-1 in intermediate_sample_indices:
                        intermediate_samples_out.append((u, reconstructed))

                # final step
                reconstructed = model(u, reconstructed, vec_t * 0, None)
                u = u - degradation_operator(reconstructed, vec_t * 0) + reconstructed
                intermediate_samples_out.append((u, reconstructed))
                
                return u, config.model.K, [u for (u, reconstructed) in intermediate_samples_out], [reconstructed for (u, reconstructed) in intermediate_samples_out]

    return sampler


def get_sampling_fn_inverse_heat_interpolate(config, initial_sample,
                                             delta, device, num_points):
    """Returns an interpolation between two images, where the interpolation is
    done with the latent noise and the initial sample. Arguments: 
    initial_sample: Two initial states to interpolate over. 
                                    Shape: (2, num_channels, height, width)
    delta: Sampling noise standard deviation
    num_points: Number of point to use in the interpolation 
    """
    assert initial_sample.shape[0] == 2
    shape = initial_sample.shape

    # Linear interpolation between the two input states
    init_input = torch.linspace(1, 0, num_points, device=device)[:, None, None, None] * initial_sample[0][None] + \
        (1-torch.linspace(1, 0, num_points, device=device)
         )[:, None, None, None] * initial_sample[1][None]
    init_input = init_input.to(config.device).float()
    logging.info("init input shape: {}".format(init_input.shape))

    # Get all the noise steps
    noise1 = [torch.randn_like(init_input[0]).to(device)[None]
              for i in range(0, config.model.K)]
    noise1 = torch.cat(noise1, 0)
    noise2 = [torch.randn_like(init_input[0]).to(device)[None]
              for i in range(0, config.model.K)]
    noise2 = torch.cat(noise2, 0)

    # Spherical interpolation between the noise endpoints.
    noise_weightings = torch.linspace(
        0, np.pi/2, num_points, device=device)[None, :, None, None, None]
    noise1 = noise1[:, None, :, :, :]
    noise2 = noise2[:, None, :, :, :]
    noises = torch.cos(noise_weightings) * noise1 + \
        torch.sin(noise_weightings) * noise2

    K = config.model.K

    def sampler(model):
        with torch.no_grad():
            x = init_input.to(config.device).float()
            for i in range(K, 0, -1):
                vec_fwd_steps = torch.ones(
                    num_points, device=device, dtype=torch.long) * i
                x_mean = model(x, vec_fwd_steps) + x
                noise = noises[i-1]
                x = x_mean + noise*delta
            x_sweep = x_mean
            return x_sweep

    return sampler, init_input


def get_initial_sample(config, forward_heat_module, delta, batch_size=None, workdir="/"):
    """Take a draw from the prior p(u_K)"""
    trainloader, testloader = datasets.get_dataset(config,
                                          uniform_dequantization=config.data.uniform_dequantization,
                                          train_batch_size=batch_size)

    initial_sample = next(iter(testloader))[0].to(config.device)
    original_images = initial_sample.clone()
    # save forward as gif
    trajectory = forward_trajectory(initial_sample, config, forward_heat_module)
    #save_gif(workdir, trajectory, name="forward.gif")

    initial_sample = forward_heat_module(initial_sample,
                                         torch.ones(initial_sample.shape[0], dtype=torch.long).to(config.device))
    return initial_sample, original_images

def get_zero_initial_sample(config):
    """Take a draw from the prior p(u_K), i.e. zero vectors"""
    initial_sample = torch.zeros(config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size).to(config.device)
    return initial_sample

def get_noise_initial_sample(config):
    initial_sample = torch.randn(config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size).to(config.device)
    return initial_sample

def forward_trajectory(images, config, forward_heat_module):
    trajectory = [images]
    for t in np.linspace(0, 1, config.model.K + 1):
        vec_t = torch.ones(images.shape[0], device=config.device, dtype=torch.long) * t
        blurred = forward_heat_module(images, vec_t)
        trajectory.append(blurred)

    return trajectory
