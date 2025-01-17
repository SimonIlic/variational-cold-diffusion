"""All functions and modules related to model definition."""
import torch
import torch.nn as nn
import logging
import numpy as np
from model_code.unet import UNetModel
from model_code.diffusion_vae import DiffusionVAE
import torch_dct
import torchgeometry as tgm


def cosine_schedule(t, s=0.0):
    return torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2

class HardVignette(nn.Module):
    def __init__(self, config):
        super(HardVignette, self).__init__()
        self.config = config
        self.image_size = config.data.image_size
        self.max_radius = np.sqrt(self.image_size**2 / 2)
        self.device = config.device

    def create_circle_images(self, image_size, radii):
        # Create a meshgrid of coordinates
        y, x = torch.meshgrid(torch.arange(image_size, device=self.device), torch.arange(image_size, device=self.device))
        y, x = y.float(), x.float()
        
        # Calculate the center coordinates of the circle
        center = image_size // 2

        # Calculate distance from the center for each coordinate
        dist_from_center = torch.sqrt((x - center) ** 2 + (y - center) ** 2)

        # Initialize a tensor to hold all the circle images
        num_radii = len(radii)
        images = torch.ones((num_radii, image_size, image_size), device=self.device)

        # Create circle images for each radius
        for idx, radius in enumerate(radii):
            images[idx] = (dist_from_center > radius).float()

        return images
    
    def schedule(self, t):
        return self.max_radius * (1 - cosine_schedule(t).sqrt())
    
    def forward(self, x, t):
        radii = self.schedule(t)
        circle_images = self.create_circle_images(self.image_size, radii).to(x.device)
        return x * circle_images[:, None, :, :]
        


class Fade(nn.Module):
    def __init__(self):
        super(Fade, self).__init__()

    def forward(self, x, t):
        fade = cosine_schedule(t).sqrt().to(x.device)
        fade[fade < 0.001] = 0.0
        # reshape sigmas for color images
        if len(x.shape) == 4:
            fade = fade[:, None, None, None]
        elif len(x.shape) == 3:
            fade = fade[:, None, None]
        return x * fade


class DCTBlur(nn.Module):

    def __init__(self, max_blur, image_size, device, min_scale=0.0) :
        super(DCTBlur, self).__init__()
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
        self.max_blur = max_blur
        self.min_blur = 0.5
        self.min_scale = min_scale

    def schedule(self, t):
        return torch.exp(np.log(self.min_blur) * (1 - t) + np.log(self.max_blur) * t)

        return self.max_blur * torch.sin(t * torch.pi / 2)**2
        #return self.max_blur * torch.sin(t * torch.pi / 2)**2

    def forward(self, x, t):
        # ensure float64 for idct precision
        dtype = x.dtype
        if "mps" not in str(x.device): x = x.to(torch.float64)
        # reshape sigmas for color images
        if len(x.shape) == 4:
            sigmas = self.schedule(t)[:, None, None, None]
            t = t[:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.schedule(t)[:, None, None]
            t = t[:, None, None]
        
        # convert sigma to tau
        tau = sigmas ** 2 / 2

        fade_freq = torch.exp(- self.frequencies_squared * tau)
        # limit fade to 0.01 to 0.0 to prevent information leaking
        fade_freq[fade_freq < 0.01] = 0.0
    
        dct_x = torch_dct.dct_2d(x, norm='ortho')
        blurred_dct_x = dct_x * (fade_freq * (1 - self.min_scale) + self.min_scale)
        blurred_x =  torch_dct.idct_2d(blurred_dct_x, norm='ortho')

        return blurred_x.to(dtype)
    
class Vignette(nn.Module):
    def __init__(self, config, max_sigma, image_size, device):
        super(Vignette, self).__init__()
        self.image_size = image_size
        self.device = device
        self.max_sigma = max_sigma

        self.channels = config.data.num_channels

        if image_size % 2 == 0:
            self.kernel_size = image_size + 1
        else:
            self.kernel_size = image_size

    def gaussian_kernel(self, batch_size: int, channels: int, kernel_size: int, sigmas: torch.Tensor):
        """ Returns 2D gaussian kernel of batch size with size x size dimensions and standard deviation sigma. Sigmas is a vector with a different sigma for every element in the batch. Final tensor shape is [B, C, size, size]"""
        kernels = []
        for sigma in sigmas:
            kernel = tgm.image.get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
            # normalise the kernel
            kernel = kernel / kernel.max()
            kernel = 1 - kernel

            kernels.append(kernel)
        kernels = torch.stack(kernels)

        if self.image_size % 2 == 0:
            kernels = kernels[:, 1:, 1:]

        kernels = kernels.unsqueeze(1).repeat(1, channels, 1, 1)

        return kernels
        
    def schedule(self, t):
        return  self.max_sigma * t    
      
    def forward(self, x, t):
        sigmas = self.schedule(t)

        vignette = self.gaussian_kernel(x.shape[0], self.channels, self.kernel_size, sigmas)

        return x * vignette.to(x.device)
    
class Noise(nn.Module):
    def __init__(self, config):
        super(Noise, self).__init__()

    def forward(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        a = cosine_schedule(t)

        if len(x.shape) == 4:
            a = a[:, None, None, None]
        elif len(x.shape) == 3:
            a = a[:, None, None]

        return a.sqrt() * x + (1-a).sqrt() * noise
    
class ComboDegrader(nn.Module):
    def __init__(self, degredation_list):
        super(ComboDegrader, self).__init__()
        self.degradation_list = degredation_list

    def forward(self, x, t):
        for degradation in self.degradation_list:
            x = degradation(x, t)
        return x

def create_forward_process_from_sigmas(config, device):
    forward_process_module = DCTBlur(config.model.blur_sigma_max , config.data.image_size, device)
    return forward_process_module

def fade_forward_process(config):
    return Fade()

def vignette_forward_process(config):
    return Vignette(config, config.model.blur_sigma_max, config.data.image_size, config.device)

def hard_vignette_forward_process(config):
    return HardVignette(config)

def noise_forward_process(config):
    return Noise(config)

def combo_forward_process(config, process_list):
    return ComboDegrader(process_list)


"""Utilities related to log-likelihood evaluation"""


def KL(dists, sigma0, sigma1, dim):
    # Calculates a matrix of KL divergences between spherical Gaussian distributions
    # with distances dists between their centers, where dists is a matrix
    return 0.5 * ((sigma0**2/sigma1**2)*dim + (dists)**2/sigma1**2 - dim + 2*dim*np.log(sigma1/sigma0))


def L_K_upperbound(K, trainloader, testloader, blur_module, sigma_inf, sigma_prior, dim,
                   train_size, test_size, device='cpu'):

    # Calculates the upper bound for the term E_q[KL[q(x_K|x_0)|p(x_K)]]
    # in a memory-efficient way, that is, calculates the distances between
    # test and training data points in batches, and uses those distances to calculate
    # the upper bound

    KL_div_upper_bound = torch.zeros(test_size, device=device)
    testdata_count = 0
    count = 0
    for testbatch in testloader:
        logging.info("Batch {}".format(count))
        count += 1
        blur_fwd_steps_test = [K] * len(testbatch[0])
        testbatch = blur_module(testbatch[0].to(
            device), blur_fwd_steps_test).reshape(len(testbatch[0]), -1)
        dists = torch.zeros(train_size, len(testbatch), device=device)
        traindata_count = 0
    # Get distances between the test batch and training data
        for trainbatch in trainloader:
            blur_fwd_steps_train = [K] * len(trainbatch[0])
            trainbatch = blur_module(trainbatch[0].to(
                device), blur_fwd_steps_train).reshape(len(trainbatch[0]), -1)
            dists[traindata_count:traindata_count +
                  len(trainbatch), :] = torch.cdist(trainbatch, testbatch)
            traindata_count += len(trainbatch)
        # Calculate the upper bounds on the KL divergence for each test batch element
        kl_divs = KL(dists, sigma_inf, sigma_prior, testbatch.shape[-1])
        inference_entropy = dim*0.5 * \
            torch.log(
                2*np.pi*torch.exp(torch.tensor([1]))*sigma_inf**2).to(device)
        cross_entropies = kl_divs + inference_entropy
        # log-sum-exp trick
        log_phi = -kl_divs - torch.logsumexp(-kl_divs, 0)[None, :]
        phi = torch.exp(log_phi)
        KL_div_upper_bound_batch = -inference_entropy + \
            (phi * (cross_entropies + log_phi + np.log(train_size))).sum(0)
        KL_div_upper_bound[testdata_count:testdata_count +
                           len(testbatch)] = KL_div_upper_bound_batch
        testdata_count += len(testbatch)
    return KL_div_upper_bound


def neg_ELBO(config, trainloader, testloader, blur_module, sigma, delta, image_size,
             train_size, test_size, model, device='cpu', num_epochs=10):
    """Estimates the terms in the negative evidence lower bound for the model
    num_epochs: Used for the estimation of terms L_k: How many epochs through these?"""

    logging.info("Calculating the upper bound for L_K...")
    L_K_upbound = L_K_upperbound(config.model.K, trainloader, testloader, blur_module, sigma,
                               delta, image_size**2, train_size, test_size, device)
    logging.info("... done! Value {}, len {}".format(
      L_K_upbound, len(L_K_upbound)))

    model_fn = get_model_fn(model, train=False)
    num_dims = image_size**2 * next(iter(trainloader))[0].shape[1]

    # There are K - 1 intermediate scales
    L_others = torch.zeros(config.model.K, device=device)
    mse_losses = torch.zeros(config.model.K, device=device)

    logging.info("Calculating the other terms...")
    with torch.no_grad():
        # Go through the set a few times for more accuracy, not just once
        for i in range(num_epochs):
            for testbatch in testloader:
                testbatch = testbatch[0].to(device).float()
                batch_size = len(testbatch)
                fwd_steps = torch.randint(
                    1, config.model.K, (batch_size,), device=device)
                blurred_batch = blur_module(testbatch, fwd_steps).float()
                less_blurred_batch = blur_module(testbatch, fwd_steps-1).float()
                noise = torch.randn_like(blurred_batch) * sigma
                perturbed_data = noise + blurred_batch
                diff = model_fn(perturbed_data, fwd_steps)
                prediction = perturbed_data + diff
                mse_loss = ((less_blurred_batch - prediction)
                            ** 2).sum((1, 2, 3))
                loss = mse_loss / delta**2
                loss += 2*num_dims*np.log(delta/sigma)
                loss += sigma**2/delta**2*num_dims
                loss -= num_dims
                loss /= 2
                # Normalize so that the significance of these terms matches with L_K and L_0
                # This way, we only go through once for each data point
                loss *= (config.model.K-1)
                mse_loss *= (config.model.K-1)
                L_others.scatter_add_(0, fwd_steps, loss)
                mse_losses.scatter_add_(0, fwd_steps, mse_loss)

        L_others = L_others / (test_size*num_epochs)
        mse_losses = mse_losses / (test_size*num_epochs)

        # Calculate L_0
        for testbatch in testloader:
            testbatch = testbatch[0].to(device).float()
            batch_size = len(testbatch)
            blurred_batch = blur_module(testbatch, [1]).float()
            non_blurred_batch = testbatch
            fwd_steps = torch.ones(batch_size, device=device)
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            diff = model_fn(perturbed_data, fwd_steps)
            prediction = perturbed_data + diff
            mse_loss = ((non_blurred_batch - prediction)**2).sum((1, 2, 3))
            loss = 0.5*mse_loss/delta**2
            # Normalization constant
            loss += num_dims*np.log(delta*np.sqrt(2*np.pi))
            L_others[0] += loss.sum()
            mse_losses[0] += mse_loss.sum()
        L_others[0] = L_others[0] / test_size
        mse_losses[0] = mse_losses[0] / test_size

    logging.info("... Done! Values {}".format(L_others))
    return L_K_upbound.detach().cpu(), L_others.detach().cpu(), mse_losses.detach().cpu()


"""The next two functions based on https://github.com/yang-song/score_sde"""


def create_model(config, device_ids=None):
    """Create the model."""
    if config.model.type == 'diffusion':
        model = UNetModel(config)
        model = model.to(config.device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif config.model.type == 'vae':
        print("Creating VAE model")
        model = DiffusionVAE(config)
        model = model.to(config.device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        raise NotImplementedError("Model type not implemented")
    return model


def get_model_fn(model, train=False, sample=False, z=None):
    """A wrapper for using the model in eval or train mode"""
    def model_fn(x, *args):
        """Args:
                x: A mini-batch of input data.
                fwd_steps: A mini-batch of conditioning variables for different levels.
        """
        if not train:
            model.eval()
            return model(x, *args) if not sample else model.module.sample(x, *args)
        else:
            model.train()
            return model(x, *args)
    return model_fn
