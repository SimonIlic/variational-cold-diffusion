"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from model_code import utils as mutils
import torch.distributions
from scripts import hoogeboom as hgb


def get_optimizer(config, params):
    """Returns an optimizer object based on `config`.
    Copied from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`.
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if config.optim.automatic_mp:
        def optimize_fn(optimizer, params, step, scaler, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative).
            Before that, unscales the gradients to the regular range from the 
            scaled values for automatic mixed precision"""
            scaler.unscale_(optimizer)
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            # Since grads already scaled, this just takes care of possible NaN values
            scaler.step(optimizer)
            scaler.update()
    else:
        def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                        warmup=config.optim.warmup,
                        grad_clip=config.optim.grad_clip):
            """Optimizes with warmup and gradient clipping (disabled if negative)."""
            if warmup > 0:
                for g in optimizer.param_groups:
                    g['lr'] = lr * np.minimum(step / warmup, 1.0)
            if grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
    return optimize_fn


def get_label_sampling_function(K):
    return lambda batch_size, device: torch.randint(1, K, (batch_size,), device=device)


def get_inverse_heat_loss_fn(config, train, scales, device, heat_forward_module):

    sigma = config.model.sigma
    label_sampling_fn = get_label_sampling_function(config.model.K)
    blur_schedule = torch.tensor(config.model.blur_schedule, device=config.device)

    if config.model.type == 'vae':
        if config.model.loss_type == 'risannen':
            def loss_fn(model, batch):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                fwd_steps = label_sampling_fn(batch.shape[0], batch.device)
                scales = blur_schedule[fwd_steps]
                blurred_batch = heat_forward_module(batch, fwd_steps).float()
                less_blurred_batch = heat_forward_module(batch, fwd_steps-1).float()

                noise = torch.randn_like(blurred_batch) * sigma
                perturbed_data = blurred_batch + noise

                diff, z, mu, log_var = model_fn(less_blurred_batch, perturbed_data, scales)
                prediction = blurred_batch + diff

                reconstruction_loss = (less_blurred_batch - prediction)**2
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)

                loss = torch.mean(reconstruction_loss) + torch.mean(kl_div)

                return loss, (reconstruction_loss, kl_div), fwd_steps
            
        elif config.model.loss_type == 'hoogeboom':
            def loss_fn(model, batch, iteration=0):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                t = torch.tensor(np.float32(np.random.uniform(0, 1, batch.shape[0])), device=device)
                
                blurred, sigma = hgb.diffuse(batch, t, config.data.image_size, config.model.blur_sigma_max)
                eps = torch.randn_like(blurred)
                perturbed_blurred = blurred + sigma * eps

                prediction, z, (mu, log_var) = model_fn(perturbed_blurred, blurred, t)

                # l2 norm of eps prediction
                reconstruction_loss = (eps - prediction)**2
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                # kl divergence of z_t encoder
                kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)

                if iteration > 1000:
                    loss = torch.mean(reconstruction_loss) + torch.mean(kl_div) * np.clip(iteration / 10000, 0, 1)
                else:
                    loss = torch.mean(reconstruction_loss)

                iteration += 1
                print(iteration)

                return loss, (reconstruction_loss, kl_div), t


            
        elif config.model.loss_type == 'trajectory_matching_old':
            def loss_fn(model, batch):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                fwd_steps = label_sampling_fn(batch.shape[0], batch.device)
                scales = blur_schedule[fwd_steps]

                input = heat_forward_module(batch, blur_schedule[fwd_steps+1]).float()
                intermediate = heat_forward_module(batch, blur_schedule[fwd_steps]).float()

                # get preturbed input
                with torch.no_grad():
                    input, z, latent_params = model_fn(input, intermediate, blur_schedule[fwd_steps+1])
                
                target = heat_forward_module(batch, blur_schedule[fwd_steps-1]).float()
                prediction, z, (mu, log_var) =model_fn(input, target, scales)

                # l2 norm of the difference between the reconstructed and the original image
                reconstruction_loss = (prediction - target)**2
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)

                loss = torch.mean(reconstruction_loss) + torch.mean(kl_div)

                return loss, (reconstruction_loss, kl_div), fwd_steps
            
        elif config.model.loss_type == 'trajectory_matching':
            K = config.model.K
            def loss_fn(model, batch, step=0):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                
                # warmup is normal optimization of known forward
                if step < config.model.warmup:
                    t = torch.tensor(np.float32(np.random.uniform(1/K, 1, batch.shape[0])), device=device)
                    input = heat_forward_module(batch, t)
                    target = heat_forward_module(batch, t - 1/K)
                
                # after warmup we start trajectory matching
                else:
                    # split batch into two
                    batch1, batch2 = torch.split(batch, int(config.model.trajectory_fraction * batch.shape[0]))

                    # batch1 will be perturbed
                    t1 = torch.tensor(np.float32(np.random.uniform(1/K, 1 - 1/K, batch1.shape[0])), device=device)
                    input1 = heat_forward_module(batch1, t1 + 1/K)
                    intermediate = heat_forward_module(batch1, t1)
                    with torch.no_grad():
                        input1, z, latent_params = model_fn(input1, intermediate, t1 + 1/K)
                    target1 = heat_forward_module(batch1, t1 - 1/K)

                    # batch2 will be deterministc
                    t2 = torch.tensor(np.float32(np.random.uniform(1/K, 1, batch2.shape[0])), device=device)
                    input2 = heat_forward_module(batch2, t2)
                    target2 = heat_forward_module(batch2, t2 - 1/K)

                    input = torch.cat((input1, input2), dim=0)
                    target = torch.cat((target1, target2), dim=0)
                    t = torch.cat((t1, t2), dim=0)

                prediction, z, (mu, log_var) = model_fn(input, target, t)

                reconstruction_loss = (prediction - target)**2
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)

                # KL annealing
                if step > config.model.warmup:
                    loss = torch.mean(reconstruction_loss) + torch.mean(kl_div) * np.clip((step - config.model.warmup) / config.model.anneal_steps, 0,1) * config.model.beta
                else:
                    loss = torch.mean(reconstruction_loss)


                return loss, (reconstruction_loss, kl_div), t
                    
                    
        elif config.model.loss_type == 'bansal':
            def loss_fn(model, batch):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                
                t = torch.tensor(np.float32(np.random.uniform(0, 1, batch.shape[0])), device=device)

                blurred_batch = heat_forward_module(batch, t).float()
                reconstructed, z, (mu, log_var) = model_fn(blurred_batch, batch, t)

                # l1 norm of the difference between the reconstructed and the original image
                reconstruction_loss = torch.abs(reconstructed - batch)
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)

                loss = torch.mean(reconstruction_loss) + torch.mean(kl_div)

                return loss, (reconstruction_loss, kl_div), t

        elif config.model.loss_type == 'variable_t':
            def loss_fn(model, batch):
                model_fn = mutils.get_model_fn(
                    model, train=train)
                # get time for input and target
                input_t = np.float32(np.random.uniform(0, 1, batch.shape[0]))
                input_t[input_t.argmax()] = 1.0  # clip max to 1 to expose model to completely dark image in training
                target_t = np.float32(np.random.uniform(0, input_t))

                # move times to device
                input_t = torch.tensor(input_t, device=device)
                target_t = torch.tensor(target_t, device=device)

                blurred_batch = heat_forward_module(batch, input_t).float()
                target_batch = heat_forward_module(batch, target_t).float()
                reconstructed, z, latent_params = model_fn(target_batch, blurred_batch, input_t - target_t)

                # l1 norm of the difference between the reconstructed and the original image
                reconstruction_loss = torch.abs(reconstructed - target_batch)
                reconstruction_loss = torch.sum(reconstruction_loss.reshape(reconstruction_loss.shape[0], -1), dim=-1)

                kl_divs = []
                for params in latent_params:
                    mu, log_var = params[:, 0, ...], params[:, 1, ...]
                    mu = mu.flatten(start_dim=1)
                    log_var = log_var.flatten(start_dim=1)

                    kl_div = 0.5 * torch.sum(torch.exp(log_var) + mu**2 - 1 - log_var, dim=-1)
                    kl_divs.append(kl_div)

                kl_divs = torch.cat(kl_divs)

                loss = torch.mean(reconstruction_loss) + torch.mean(kl_divs)

                return loss, (reconstruction_loss, kl_divs), input_t


    elif config.model.loss_type == 'risannen':
        def loss_fn(model, batch):
            model_fn = mutils.get_model_fn(
                model, train=train)  # get train/eval model
            fwd_steps = label_sampling_fn(batch.shape[0], batch.device)
            blurred_batch = heat_forward_module(batch, fwd_steps).float()
            less_blurred_batch = heat_forward_module(batch, fwd_steps-1).float()
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            diff = model_fn(perturbed_data, fwd_steps)
            prediction = perturbed_data + diff
            losses = (less_blurred_batch - prediction)**2
            losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)
            return loss, losses, fwd_steps
    
    elif config.model.loss_type == 'bansal':
        def loss_fn(model, batch):
            model_fn = mutils.get_model_fn(
                model, train=train)
            fwd_steps = label_sampling_fn(batch.shape[0], batch.device)
            blurred_batch = heat_forward_module(batch, fwd_steps).float()
            reconstructed = model_fn(blurred_batch, fwd_steps)

            # l1 norm of the difference between the reconstructed and the original image
            losses = torch.abs(reconstructed - batch)
            losses = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)

            return loss, losses, fwd_steps

    return loss_fn


def get_step_fn(train, scales, config, optimize_fn=None,
                heat_forward_module=None, device=None):
    """A wrapper for loss functions in training or evaluation
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if device == None:
        device = config.device

    loss_fn = get_inverse_heat_loss_fn(config, train,
                                       scales, device, heat_forward_module=heat_forward_module)

    # For automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    def step_fn(state, batch, step):
        """Running one step of training or evaluation.
        Returns:
                loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            if config.optim.automatic_mp:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, losses_batch, fwd_steps_batch = loss_fn(model, batch, step)
                    # amp not recommended in backward pass, but had issues getting this to work without it
                    # Followed https://github.com/pytorch/pytorch/issues/37730
                    scaler.scale(loss).backward()
                scaler.scale(losses_batch)
                optimize_fn(optimizer, model.parameters(), step=state['step'],
                            scaler=scaler)
                state['step'] += 1
                state['ema'].update(model.parameters())
            else:
                optimizer.zero_grad()
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch, step)
                loss.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss, losses_batch, fwd_steps_batch

    return step_fn
