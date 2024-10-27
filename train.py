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

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
#flags.DEFINE_string("initialization", "prior", "How to initialize sampling")


def main(argv):
    train(FLAGS.config, FLAGS.workdir)


def train(config, workdir):
    """Runs the training pipeline. 
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """

    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")

    # Set up logging
    track_experiment(config)
    # Seed everything
    utils.seed_everything(config.seed)

    # Create directory for saving intermediate samples
    sample_dir = os.path.join(workdir, "samples")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    # Create directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # export config to json
    with open(os.path.join(workdir, "config.json"), "w") as f:
        f.write(config.to_json_best_effort())

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

    # Build data iterators
    trainloader, testloader = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the forward process definition
    scales = config.model.blur_schedule
    degrader = create_degrader(config)

    # Get the loss function
    train_step_fn = losses.get_step_fn(train=True, scales=scales, config=config, optimize_fn=optimize_fn,
                                       heat_forward_module=degrader)
    eval_step_fn = losses.get_step_fn(train=False, scales=scales, config=config, optimize_fn=optimize_fn,
                                      heat_forward_module=degrader)

    # Building sampling functions
    delta = config.model.delta
    initial_sample, _ = sampling.get_initial_sample(config, degrader, delta, config.eval.batch_size)

    if config.degrader == 'noise':
        sampling_fn = sampling.get_sampling_fn_noise_forward(config,
                                                        initial_sample, intermediate_sample_indices=list(
                                                            range(config.model.K+1)),
                                                        delta=config.model.delta, device=config.device,
                                                        degradation_operator=degrader)
    else:
        sampling_fn = sampling.get_sampling_fn_inverse_heat(config, initial_sample, intermediate_sample_indices=list(
            range(config.model.K+1)), delta=config.model.delta, device=config.device, degradation_operator=degrader)
    
    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))

    # For analyzing the mean values of losses over many batches, for each scale separately
    pooled_losses = torch.zeros(len(scales))

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        try:
            batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            batch = next(train_iter)[0].to(config.device).float()
        loss, losses_batch, fwd_steps_batch = train_step_fn(state, batch, step)

        writer.add_scalar("training_loss", loss.item(), step)
        wandb.log({"training_loss": loss.item(), "step": step})

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info("Saving temporary checkpoint")
            utils.save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            logging.info("Starting evaluation")
            # Use 1 batches for test-set evaluation, arbitrary choice
            N_evals = 1
            for i in range(N_evals):
                try:
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                eval_loss, eval_losses_batch, fwd_steps_batch = eval_step_fn(state, eval_batch, step)
                eval_loss = eval_loss.detach()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            logging.info("step: %d, train_loss: %.5e" % (step, loss.item()))
            wandb.log({"eval_loss": eval_loss.item(), "step": step})

            if config.model.type == 'vae':
                # also save kl - reconstruction divide
                train_reconstruction_loss = torch.mean(losses_batch[0])
                train_kl_div = torch.mean(losses_batch[1])
                eval_reconstruction_loss = torch.mean(eval_losses_batch[0])
                eval_kl_div = torch.mean(eval_losses_batch[1])
                wandb.log({"train_reconstruction_loss": train_reconstruction_loss.item(), "train_kl_div": train_kl_div.item(), "eval_reconstruction_loss": eval_reconstruction_loss.item(), "eval_kl_div": eval_kl_div.item(), "step": step})



        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info("Saving a checkpoint")
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)

        # Generate samples periodically
        if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps:
            logging.info("Sampling...")
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            sample, n, intermediate_samples, model_predictions = sampling_fn(model_evaluation_fn)
            ema.restore(model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "final.np")
            utils.save_png(this_sample_dir, sample, "final.png")
            if initial_sample != None:
                utils.save_png(this_sample_dir, initial_sample, "init.png")
            # save sample trajectory
            utils.save_gif(this_sample_dir, intermediate_samples)
            utils.save_video(this_sample_dir, intermediate_samples)
            # save model prediction trajectory
            utils.save_gif(this_sample_dir, model_predictions, "model_predictions.gif")
            utils.save_video(this_sample_dir, model_predictions, "model_predictions.mp4")

def create_degrader(config):
    if config.degrader == 'hard_vignette':
        return mutils.hard_vignette_forward_process(config)
    elif config.degrader == 'vignette':
        return mutils.vignette_forward_process(config)
    elif config.degrader == 'blur':
        return mutils.create_forward_process_from_sigmas(config, config.device)
    elif config.degrader == 'fade':
        return mutils.fade_forward_process(config)
    elif config.degrader == 'blur_fade':
        blur = mutils.create_forward_process_from_sigmas(config, config.device)
        fade = mutils.fade_forward_process(config)
        return mutils.combo_forward_process(config, [blur, fade])
    elif config.degrader == 'noise':
        return mutils.noise_forward_process(config)

    

def track_experiment(config):
    wandb.init(
    # set the wandb project where this run will be logged
    project="diffusion-vae",

    # track hyperparameters and run metadata
    config=config.to_dict(),
    )

if __name__ == "__main__":
    app.run(main)
