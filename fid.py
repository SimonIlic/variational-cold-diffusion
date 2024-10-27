import os
from pathlib import Path
import logging
from scripts import sampling
from model_code import utils as mutils
import torch
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import scripts.datasets as datasets
from tqdm import tqdm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("checkpoint", None,
                     "Checkpoint number to use for custom sampling")
flags.mark_flags_as_required(["workdir", "config", "checkpoint"])
flags.DEFINE_integer("save_sample_freq", 1,
                     "How often to save samples for output videos?")
flags.DEFINE_float(
    "delta", 0.0, "The standard deviation of noise to add at each step with predicted reverse blur")
flags.DEFINE_integer(
    "batch_size", None, "Batch size of sampled images. Defaults to the training batch size")
flags.DEFINE_bool("same_init", False,
                  "Whether to initialize all samples at the same image")
flags.DEFINE_bool("share_noise", False,
                  "Whether to use the same noises for each image in the generated batch")
flags.DEFINE_integer(
    "num_points", 10, "Default amount of points for sweeping the input from one place to another")
flags.DEFINE_float("final_noise", None,
                   "How much should the noise at the end be? Linear interpolation from noise_amount ot this. If none, use noise_amount")
flags.DEFINE_bool("interpolate", False, "Whether to do interpolation")
flags.DEFINE_integer(
    "number", None, "add a number suffix to generated sample in interpolate")
# add flag for K steps
flags.DEFINE_integer("K", None, "Number of steps in the diffusion process")
# total number of samples
flags.DEFINE_integer("total_samples", 300, "Total number of samples to generate")


def main(argv):
    if FLAGS.interpolate:
        sample_interpolate(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint,
                           FLAGS.delta, FLAGS.num_points, FLAGS.number)
    else:
        sample(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint, FLAGS.save_sample_freq, FLAGS.delta,
               FLAGS.batch_size, FLAGS.share_noise, FLAGS.same_init, FLAGS.K, FLAGS.total_samples)


def sample(config, workdir, checkpoint, save_sample_freq=1,
           delta=0, batch_size=None, share_noise=False, same_init=False, K=None, total_samples=300):
    
    trainloader, testloader = datasets.get_dataset(config,
                                          uniform_dequantization=config.data.uniform_dequantization,
                                          train_batch_size=batch_size)
    train = iter(trainloader)
    test = iter(testloader)

    if batch_size == None:
        batch_size = config.training.batch_size
    config.eval.batch_size = batch_size

    if K is not None:
        config.model.K = K

    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:  # Checkpoint means the latest checkpoint
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)

    model_fn = mutils.get_model_fn(model, train=False, sample=True if config.model.type == 'vae' else False)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    logging.info("Running on {}".format(config.device))

    logging.info("Creating the forward process...")
    heat_forward_module = create_degrader(config)
    logging.info("Done")

    intermediate_sample_indices = list(
        range(0, config.model.K+1, save_sample_freq))
    sample_dir = os.path.join(workdir, "fid_samples")
    this_sample_dir = os.path.join(
        sample_dir, "checkpoint_{}".format(checkpoint))

    this_sample_dir = os.path.join(this_sample_dir, "delta_{}".format(delta))
    if same_init:
        this_sample_dir += "_same_init"
    if share_noise:
        this_sample_dir += "_share_noise"

    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Do sampling")
    for n in tqdm(range((total_samples // batch_size) + 1)):
        try:
            batch = next(train)
        except StopIteration:
            train = iter(trainloader)
            batch = next(train)
    
        initial_sample = heat_forward_module(batch[0].to(config.device), torch.ones(batch[0].shape[0], dtype=torch.long).to(config.device))
        print(initial_sample.mean(), initial_sample.std())
        sampling_fn = sampling.get_sampling_fn_inverse_heat(config, initial_sample,
                                                            intermediate_sample_indices, delta, config.device, share_noise=share_noise, degradation_operator=heat_forward_module)
        sample, K, intermediate_samples, model_predictions = sampling_fn(model_fn)
        for i, img in enumerate(sample):
            img_path = os.path.join(this_sample_dir, f"sample_{n}_{i}.png")
            utils.save_image(img, img_path)

    # Concatenate all samples
    logging.info("Sampling done for all batches!")

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

if __name__ == "__main__":
    app.run(main)
