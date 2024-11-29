import os
from PIL import Image, ImageFile

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional
import accelerate
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from tqdm.auto import tqdm
import diffusers
from anyword_data_lmdb import AnyWordLmdbDataset
from mmengine import Config

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

check_min_version("0.15.0.dev0")
logger = get_logger(__name__, log_level="INFO")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def parse_cfgs():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    cfgs = Config.fromfile(args.config)

    # default to using the same revision for the non-ema model if not specified
    if cfgs.non_ema_revision is None:
        cfgs.non_ema_revision = cfgs.revision

    return cfgs


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

    
def main():
    cfgs = parse_cfgs()

    if cfgs.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(cfgs.output_dir, cfgs.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=cfgs.checkpoints_total_limit)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfgs.gradient_accumulation_steps,
        mixed_precision=cfgs.mixed_precision,
        log_with=cfgs.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfgs.seed is not None:
        set_seed(cfgs.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfgs.push_to_hub:
            if cfgs.hub_model_id is None:
                repo_name = get_full_repo_name(Path(cfgs.output_dir).name, token=cfgs.hub_token)
            else:
                repo_name = cfgs.hub_model_id
            create_repo(repo_name, exist_ok=True, token=cfgs.hub_token)
            repo = Repository(cfgs.output_dir, clone_from=repo_name, token=cfgs.hub_token)

            with open(os.path.join(cfgs.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif cfgs.output_dir is not None:
            os.makedirs(cfgs.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfgs.scheduler_path
    )

    vae = AutoencoderKL.from_pretrained(
        cfgs.pretrained_vae, revision=cfgs.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfgs.pretrained_unet, revision=cfgs.non_ema_revision, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    ) 

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Create EMA for the unet.
    if cfgs.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            cfgs.pretrained_unet, revision=cfgs.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if cfgs.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if cfgs.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if cfgs.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if cfgs.gradient_checkpointing:
        unet.enable_gradient_checkpointing()


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfgs.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfgs.scale_lr:
        cfgs.learning_rate = (
            cfgs.learning_rate * cfgs.gradient_accumulation_steps * cfgs.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if cfgs.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=cfgs.learning_rate,
        betas=(cfgs.adam_beta1, cfgs.adam_beta2),
        weight_decay=cfgs.adam_weight_decay,
        eps=cfgs.adam_epsilon,
    )

    def collate_fn_train(examples):
        pixel_values = torch.stack([example["image"] for example in examples]).float()
        masks = torch.stack([example["mask"] for example in examples]).float()
        masked_images = torch.stack([example["masked_image"] for example in examples]).float()
        ttf_imgs = torch.stack([example["ttf_img"] for example in examples]).float()
        glyphs = torch.stack([example["glyph"] for example in examples]).float()

        batch = {
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images,
            "ttf_images": ttf_imgs,
            "glyphs": glyphs
        }

        return batch

    datasets_st = AnyWordLmdbDataset(
        lmdb_path=cfgs.lmdb_path,
        resolution=cfgs.resolution,
        seed=cfgs.seed,
        ttf_size=cfgs.ttf_size,
        max_len=cfgs.max_len,
        )

    train_dataloader = torch.utils.data.DataLoader(
        datasets_st,
        shuffle=True,
        collate_fn=collate_fn_train,
        batch_size=cfgs.train_batch_size,
        num_workers=cfgs.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfgs.gradient_accumulation_steps)
    if cfgs.max_train_steps is None:
        cfgs.max_train_steps = cfgs.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfgs.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfgs.lr_warmup_steps * cfgs.gradient_accumulation_steps,
        num_training_steps=cfgs.max_train_steps * cfgs.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if cfgs.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfgs.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfgs.max_train_steps = cfgs.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfgs.num_train_epochs = math.ceil(cfgs.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        cfgs_dict = vars(cfgs)
        valid_types = (int, float, str, bool, torch.Tensor)
        cfgs_dict = {key: value for key, value in cfgs_dict.items() if isinstance(value, valid_types)}
        accelerator.init_trackers("unet-fine-tune", config=cfgs_dict)

    # Train!
    total_batch_size = cfgs.train_batch_size * accelerator.num_processes * cfgs.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(datasets_st)}")
    logger.info(f"  Num Epochs = {cfgs.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfgs.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfgs.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfgs.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if cfgs.resume_from_checkpoint:
        if cfgs.resume_from_checkpoint != "latest":
            path = cfgs.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfgs.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfgs.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfgs.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfgs.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * cfgs.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * cfgs.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, cfgs.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # guidance_scale = cfgs.guidance_scale
    for epoch in range(first_epoch, cfgs.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfgs.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfgs.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)

                # Rex: prepare mask && mask latent as input of UNET         
                _, _, width, height = batch["masks"].size()
                mask = batch["masks"]
                mask = torch.nn.functional.interpolate(
                    mask, size=[width // vae_scale_factor, height // vae_scale_factor]
                )
                mask = mask.to(weight_dtype)

                masked_image_latents = vae.encode(batch["masked_images"].to(weight_dtype)).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor

                glyph_latents = vae.encode(batch["glyphs"].to(weight_dtype)).latent_dist.sample()
                glyph_latents = glyph_latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                sample = torch.cat([noisy_latents, masked_image_latents, glyph_latents, mask], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                # Predict the noise residual and compute loss
                model_pred = unet(
                    sample=sample, 
                    timestep=timesteps, 
                    encoder_hidden_states=batch["ttf_images"].to(weight_dtype), 
                ).sample  

                model_pred = model_pred.to(weight_dtype)

                loss = F.mse_loss(model_pred, target, reduction="mean")

                # # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfgs.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfgs.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), cfgs.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if cfgs.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % cfgs.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(cfgs.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= cfgs.max_train_steps:
                break
    if accelerator.is_main_process:
        save_path = os.path.join(cfgs.output_dir, f"final")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
