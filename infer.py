from typing import List, Optional, Union
import PIL
from PIL import Image
import numpy as np
import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from anyword_data import AnyWordDataset
import os
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel, scheduler: DDPMScheduler):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device"):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[torch.FloatTensor, PIL.Image.Image],
        glyph: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask: Union[torch.FloatTensor, PIL.Image.Image],
        num_inference_steps: int = 50,
        device=None
    ):
        if mask_image is None:
            raise ValueError("`mask_image` input cannot be undefined.")

        batch_size = prompt.shape[0]
        vae.to(device)
        unet.to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Preprocess mask and image
        vae_scale_factor = self.vae_scale_factor
        _, _, mask_height, mask_width = mask.size()
        mask = torch.nn.functional.interpolate(mask, size=[mask_width // vae_scale_factor, mask_height // vae_scale_factor])

        glyph_latents = vae.encode(glyph).latent_dist.sample() * vae.config.scaling_factor
        masked_image_latents = vae.encode(mask_image).latent_dist.sample() * vae.config.scaling_factor

        shape = (batch_size, vae.config.latent_channels, mask_height // vae_scale_factor, mask_width // vae_scale_factor)
        latents = randn_tensor(shape, generator=torch.manual_seed(20), device=device) * self.scheduler.init_noise_sigma

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # glyph_latents
                sample = torch.cat([latent_model_input, masked_image_latents, glyph_latents, mask], dim=1)
                noise_pred = unet(sample=sample, timestep=t, encoder_hidden_states=prompt, ).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.update()

        pred_latents = latents / vae.config.scaling_factor
        image_vae = vae.decode(pred_latents).sample
        image = (image_vae / 2 + 0.5) * 255.0
        return image, image_vae

# Initialize models
vae = AutoencoderKL.from_pretrained("./model/vae_ft/checkpoint-150000/vae")
unet = UNet2DConditionModel.from_pretrained("./model/step2/checkpoint-25000/unet")
noise_scheduler = DDPMScheduler.from_pretrained("./model/stable-diffusion-v2-1/scheduler")

# Create pipeline
pipe = StableDiffusionPipeline(vae=vae, unet=unet, scheduler=noise_scheduler)

# Paths to dataset
save_dir = "./output/ic13"
os.makedirs(os.path.join(save_dir, "region"),  exist_ok=True)
os.makedirs(os.path.join(save_dir, "local"),  exist_ok=True)

# Create dataset and dataloader
datasets = AnyWordDataset(
    json_path="./benchmark/ic13/test.json",
    resolution=256,
    seed=42,
    ttf_size=64,
    max_len=25,
)

# Set batch size to 1 for single card inference
dataloader = DataLoader(datasets, shuffle=False, batch_size=32, num_workers=0)

cnt = 0
results = {}

for batch in tqdm(dataloader):
    imgs = batch["image"].to("cuda")
    masked_images = batch["masked_image"].to("cuda")
    masks = batch["mask"].to("cuda")
    ttf_imgs = batch["ttf_img"].to("cuda")
    glyphs = batch["glyph"].to("cuda")
    texts = batch["text"]

    # Generate images using pipeline
    image, _ = pipe(
        prompt=ttf_imgs,
        glyph=glyphs,
        mask_image=masked_images,
        mask=masks,
        num_inference_steps=20,
        device=torch.device("cuda")
    )

    for i, img in enumerate(image):

        mask_np = masks[i].cpu().detach().numpy().astype(np.uint8)
        coords = np.column_stack(np.where(mask_np == 0))
        if coords.size > 0:
            y_min, x_min = coords[:, 1].min(), coords[:, 2].min()
            y_max, x_max = coords[:, 1].max(), coords[:, 2].max()
            cropped_output_image = img[:, y_min:y_max+1, x_min:x_max+1]
        else:
            cropped_output_image = img

        # Save cropped image
        file_idx = i + cnt
        img_np = img.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(save_dir, "local", f"{file_idx}.png"))

        cropped_output_image_np = cropped_output_image.cpu().permute(1, 2, 0).float().detach().numpy().astype(np.uint8)
        cropped_output_image_pil = Image.fromarray(cropped_output_image_np)
        file_path = os.path.join(save_dir, "region", f"{file_idx}.png")
        cropped_output_image_pil.save(file_path)
        results[f"{file_idx}.png"] = texts[i]

    cnt += len(image)

# Save results
with open(f"{save_dir}/labels.json", 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=4)
