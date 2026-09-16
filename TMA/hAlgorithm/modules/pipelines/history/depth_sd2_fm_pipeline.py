import json
import logging
import os
import shutil
from copy import deepcopy

import diffusers
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from torch.nn.parameter import Parameter
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from hAlgorithm.modules.utils.alignment import align_depth_least_square
from hAlgorithm.modules.utils.multi_res_noise import multi_res_noise_like
from hAlgorithm.utils import colorize_depth_maps

from .outputs import DepthOutput


class DepthSD2FlowMatchPipeline(DiffusionPipeline):
    def __init__(
        self,
        pretrained_model_name_or_path,  # Path or name of the pre-trained model to use
        vae=None,  # Path or name of the pre-trained vae to use
        unet=None,  # Path or name of the pre-trained unet to use
        scheduler=None,  # Scheduler for controlling the denoising process
        denoising_steps=1,  # Number of steps in the denoising process
        seed=None,  # Seed for reproducibility
        gradient_checkpointing=False,  # Whether to use gradient checkpointing to save memory
        weighting_scheme=None,  # Scheme for weighting different components of the loss function
        logit_mean=None,  # Mean value for logits, used in normalization
        logit_std=None,  # Standard deviation for logits, used in normalization
        mode_scale=None,  # Scaling factor for the mode
        depth_gt_type="depth",  # Type of depth gt to use
        depth_gt_mask_type="depth_mask",  # Type of depth gt mask to use
        depth_type="depth_norm",  # Type of depth representation to use
        depth_mask_type="depth_mask",  # Type of mask to apply to the depth data
        sparse_depth_type=None,  # Type of sparse depth data, if any
        multi_res_noise=None,  # Configuration for multi-resolution noise, if enabled
        min_depth=1e-6,
        max_depth=200,
        match_input_res=False,
        **kwargs,  # Additional keyword arguments passed to the parent class
    ):
        super().__init__()

        # Set up the random seed for reproducibility
        self.seed = seed
        self.global_seed_sequence = []  # List to store global seed sequence for batch processing

        # Depth configurations
        self.depth_gt_type = depth_gt_type
        self.depth_gt_mask_type = depth_gt_mask_type
        self.depth_type = depth_type
        self.depth_mask_type = depth_mask_type
        self.sparse_depth_type = sparse_depth_type

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.match_input_res = match_input_res

        # Denoising process configurations
        self.denoising_steps = denoising_steps
        self.gradient_checkpointing = gradient_checkpointing

        # Weighting and normalization parameters
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

        # Multi-resolution noise configuration
        self.multi_res_noise = multi_res_noise
        self.apply_multi_res_noise = self.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.multi_res_noise["strength"]
            self.annealed_mr_noise = self.multi_res_noise["annealed"]
            self.mr_noise_downscale_strategy = self.multi_res_noise["downscale_strategy"]

        # Load FlowMatch scheduler, save with depth
        # scheduler = FlowMatchEulerDiscreteScheduler(
        #     num_train_timesteps=scheduler["num_train_timesteps"],
        #     shift=scheduler["shift"],
        #     use_dynamic_shifting=scheduler["use_dynamic_shifting"],
        #     base_shift=scheduler["base_shift"],
        #     max_shift=scheduler["max_shift"],
        #     base_image_seq_len=scheduler["base_image_seq_len"],
        #     max_image_seq_len=scheduler["max_image_seq_len"],
        # )
        if scheduler is None:
            scheduler = DDIMScheduler.from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
            scheduler_config = os.path.join(
                pretrained_model_name_or_path, "scheduler", "scheduler_config.json"
            )
            with open(scheduler_config, "r") as f:
                scheduler_config = json.load(f)
                scheduler_type = scheduler_config["_class_name"]
            scheduler = getattr(diffusers, scheduler_type).from_pretrained(
                pretrained_model_name_or_path, subfolder="scheduler"
            )
        else:
            scheduler_type = scheduler.pop("type", "FlowMatchEulerDiscreteScheduler")
            scheduler = getattr(diffusers, scheduler_type)(**scheduler)

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            use_fast=False,
        )

        if vae is None:
            vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        else:
            vae = AutoencoderKL.from_pretrained(vae)

        if unet is None:
            unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet"
            )
        else:
            use_safetensors = os.path.exists(
                os.path.join(unet, "diffusion_pytorch_model.safetensors")
            )
            unet = UNet2DConditionModel.from_pretrained(unet, use_safetensors=use_safetensors)

        # Adapt input layers
        if self.sparse_depth_type is None:
            if unet.config["in_channels"] != 8:
                self._replace_unet_conv_in(unet, multiple=2)
                assert unet.config["in_channels"] == 8
        else:
            if unet.config["in_channels"] != 12:
                self._replace_unet_conv_in(unet, multiple=3)
                assert unet.config["in_channels"] == 12

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.train_scheduler = deepcopy(self.scheduler)

        self.unet.enable_xformers_memory_efficient_attention()

        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)

        self.empty_text_embed = None

    def _replace_unet_conv_in(self, unet, multiple=2):
        # replace the first layer to accept 8 in_channels
        _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, multiple, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight = _weight / multiple
        # new conv_in channel
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in = torch.nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        unet.config["in_channels"] = unet.config["in_channels"] * multiple
        logging.info(f"Unet config is updated, in_channels = {unet.config['in_channels']}")

    def load_checkpoint(self, ckpt_path):
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.unet.load_state_dict(torch.load(_model_path, map_location="cpu", weights_only=False))
        logging.info(f"UNet parameters are loaded from {_model_path}")

    def get_train_parameters(self):
        return self.unet.parameters()

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.vae.to(device=accelerator.device)
        self.text_encoder.to(device=accelerator.device, dtype=weight_dtype)
        self.unet.to(device=accelerator.device, dtype=weight_dtype)

        self.unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        return optimizer, train_dataloader, lr_scheduler

    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(self.text_encoder.device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]
        return text_embeddings

    def encode(self, sample, sample_posterior=False):
        if sample.dtype != self.vae.dtype:
            sample = sample.to(self.vae.dtype)
        if sample.shape[1] == 1:
            sample = sample.repeat(1, 3, 1, 1)
        if sample_posterior:
            latents = self.vae.encode(sample).latent_dist.sample()
        else:
            latents = self.vae.encode(sample).latent_dist.mode()
        model_input = latents * self.vae.config.scaling_factor
        return model_input

    def decode_latents(self, latents):
        if latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def decode_depth(self, latents):
        if latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents
        depth = self.vae.decode(latents).sample
        depth = depth.mean(dim=1, keepdim=True)
        depth = (depth / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        depth = depth.cpu().float()
        return depth

    def train_step(self, batch):
        self.unet.train()

        device = self.unet.device
        generator = batch.get("generator", None)

        image = batch["image"].to(device)
        batch_size = image.shape[0]

        depth = batch[self.depth_type].to(device)
        depth_valid_mask = batch[self.depth_mask_type].to(device)

        if self.sparse_depth_type is not None:
            sparse_depth = batch[self.sparse_depth_type].to(device)

        depth_invalid_mask = ~depth_valid_mask
        valid_mask_down = ~torch.max_pool2d(depth_invalid_mask.float(), 8, 8).bool()
        valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))

        with torch.no_grad():
            image_latent = self.encode(image)  # [B, 4, h, w]
            depth_latent = self.encode(depth)  # [B, 4, h, w]

            if self.sparse_depth_type is not None:
                sparse_depth_latent = self.encode(sparse_depth)  # [B, 4, h, w]

        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[indices].to(device=device)

        # Sample noise
        if self.apply_multi_res_noise:
            strength = self.mr_noise_strength
            if self.annealed_mr_noise:
                # calculate strength depending on t
                strength = strength * (timesteps / self.train_scheduler.config.num_train_timesteps)
            noise = multi_res_noise_like(
                depth_latent,
                strength=strength,
                downscale_strategy=self.mr_noise_downscale_strategy,
                generator=generator,
                device=device,
            )
        else:
            noise = torch.randn(
                depth_latent.shape,
                device=device,
                generator=generator,
            )  # [B, 4, h, w]

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.train_scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = self.train_scheduler.timesteps.to(device)
            timesteps = timesteps.to(device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        sigmas = get_sigmas(timesteps, n_dim=depth_latent.ndim, dtype=depth_latent.dtype)
        noisy_depth_latent = (1.0 - sigmas) * depth_latent + sigmas * noise

        # Text embedding
        with torch.no_grad():
            if self.empty_text_embed is None:
                self.empty_text_embed = self.encode_prompt("")

        text_embed = self.empty_text_embed.repeat((batch_size, 1, 1))  # [B, 77, 1024]

        # Concat image and depth latents
        if self.sparse_depth_type is None:
            cat_latents = torch.cat([image_latent, noisy_depth_latent], dim=1)  # [B, 8, h, w]
        else:
            cat_latents = torch.cat(
                [image_latent, sparse_depth_latent, noisy_depth_latent], dim=1
            )  # [B, 12, h, w]

        # NOTE: 使用 deepspeed 后，即使没有显示设置 self.unet dtype，它也会变为 accelerator 的 精度
        if hasattr(self.unet, "dtype") and self.unet.dtype != cat_latents.dtype:
            cat_latents = cat_latents.to(self.unet.dtype)
            text_embed = text_embed.to(self.unet.dtype)

        # Predict the noise residual
        model_pred = self.unet(cat_latents, timesteps, text_embed).sample  # [B, 4, h, w]
        if torch.isnan(model_pred).any():
            logging.warning("model_pred contains NaN.")

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )

        # flow matching loss
        target = noise - depth_latent

        # Masked latent loss
        valid_mask_down = valid_mask_down.float()
        batch_valid_nums = valid_mask_down.reshape(batch_size, -1).sum(-1)
        batch_nan_mask = batch_valid_nums == 0
        batch_valid_nums = batch_valid_nums.clip(1)

        latent_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
        latent_loss = (
            (latent_loss * valid_mask_down).reshape(batch_size, -1) / batch_valid_nums.unsqueeze(-1)
        ).sum(-1)
        latent_loss = latent_loss * (~batch_nan_mask).float()

        loss = latent_loss.mean()

        return loss, None

    @torch.inference_mode()
    def infer(
        self,
        image,
        generator=None,
        show_pbar=False,
        meta_data=None,
        **kwargs,
    ):
        self.unet.eval()

        device = self.unet.device

        image = image.to(device)
        depth_gt = kwargs[self.depth_gt_type].squeeze().numpy()
        depth_gt_valid_mask = kwargs[self.depth_gt_mask_type].squeeze().numpy()
        depth_gt_valid_mask = (
            depth_gt_valid_mask & (depth_gt >= self.min_depth) & (depth_gt <= self.max_depth)
        )

        # Encode the input image to latent space
        image_latent = self.encode(image)

        # Encode the sparse depth to latent space
        if self.sparse_depth_type is not None:
            sparse_depth = kwargs[self.sparse_depth_type].to(device)
            sparse_depth_latent = self.encode(sparse_depth)

        # Initialize depth latent with noise
        depth_latent = torch.randn(
            image_latent.shape,
            device=device,
            dtype=image_latent.dtype,
            generator=generator,
        )

        # Set timesteps for the scheduler
        self.scheduler.set_timesteps(self.denoising_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Prepare batched empty text embedding
        if self.empty_text_embed is None:
            self.empty_text_embed = self.encode_prompt("")

        batch_empty_text_embed = self.empty_text_embed.repeat((image_latent.shape[0], 1, 1))

        # Denoising loop
        iterable = (
            tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc="Diffusion denoising",
            )
            if show_pbar
            else enumerate(timesteps)
        )

        for i, t in iterable:
            if self.sparse_depth_type is None:
                unet_input = torch.cat([image_latent, depth_latent], dim=1)
            else:
                unet_input = torch.cat([image_latent, sparse_depth_latent, depth_latent], dim=1)

            if hasattr(self.unet, "dtype") and self.unet.dtype != unet_input.dtype:
                unet_input = unet_input.to(dtype=self.unet.dtype)
                batch_empty_text_embed = batch_empty_text_embed.to(dtype=self.unet.dtype)

            # Predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample

            # Compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(
                noise_pred, t, depth_latent, generator=generator
            ).prev_sample

        # Decode the depth latent to the final depth map
        depth_pred = self.decode_depth(depth_latent)

        if self.match_input_res:
            depth_pred = resize(
                depth_pred,
                depth_gt.shape,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )

        depth_pred = depth_pred.squeeze().numpy()

        depth_pred_align = align_depth_least_square(
            gt_arr=depth_gt,
            pred_arr=depth_pred,
            valid_mask_arr=depth_gt_valid_mask,
            return_scale_shift=False,
            max_resolution=None,
        )

        depth_pred_align = np.clip(
            depth_pred_align,
            a_min=self.min_depth,
            a_max=self.max_depth,
        )

        return DepthOutput(depth_norm=depth_pred, depth_align=depth_pred_align)

    def save_checkpoint(self, accelerator, ckpt_dir):

        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        unwrapped_unet = accelerator.unwrap_model(self.unet)
        unwrapped_unet.save_pretrained(
            unet_path,
            safe_serialization=False,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        logging.info(f"UNet is saved to: {unet_path}")

        # Save Scheduler
        scheduler_path = os.path.join(ckpt_dir, "scheduler")
        self.train_scheduler.save_pretrained(
            scheduler_path,
            safe_serialization=False,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        logging.info(f"Scheduler is saved to: {scheduler_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def visualize(self, outputs, meta_data, out_dir):
        depth_pred_colored = colorize_depth_maps(
            outputs.depth_norm, 0, 1, cmap="turbo"
        )  # [3, H, W], value in (0, 1)
        save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.jpg")
        depth_pred_colored.save(save_path)

        logging.info(f"visualize save: {save_path}")

    def save_output(self, outputs, meta_data, out_dir):
        save_path = os.path.join(out_dir, f"detph_{meta_data['data_idx'][0]:06d}.npy")
        np.save(save_path, outputs.depth_align)

        data_path = meta_data["data_path"][0]
        new_data_path = os.path.join(out_dir, f"data_info_with_depth.json")
        assert new_data_path != data_path
        if not os.path.exists(new_data_path):
            shutil.copy(data_path, new_data_path)

        with open(new_data_path, "r") as f:
            data_info = json.load(f)
            assert (
                data_info["files"][meta_data["data_idx"]]["rgb"] == meta_data["data_info"]["rgb"][0]
            )
            data_info["files"][meta_data["data_idx"]]["pred_depth"] = save_path
            data_info["files"][meta_data["data_idx"]]["depth_scale"] = meta_data["depth_scale"][
                0
            ].item()

        with open(new_data_path, "w") as f:
            json.dump(data_info, f, indent=2)

        logging.info(f"output save: {save_path}")
