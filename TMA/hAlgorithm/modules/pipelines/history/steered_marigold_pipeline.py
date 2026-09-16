# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from scipy.interpolate import griddata
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor // 8) * 8
    new_height = (int(original_height * downscale_factor) // 8) * 8

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class SteeredMarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        show_progress_bar: bool = True,
        sparse_depth=None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)
        resample_method_depth: InterpolationMode = get_tv_resample_method("nearest")

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            4 == rgb.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
            if sparse_depth is not None:
                sparse_depth = resize_max_res(
                    sparse_depth,
                    max_edge_resolution=processing_res,
                    resample_method=resample_method_depth,
                )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        _bs = batch_size

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
                sparse_depth=sparse_depth,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------

        depth_pred = depth_preds
        pred_uncert = None

        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred,
                input_size[-2:],
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        if sparse_depth is None:
            depth_pred = depth_pred.clip(0, 1)

        return MarigoldDepthOutput(
            depth=depth_pred,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    def distance_match(self, sparse_depth_map, dense_depth_map):
        X = sparse_depth_map
        Y = dense_depth_map

        mask = (X > 1e-8).float()

        X_mean = (mask * X).sum() / mask.sum()
        Y_mean = (mask * Y).sum() / mask.sum()

        # 加权方差和协方差
        covariance = ((mask * (X - X_mean)) * (Y - Y_mean)).sum()
        variance = (mask * (X - X_mean) ** 2).sum()

        # 计算 scale 和 shift
        scale = covariance / variance
        shift = Y_mean - scale * X_mean

        sparse_matched = scale * X + shift
        return sparse_matched, scale, shift

    # def align_scale_shift(self, pred: torch.tensor, target: torch.tensor):
    #     mask = target > 0
    #     target_mask = target[mask].cpu().numpy()
    #     pred_mask = pred[mask].cpu().numpy()
    #     if torch.sum(mask) > 10:
    #         scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
    #         if scale < 0:
    #             scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    #             shift = 0
    #     else:
    #         scale = 1
    #         shift = 0
    #     pred = torch.tensor(pred * scale + shift).cuda()
    #     return pred, scale

    def sparse_to_dense_depth(self, sparse_depth_map):
        """
        Converts a sparse depth map to a dense depth map using bilinear interpolation.

        Args:
        sparse_depth_map (np.ndarray): Sparse depth map where depth values are non-zero at specific locations.
        resolution (tuple): The resolution (height, width) of the desired dense depth map.

        Returns:
        np.ndarray: Dense depth map.
        """
        # Get non-zero indices (x, y positions of known depth values)
        sparse_depth_map = sparse_depth_map.squeeze().cpu().numpy()
        resolution = sparse_depth_map.shape
        y_indices, x_indices = np.nonzero(sparse_depth_map)

        # Get depth values at those positions
        known_depth_values = sparse_depth_map[y_indices, x_indices]

        # Create a grid of coordinates for the desired dense depth map
        x, y = np.meshgrid(np.arange(resolution[1]), np.arange(resolution[0]))

        # Perform bilinear interpolation

        dense_depth_map = griddata(
            (x_indices, y_indices), known_depth_values, (x, y), method="linear"
        )
        dense_depth_map = np.nan_to_num(dense_depth_map, nan=0.0)

        return torch.tensor(dense_depth_map).float().cuda().unsqueeze(0).unsqueeze(0)

    # def sparse_to_dense_depth_torch(self, depth):
    #     # 生成高斯核 (5x5)
    #     def gaussian_kernel(ksize=5, sigma=1.0):
    #         """生成高斯核"""
    #         ax = torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32).cuda()
    #         xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    #         kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    #         return kernel / kernel.sum()

    #     # 创建高斯核
    #     ksize = 5
    #     sigma = 1.0
    #     kernel = gaussian_kernel(ksize, sigma).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

    #     # 生成掩码，标记非 0 的有效值
    #     mask = (depth > 0).float()

    #     # 分别对深度图和掩码进行高斯卷积
    #     filtered_sum = F.conv2d(depth, kernel, padding=ksize // 2)
    #     valid_count = F.conv2d(mask, kernel, padding=ksize // 2)

    #     # 避免除以 0
    #     valid_count[valid_count == 0] = 1
    #     filtered_result = filtered_sum / valid_count

    #     # 保留原始的 0 值（无效值）
    #     filtered_result[mask != 0] = depth[mask != 0]
    #     return filtered_result

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        sparse_depth: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        if sparse_depth is not None:
            sparse_depth = sparse_depth.to(device).repeat(rgb_in.shape[0], 1, 1, 1)
            cv2.imwrite("sparse_depth.png", sparse_depth.squeeze().cpu().numpy() * 20000)
            cv2.imwrite(
                "origin_image.png",
                (
                    (rgb_in.permute(2, 3, 1, 0).squeeze().cpu().numpy()[:, :, ::-1] + 1) / 2 * 255
                ).astype(np.uint8),
            )
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1)).to(
            device
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        # if sparse_depth is not None:
        #     random_selected_indices = (self.sample_points_from_sparse_depth_map(sparse_depth.clone(), 13) != 0).cuda()
        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # this order is important

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            output = self.scheduler.step(noise_pred, t, depth_latent, generator=generator)
            depth_latent = output.prev_sample

            def gaussian_kernel(size, sigma):
                """生成一个高斯核"""
                kernel = torch.Tensor(size, size)
                mean = size // 2
                sum_val = 0.0

                for x in range(size):
                    for y in range(size):
                        x = torch.tensor(x)
                        y = torch.tensor(y)
                        kernel[x, y] = torch.exp(
                            -((x - mean) ** 2 + (y - mean) ** 2) / (2 * sigma**2)
                        )
                        sum_val += kernel[x, y]

                kernel /= sum_val
                return kernel

            kernel = gaussian_kernel(27, 1.0).unsqueeze(0).unsqueeze(0).cuda()

            if sparse_depth is not None:
                if isinstance(self.scheduler, DDIMScheduler):
                    origin_latent = output.pred_original_sample
                else:
                    origin_latent = output.denoised
                clean_sample = self.decode_depth(origin_latent)
                clean_sample = (clean_sample + 1) / 2
                clean_sample = clean_sample.clip(0, 1)
                sparse_depth_matched, scale, shift = self.distance_match(
                    sparse_depth.clone(), clean_sample.clone()
                )

                indices = sparse_depth > 1e-8

                f1_map = torch.zeros_like(clean_sample)
                f1_map[indices] = clean_sample[indices]

                f2_map = torch.zeros_like(clean_sample)
                f2_map[indices] = sparse_depth_matched[indices]

                f2_for_sample = F.conv2d(f2_map, kernel, padding=13)
                indices_for_sample = f2_for_sample == 0

                # Random select for sample x0
                # true_indices = torch.nonzero(indices_for_sample).squeeze()
                # sampled_indices = true_indices[torch.randperm(len(true_indices))[:-13600]]
                # modified_indices_for_sample = indices_for_sample.clone()
                # modified_indices_for_sample[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2], sampled_indices[:, 3]] = False

                f2_map[indices_for_sample] = clean_sample[indices_for_sample]
                f1_map[indices_for_sample] = clean_sample[indices_for_sample]

                interploted_f1 = self.sparse_to_dense_depth(f1_map)
                interploted_f2 = self.sparse_to_dense_depth(f2_map)
                # cv2.imwrite("inter1.png", (interploted_f1.squeeze().cpu().numpy() * 65535).astype(np.uint16))
                # cv2.imwrite("inter2.png", (interploted_f2.squeeze().cpu().numpy() * 65535).astype(np.uint16))

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                lambda_weight = 0.1 * beta_prod_t ** (0.5)

                # res_map = (interploted_f2 - interploted_f1).squeeze().detach().cpu().numpy()
                # os.makedirs("logs", exist_ok=True)
                # cv2.imwrite(f"logs/res_{i}.png", ((res_map + 1)/2 * 65535).astype(np.uint16))
                clean_sample_adjust = (clean_sample + interploted_f2 - interploted_f1).clip(
                    0, 1
                ) * 2 - 1
                depth_latent = depth_latent + lambda_weight * (
                    self.encode_rgb(clean_sample_adjust.repeat(1, 3, 1, 1)) - origin_latent
                )
                self.encode_rgb(self.decode_depth(origin_latent).repeat(1, 3, 1, 1))

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
        if sparse_depth is not None:
            _, scale, shift = self.distance_match(sparse_depth.clone(), depth.clone())
            depth = (depth - shift) / scale

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
