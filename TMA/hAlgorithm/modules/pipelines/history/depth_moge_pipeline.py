import importlib
import json
import logging
from numbers import Number
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hAlgorithm.modules.models.moge.model.residual_conv_block import ResidualConvBlock
from hAlgorithm.modules.models.moge.model.utils import wrap_dinov2_attention_with_sdpa
from hAlgorithm.modules.models.moge.utils.geometry_torch import (
    normalized_view_plane_uv,
    roe_alignment_3d,
)
from hAlgorithm.utils import instantiate_from_config

from .outputs import DepthOutput


class Head(nn.Module):
    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: List[int],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal["group_norm", "layer_norm"] = "group_norm",
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
    ):
        super().__init__()

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_proj,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for _ in range(num_features)
            ]
        )

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch + 2, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch,
                            out_ch,
                            dim_times_res_block_hidden * out_ch,
                            activation="relu",
                            norm=res_block_norm,
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
            ]
        )

        self.output_block = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1] + 2,
                    dim_out_,
                    dim_times_res_block_hidden,
                    last_res_blocks,
                    last_conv_channels,
                    last_conv_size,
                    res_block_norm,
                )
                for dim_out_ in dim_out
            ]
        )

    def _make_upsampler(self, in_channels: int, out_channels: int):
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: Literal["group_norm", "layer_norm"],
    ):
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode="replicate",
            ),
        )

    def forward(self, hidden_states: torch.Tensor, image: torch.Tensor):
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        x = torch.stack(
            [
                proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, (feat, clstoken) in zip(self.projects, hidden_states)
            ],
            dim=1,
        ).sum(dim=1)

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(
                width=x.shape[-1],
                height=x.shape[-2],
                aspect_ratio=img_w / img_h,
                dtype=x.dtype,
                device=x.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = layer(x)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(
            width=x.shape[-1],
            height=x.shape[-2],
            aspect_ratio=img_w / img_h,
            dtype=x.dtype,
            device=x.device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)

        if isinstance(self.output_block, nn.ModuleList):
            output = [self.output_block[i](x) for i in range(len(self.output_block))]
        else:
            output = self.output_block(x)

        return output


class DepthMoGePipeline(nn.Module):
    def __init__(
        self,
        target_name: str = "pointmap",
        backbone_pretrain: str = None,
        head_pretrain: str = None,
        losses: List[Dict] = None,
        encoder: str = "dinov2_vitb14",
        intermediate_layers: Union[int, List[int]] = 4,
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        output_mask: bool = False,
        split_head: bool = False,
        remap_output: Literal[False, True, "linear", "sinh", "exp", "sinh_exp"] = "linear",
        res_block_norm: Literal["group_norm", "layer_norm"] = "group_norm",
        trained_diagonal_size_range: Tuple[Number, Number] = (600, 900),
        trained_area_range: Tuple[Number, Number] = (500 * 500, 700 * 700),
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        **kwargs,
    ):
        """
        Initializes the DepthMoGePipeline with specified parameters.

        :param target_name: The name of the target output in the batch dictionary.
        :param backbone_pretrain: Path to the pretrained backbone weights.
        :param head_pretrain: Path to the pretrained head weights.
        :param losses: A list of dictionaries specifying the loss configurations.
        :param encoder: The type of encoder to use, default is "dinov2_vitb14".
        :param intermediate_layers: Layers from which to extract features.
        :param dim_proj: Dimension of the projection layer.
        :param dim_upsample: Dimensions for upsampling layers.
        :param dim_times_res_block_hidden: Dimension multiplier for ResBlock hidden layers.
        :param num_res_blocks: Number of residual blocks.
        :param output_mask: Whether to output a mask along with points.
        :param split_head: Whether to split the head into separate outputs for points and mask.
        :param remap_output: Method to remap the output, can be "linear", "sinh", "exp", or "sinh_exp".
        :param res_block_norm: Normalization method for ResBlocks, can be "group_norm" or "layer_norm".
        :param trained_diagonal_size_range: Range of diagonal sizes used during training.
        :param trained_area_range: Range of areas used during training.
        :param last_res_blocks: Number of residual blocks at the end of the network.
        :param last_conv_channels: Number of channels in the final convolutional layer.
        :param last_conv_size: Kernel size of the final convolutional layer.
        """
        super(DepthMoGePipeline, self).__init__()

        self.target_name = target_name
        self.backbone_pretrain = backbone_pretrain
        self.head_pretrain = head_pretrain

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.trained_diagonal_size_range = trained_diagonal_size_range
        self.trained_area_range = trained_area_range
        self.output_mask = output_mask
        self.split_head = split_head

        # Load the DINOv2 backbone
        hub_loader = getattr(
            importlib.import_module(
                "hAlgorithm.modules.models.moge.model.dinov2.hub.backbones", __package__
            ),
            encoder,
        )
        self.backbone = hub_loader(pretrained=True, weights_path=self.backbone_pretrain)
        self.enable_pytorch_native_sdpa()

        dim_feature = self.backbone.blocks[0].attn.qkv.in_features

        # Initialize the prediction head
        self.head = Head(
            num_features=(
                intermediate_layers
                if isinstance(intermediate_layers, int)
                else len(intermediate_layers)
            ),
            dim_in=dim_feature,
            dim_out=(3 if not output_mask else 4 if output_mask and not split_head else [3, 1]),
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
        )
        if self.head_pretrain is not None:
            self.head.load_state_dict(
                torch.load(self.head_pretrain, map_location="cpu", weights_only=False)
            )

        # Register image mean and std as buffers
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.build_losses(losses)

    def enable_pytorch_native_sdpa(self):
        """
        Enables PyTorch's native scaled dot product attention (SDPA) for the backbone's attention layers.
        """
        for block in self.backbone.blocks:
            block.attn = wrap_dinov2_attention_with_sdpa(block.attn)

    def build_losses(self, loss_cfgs: List[Dict]):
        """
        Constructs the loss functions based on the provided configuration.

        :param loss_cfgs: A list of dictionaries containing loss configurations.
        """
        self.losses = []
        for loss_cfg in loss_cfgs:
            obj_cls = instantiate_from_config(
                loss_cfg
            )  # Assuming `build_from_cfg` is a function that builds loss objects
            self.losses.append(obj_cls)

    def load_checkpoint(self, ckpt_path):
        res = self.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=False), strict=False
        )
        logging.info(f"Model parameters are loaded from {ckpt_path}")
        logging.info(f"unexpected_keys: {res.unexpected_keys}")
        logging.info(f"missing_keys: {res.missing_keys}")

    def get_train_parameters(self):
        return self.parameters()

    def accelerator_prepare(self, accelerator, optimizer, lr_scheduler, train_dataloader):
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        logging.info(f"weight_dtype: {weight_dtype}")

        self.cuda(accelerator.device)
        self.device = accelerator.device
        self.dtype = weight_dtype

        if accelerator.deepspeed_plugin is not None:
            self, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self, optimizer, train_dataloader, lr_scheduler
            )
        else:
            self.device, self.head, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                self.device, self.head, optimizer, train_dataloader, lr_scheduler
            )
        self.accelerator = accelerator
        return optimizer, train_dataloader, lr_scheduler

    def save_checkpoint(self, accelerator, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")

        backbone = accelerator.unwrap_model(self.backbone)
        head = accelerator.unwrap_model(self.head)

        backbone_state = {f"backbone.{key}": val for key, val in backbone.state_dict().items()}
        head_state = {f"head.{key}": val for key, val in head.state_dict().items()}
        state_dict = {**backbone_state, **head_state}

        torch.save(state_dict, ckpt_path)
        logging.info(f"Model is saved to: {ckpt_path}")

    def share_step(self, image):
        """
        Processes the input image through the pipeline and returns the predicted points and mask.

        :param image: Input image tensor.
        :return: A tuple containing the predicted points and mask (if output_mask is True).
        """
        raw_img_h, raw_img_w = image.shape[-2:]
        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        # Normalize the image
        image = (image - self.image_mean) / self.image_std

        # Resize the image to match the DINOv2 input size
        image_14 = F.interpolate(
            image,
            (patch_h * 14, patch_w * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # Extract intermediate features from the backbone
        # features = self.backbone.get_intermediate_layers(image_14, self.intermediate_layers, return_class_token=True)
        features = self.backbone(image_14, self.intermediate_layers, return_class_token=True)

        # Predict points (and mask)
        output = self.head(features, image)
        if self.output_mask:
            if self.split_head:
                points, mask = output
            else:
                points, mask = output.split([3, 1], dim=1)
            points, mask = points.permute(0, 2, 3, 1), mask.squeeze(1)
        else:
            points = output.permute(0, 2, 3, 1)
            mask = None

        # Apply output remapping if necessary
        if self.remap_output == "linear" or self.remap_output is False:
            pass
        elif self.remap_output == "sinh" or self.remap_output is True:
            points = torch.sinh(points)
        elif self.remap_output == "exp":
            xy, z = points.split([2, 1], dim=-1)
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh_exp":
            xy, z = points.split([2, 1], dim=-1)
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)
        else:
            raise ValueError(f"Invalid remap output type: {self.remap_output}")

        return points, mask

    def train_step(self, batch):
        """
        Performs a single training step.

        :param batch: A dictionary containing the batch data.
        :return: The total loss for the batch.
        """
        self.train()

        image = batch["image"].to(self.accelerator.device)
        intrinsics = batch["intrinsics"].to(self.accelerator.device)

        prediction, pred_mask = self.share_step(image)

        # Create a mask where z > 0
        target = batch[self.target_name].to(self.accelerator.device)
        target = target.permute(0, 2, 3, 1)
        mask = target[..., 0] > 0

        # Align predictions to targets
        aligned_predictions = []
        for i in range(len(prediction)):
            pred, success = self.pre_align_roe(prediction[i], target[i], mask[i].squeeze(-1))
            aligned_predictions.append(pred)

            if torch.isnan(pred).any():
                logging.error(f"NAN detected in {batch['meta_data']['data_info']['rgb'][i]}")
                aligned_predictions[-1] = torch.zeros_like(pred)

            if not success:
                logging.warning(f"Failed to align {batch['meta_data']['data_info']['rgb'][i]}")

        data_dict = {
            "prediction_raw": prediction,
            "prediction": torch.stack(aligned_predictions),
            "target": target,
            "intrinsics": intrinsics,
        }

        losses = 0
        losses_dict = {}
        for loss_method in self.losses:
            loss = loss_method(mask=mask, **data_dict)

            losses += loss
            losses_dict[loss_method._get_name().lower()] = loss.item()

        return losses, losses_dict

    def pre_align_roe(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """
        Pre-aligns the predicted points to the target points using robust optimization.

        :param prediction: Predicted points.
        :param target: Target points.
        :param mask: Mask indicating valid points.
        :return: A tuple containing the aligned predictions and a boolean indicating success.
        """
        pts_pred = prediction[mask]
        pts_target = target[mask]

        if pts_target.shape[0] == 0:
            logging.warning("Too few points in mask")
            return prediction, False

        sample_idx = np.random.choice(len(pts_pred), min(len(pts_pred), 400), replace=True)
        pts_pred = pts_pred[sample_idx]
        pts_target = pts_target[sample_idx]

        upper = pts_pred[..., 2].float().quantile(0.8)
        lower = pts_pred[..., 2].float().quantile(0.05)
        mask = (pts_pred[..., 2] > lower) & (pts_pred[..., 2] < upper)
        pts_pred = pts_pred[mask]
        pts_target = pts_target[mask]

        if pts_pred.shape[0] == 0 or pts_target.shape[0] == 0:
            logging.warning("Too few points in filtered mask")
            return prediction, False

        optimal_s, optimal_t, l_star = roe_alignment_3d(pts_pred.detach(), pts_target.detach())
        loss_scale = (
            torch.abs(pts_pred * optimal_s + optimal_t - pts_target).sum(dim=-1, keepdim=True)
            / pts_target[..., 2:]
        ).mean()

        if loss_scale > 0.2 or optimal_s <= 0:
            logging.warning(f"Recalculating optimal_s {loss_scale.item()} {optimal_s}")
            pred_mean = pts_pred.mean(dim=0, keepdim=True)
            target_mean = pts_target.mean(dim=0, keepdim=True)
            optimal_s = (pts_target - target_mean).norm(dim=-1).mean() / (
                pts_pred - pred_mean
            ).norm(dim=-1).mean()
            optimal_t = target_mean - pred_mean * optimal_s
            loss_scale = (
                torch.abs(pts_pred * optimal_s + optimal_t - pts_target).sum(dim=-1, keepdim=True)
                / pts_target[..., 2:]
            ).mean()

        prediction = prediction * optimal_s + optimal_t
        return prediction, True

    @torch.inference_mode()
    def infer(
        self,
        image: torch.Tensor,
        force_projection: bool = True,
        resolution_level: int = 9,
        apply_mask: bool = True,
        fov_x: Union[Number, torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        User-friendly inference function

        ### Parameters
        - `image`: input image tensor of shape (B, 3, H, W) or (3, H, W)
        - `resolution_level`: the resolution level to use for the output point map in 0-9. Default: 9 (highest)
        - `force_projection`: if True, the output point map will be computed using the actual depth map. Default: True
        - `apply_mask`: if True, the output point map will be masked using the predicted mask. Default: True
        - `fov_x`: the horizontal camera FoV in degrees. If None, it will be inferred from the predicted point map. Default: None

        ### Returns

        A dictionary containing the following keys:
        - `points`: output tensor of shape (B, H, W, 3) or (H, W, 3).
        - `depth`: tensor of shape (B, H, W) or (H, W) containing the depth map.
        - `intrinsics`: tensor of shape (B, 3, 3) or (3, 3) containing the camera intrinsics.
        """
        self.eval()

        image = image.to(self.accelerator.device)
        prediction, pred_mask = self.share_step(image)

        if image.dim() == 3:
            omit_batch_dim = True
            image = image.unsqueeze(0)
        else:
            omit_batch_dim = False

        original_height, original_width = image.shape[-2:]
        area = original_height * original_width
        aspect_ratio = original_width / original_height

        min_area, max_area = self.trained_area_range
        expected_area = min_area + (max_area - min_area) * (resolution_level / 9)

        if expected_area != area:
            expected_width, expected_height = int(
                original_width * (expected_area / area) ** 0.5
            ), int(original_height * (expected_area / area) ** 0.5)
            image = F.interpolate(
                image,
                (expected_height, expected_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        data["input"] = image
        points, _, mask, _ = self.forward(data, training=False)

        if expected_area != area:
            points = F.interpolate(
                points.permute(0, 3, 1, 2),
                (original_height, original_width),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).permute(0, 2, 3, 1)
            mask = (
                None
                if mask is None
                else F.interpolate(
                    mask.unsqueeze(1),
                    (original_height, original_width),
                    mode="bilinear",
                    align_corners=False,
                    antialias=False,
                ).squeeze(1)
            )

        return DepthOutput(pointmap=points, mask=mask)
