import importlib
import json
import logging
import warnings
from functools import partial
from numbers import Number
from pathlib import Path
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint
import torch.version
from accelerate import Accelerator
from huggingface_hub import hf_hub_download

from .. import utils3d
from ..utils.geometry_numpy import rigid_registration_ransac
from ..utils.geometry_torch import (
    gaussian_blur_2d,
    normalized_view_plane_uv,
    recover_focal_shift,
    roe_alignment_3d,
)
from ..utils.tools import timeit
from .utils import (
    unwrap_module_with_gradient_checkpointing,
    wrap_dinov2_attention_with_sdpa,
    wrap_module_with_gradient_checkpointing,
)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
    ):
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "silu":
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == "elu":
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode
            ),
            nn.GroupNorm(hidden_channels // 32 if norm == "group_norm" else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(
                hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


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
                # x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
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
            # output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
            output = [self.output_block[i](x) for i in range(len(self.output_block))]
        else:
            # output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)
            output = self.output_block(x)

        return output


class MoGeModel(nn.Module):
    image_mean: torch.Tensor
    image_std: torch.Tensor

    def __init__(
        self,
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
        criterions: dict = {},
        accelerator: Accelerator = None,
        **deprecated_kwargs,
    ):
        super(MoGeModel, self).__init__()
        if deprecated_kwargs:
            warnings.warn(
                f"The following deprecated/invalid arguments are ignored: {deprecated_kwargs}"
            )

        self.encoder = encoder
        self.remap_output = remap_output
        self.intermediate_layers = intermediate_layers
        self.trained_diagonal_size_range = trained_diagonal_size_range
        self.trained_area_range = trained_area_range
        self.output_mask = output_mask
        self.split_head = split_head
        self.criterions = (
            criterions["decoder_losses"] if criterions and "decoder_losses" in criterions else None
        )
        self.accelerator = accelerator

        # NOTE: We have copied the DINOv2 code in torchhub to this repository.
        # Minimal modifications have been made: removing irrelevant code, unnecessary warnings and fixing importing issues.
        hub_loader = getattr(importlib.import_module(".dinov2.hub.backbones", __package__), encoder)
        self.backbone = hub_loader(
            pretrained=True,
            weights_path="/mnt/personal/lx/code/depth_estimation/MoGe_training/ckpt/dinov2_vitl14_pretrain.pth",
        )
        dim_feature = self.backbone.blocks[0].attn.qkv.in_features

        self.head = Head(
            num_features=(
                intermediate_layers
                if isinstance(intermediate_layers, int)
                else len(intermediate_layers)
            ),
            dim_in=dim_feature,
            dim_out=3 if not output_mask else 4 if output_mask and not split_head else [3, 1],
            dim_proj=dim_proj,
            dim_upsample=dim_upsample,
            dim_times_res_block_hidden=dim_times_res_block_hidden,
            num_res_blocks=num_res_blocks,
            res_block_norm=res_block_norm,
            last_res_blocks=last_res_blocks,
            last_conv_channels=last_conv_channels,
            last_conv_size=last_conv_size,
        )

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if torch.__version__ >= "2.0":
            self.enable_pytorch_native_sdpa()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path, IO[bytes]],
        model_kwargs: Optional[Dict[str, Any]] = None,
        **hf_kwargs,
    ) -> "MoGeModel":
        """
        Load a model from a checkpoint file.

        ### Parameters:
        - `pretrained_model_name_or_path`: path to the checkpoint file or repo id.
        - `model_kwargs`: additional keyword arguments to override the parameters in the checkpoint.
        - `hf_kwargs`: additional keyword arguments to pass to the `hf_hub_download` function. Ignored if `pretrained_model_name_or_path` is a local path.

        ### Returns:
        - A new instance of `MoGe` with the parameters loaded from the checkpoint.
        """
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(
                pretrained_model_name_or_path, map_location="cpu", weights_only=True
            )
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                repo_type="model",
                filename="model.pt",
                **hf_kwargs,
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location="cpu", weights_only=True)
        model_config = checkpoint["model_config"]
        # NOTE: don't remap the output, different from the original moge
        # model_config['remap_output'] = False
        if model_kwargs is not None:
            model_config.update(model_kwargs)
        print(model_config)
        model = cls(**model_config)
        # model.load_state_dict(checkpoint['model'])
        # logging.critical(f"Loading backbone only")
        # backbone_dict = dict()
        # for k, v in checkpoint['model'].items():
        #     if k.startswith('backbone.'):
        #         backbone_dict[k[len('backbone.'):]] = v
        # model.backbone.load_state_dict(backbone_dict)
        return model

    @staticmethod
    def cache_pretrained_backbone(encoder: str, pretrained: bool):
        _ = torch.hub.load("facebookresearch/dinov2", encoder, pretrained=pretrained)

    def load_pretrained_backbone(self):
        "Load the backbone with pretrained dinov2 weights from torch hub"
        state_dict = torch.hub.load(
            "facebookresearch/dinov2", self.encoder, pretrained=True
        ).state_dict()
        self.backbone.load_state_dict(state_dict)

    # def enable_backbone_gradient_checkpointing(self):
    #     for i in range(len(self.backbone.blocks)):
    #         self.backbone.blocks[i] = wrap_module_with_gradient_checkpointing(self.backbone.blocks[i])

    def enable_pytorch_native_sdpa(self):
        for i in range(len(self.backbone.blocks)):
            self.backbone.blocks[i].attn = wrap_dinov2_attention_with_sdpa(
                self.backbone.blocks[i].attn
            )

    def forward(self, data: torch.Tensor, training=True) -> Dict[str, torch.Tensor]:
        # data: ['input', 'target', 'pointmap', 'intrinsics', 'filename', 'dataset', 'pad', 'data_type', 'sem_mask', 'normal']
        image = data["input"]
        raw_img_h, raw_img_w = image.shape[-2:]
        patch_h, patch_w = raw_img_h // 14, raw_img_w // 14

        image = (image - self.image_mean) / self.image_std

        # Apply image transformation for DINOv2
        image_14 = F.interpolate(
            image,
            (patch_h * 14, patch_w * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        # # Get intermediate layers from the backbone
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=mixed_precision):
        features = self.backbone.get_intermediate_layers(
            image_14, self.intermediate_layers, return_class_token=True
        )

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

        if self.remap_output == "linear" or self.remap_output == False:
            pass
        elif self.remap_output == "sinh" or self.remap_output == True:
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

        return_dict = {"prediction": points}
        if self.output_mask:
            return_dict["pred_mask"] = mask
        return_dict.update(data)
        if training:
            losses_dict = self.get_loss(return_dict)
            target = return_dict["target"]
        else:
            losses_dict = None
            target = None
        return return_dict["prediction"], losses_dict, return_dict["pred_mask"], target

    def pre_align_roe(self, prediction, target, mask):
        pts_pred = prediction[mask]
        pts_target = target[mask]
        # if mask is empty, return
        if pts_target.shape[0] == 0:
            logging.warning("Too few points in mask")
            return prediction, False

        sample_idx = np.random.choice(len(pts_pred), min(len(pts_pred), 400), replace=True)
        pts_pred = pts_pred[sample_idx]
        pts_target = pts_target[sample_idx]

        # filter top 10% points of z
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
            optimal_t = (
                pts_target.mean(dim=0, keepdim=True)
                - pts_pred.mean(dim=0, keepdim=True) * optimal_s
            )
            loss_scale = (
                torch.abs(pts_pred * optimal_s + optimal_t - pts_target).sum(dim=-1, keepdim=True)
                / pts_target[..., 2:]
            ).mean()

        if self.accelerator.is_main_process:
            logging.debug(f"s: {optimal_s}, loss: {loss_scale.item()}")

        prediction = prediction * optimal_s + optimal_t
        return prediction, True

    def get_loss(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # data: ['input', 'target', 'pointmap', 'intrinsics', 'filename', 'dataset', 'pad', 'data_type', 'sem_mask', 'normal', 'prediction', 'pred_mask']
        losses_dict = {}
        batches_data_type = np.array(data["data_type"])

        pred = data.pop("prediction")
        target = data.pop("pointmap")
        data["target"] = target
        mask = ((target[..., 2] > 0))[..., None]  # z should be > 0
        # mask = mask & (data['pred_mask'] > 0.5)[..., None].detach()
        # breakpoint()

        aligned_prediction = []
        for i in range(len(pred)):
            prediction_, success = self.pre_align_roe(pred[i], target[i], mask[i].squeeze(-1))
            aligned_prediction.append(prediction_)
            if torch.isnan(prediction_).any():
                logging.error("NAN", data["filename"][i])
                aligned_prediction[-1] = torch.zeros_like(prediction_)
            if success == False:
                logging.warning(f"Failed to align {data['filename'][i]}")
                # data['data_type'][0][i] = 'ignore'
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pred[i].detach().float().cpu().numpy().reshape([-1, 3]))
                # o3d.io.write_point_cloud('debug.ply', pcd)
                # pcd.points = o3d.utility.Vector3dVector(target[i].detach().float().cpu().numpy().reshape([-1, 3]))
                # o3d.io.write_point_cloud('debug_target.ply', pcd)
        data["prediction"] = torch.stack(aligned_prediction)

        # pred_norm = (torch.norm(pred, dim=-1, keepdim=True) * mask).flatten(1).sum(-1, keepdim=True) / (mask.flatten(1).sum(1, keepdim=True)+1e-6)
        # target_norm = (torch.norm(target, dim=-1, keepdim=True) * mask).flatten(1).sum(-1, keepdim=True) / (mask.flatten(1).sum(1, keepdim=True)+1e-6)
        # pred = pred * target_norm[..., None, None] / pred_norm[..., None, None]
        # target = target / target_norm[..., None, None]

        # logging.debug(f"pred_norm: {pred_norm.mean()}, target_norm: {target_norm.mean()}")

        data["target"] = target
        # data['prediction'] = pred
        data["prediction_raw"] = pred

        for loss_method in self.criterions:
            # new_mask = self.create_mask_as_loss(loss_method, mask, batches_data_type)
            new_mask = mask
            try:
                loss_tmp = loss_method(mask=new_mask, **data)
            except Exception as e:
                logging.error(e)
                logging.error(f"Failed to compute loss {data['filename']}")
                loss_tmp = torch.tensor(0.0, device="cuda")
            losses_dict[loss_method._get_name()] = loss_tmp

        total_loss = sum(losses_dict.values())
        losses_dict["total_loss"] = total_loss
        return losses_dict

    def create_mask_as_loss(self, loss_method, mask, batches_data_type):
        data_type_req = np.array(loss_method.data_type)[:, None]
        batch_mask = torch.tensor(np.any(data_type_req == batches_data_type, axis=0), device="cuda")
        new_mask = mask * batch_mask[:, None, None, None]
        return new_mask

    @torch.inference_mode()
    def infer(
        self,
        data: torch.Tensor,
        force_projection: bool = True,
        resolution_level: int = 9,
        apply_mask: bool = True,
        fov_x: Union[Number, torch.Tensor] = None,
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
        image = data["input"]
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

        return points, mask
